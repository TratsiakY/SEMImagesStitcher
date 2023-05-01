#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2022 Yauhen Tratsiak. All rights reserved.
# Authors: Yauhen Tratsiak <ytratsia@utk.edu>
# License: GPLv3 (GNU General Public License Version 3)
#          https://www.gnu.org/licenses/quick-guide-gplv3.html
#
# This file is part of TSL deconvolute software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

'''
This file contains small class for Scanning Electron Microscopy (SEM) images stitching. The class is primitive and should be 
improved. The main problem of SEM images stitching is high number of possible matches with high percentage of fail mathes 
among them. I implemented finding matches in small sections of possible overlap of images. In general, class has a lot of points 
of improvements. All current limitations(disadvantages) to improve the overal performance. It makes sense to add option of use optical
flow for better stitching perfection
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

class SEMImageStitcher():
    '''
    Class for stitching images
    '''
    def __init__(self, initfnames, sfname, br_corr = True, mask_distance = None, dist_between_features = None, crop_data = [None, None, None, None]) -> None:
        """Initialization of the class SEMImagesStitching. The sizes of all images are same!

        Args:
            initfnames (array, str): array of the filenames (including patches) to the images. The order is important. The files order
            should be the same as pictures should be stitched 
            sfname (str): filename for resulting file (stitched image)
            br_corr (bool, optional): It is True if the brightness of all images should be adjusted. Defaults to True.
            mask_distance (int, optional): The possible size of the overlaped regions on the two stiched images for finding matches.
            If None the whole image is used for finding matches. Defaults to None.
            dist_between_features (int, optional): The distance between features that we use an optimal distance for finding mathes. 
            If None the whole image is used for finding matches. 
            crop_data (list, optional): Array with data for images cropping. There are 4 parameters, [From_row, to_row, from_col, to_col]. 
            If None, the whole image is used. Defaults to [None, None, None, None]. The same cropping rule is applied for all 
            images in `initfnames`.
        """
        self.fnames = initfnames
        self.sfname = sfname
        self.crop_data = crop_data
        self.mask_distance = mask_distance
        self.dist_between_features = dist_between_features 
        self.br_corr = br_corr
        self._read_images()

    def _read_images(self): 
        """
        Read images using OpenCV library. The check on the file replace should be added here.
        """
        self.images = []   
        for image in self.fnames:
            try:
                img = cv.imread(image)
                self.images.append(self._crop_image(img))
            except:
                print('Troubles with reading ', image)
                sys.exit(-1)
    def _crop_image(self, img):
        """The method for cropping images.

        Args:
            img (array): OpenCV image that should be cropped

        Returns:
            array: Cropped image
        """
        row, col,_ = img.shape
        r1 = self.crop_data[0] if (self.crop_data[0] is not None and self.crop_data[0] <= row and self.crop_data[0] > 0) else 0
        r2 = self.crop_data[1] if (self.crop_data[1] is not None and self.crop_data[1] <= row  and self.crop_data[1] > r1) else row
        c1 = self.crop_data[2] if (self.crop_data[2] is not None and self.crop_data[2] <= col and self.crop_data[2] > 0) else 0
        c2 = self.crop_data[3] if (self.crop_data[3] is not None and self.crop_data[3] <= col  and self.crop_data[3] > c1) else col
        return img[r1:r2, c1:c2]

    def _calc_SIFT(self, img, active_side):
        """The method for finding features in the images. 

        Args:
            img (array): the image where the features should be found. The `mask_distance` is implemented to image
            active_side (str): Two possible parameters "left" and "right". The are used to show the active side of the image
            where we are looking for features.

        Returns:
            points, descriptors: The lists of keypoints and their descriptors
        """
        g_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cimg = g_img.copy()
        if self.mask_distance <= cimg.shape[1]:
            if active_side == 'right':
                cimg[:,:-self.mask_distance] = 0
            elif active_side == 'left':
                cimg[:,self.mask_distance:] = 0
            else:
                print("Active side parameter is incorrect!")
        else:
            print('The distance parameter is wrong!')
        sift = cv.SIFT_create()
        kp, descr = sift.detectAndCompute(cimg, None)
        return kp, descr
    
    def _calc_matches(self):
        """Calculating the distances between keypoints on the stitching images. The `dist_between_features` is implemented to reduce 
        the possible number of points

        Returns:
            array: The array of matching keypoints
        """
        bfm = cv.BFMatcher(cv.NORM_L1, crossCheck = True)
        matches = bfm.match(self.l_descr, self.r_descr)

        distances = np.array([x.distance for x in matches])
        
        i = np.where(distances < self.dist_between_features)[0]

        return np.array(matches)[i]

    def _evaluaute_homography(self):
        """Finding the homography matrix for matching points

        Returns:
            homography_matrix, mask, left_points, right_points: Homography matrix which is required for image transformation, 
            mask with information of valid points.
        """
        l_pts = np.float32([self.l_kp[m.queryIdx].pt for m in self.matches]).reshape(-1,1,2)
        r_pts = np.float32([self.r_kp[m.trainIdx].pt for m in self.matches]).reshape(-1,1,2)

        M, mask = cv.findHomography(l_pts, r_pts, cv.RANSAC,5.0)

        return M, mask, l_pts, r_pts

    def _combine_images(self):
        """Stitching two images using common points that we found. 

        Returns:
            array: stitched image
        """
        points0 = np.array([[0, 0], [0, self.img_r.shape[0]], [self.img_r.shape[1], self.img_r.shape[0]], [self.img_r.shape[1], 0]],
            dtype=np.float32).reshape((-1, 1, 2))
        points1 = np.array([[0, 0], [0, self.img_l.shape[0]], [self.img_l.shape[1], self.img_l.shape[0]], [self.img_l.shape[1], 0]],
            dtype=np.float32).reshape((-1, 1, 2))

        points2 = cv.perspectiveTransform(points1, self.M)
        points = np.concatenate((points0, points2), axis=0)
        print("Images perspective was calculated")

        [x_min, y_min] = (points.min(axis=0).ravel()).astype(np.int32)
        [x_max, y_max] = (points.max(axis=0).ravel()).astype(np.int32)

        h_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        output_img = cv.warpPerspective(self.img_l, h_translation.dot(self.M), (x_max - x_min, y_max - y_min))
        output_img[-y_min:self.img_r.shape[0] - y_min, -x_min:self.img_r.shape[1] - x_min] = self.img_r
        return output_img

    def _correct_brightness(self):
        """Automatic brightness correction of images

        Returns:
            array: images with corrected brightness
        """
        gray_im = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in self.images]
        img = []
        for im in gray_im:
            img.append(cv.cvtColor(cv.equalizeHist(im), cv.COLOR_GRAY2BGR))

        return np.array(img)
    
    def stitch(self):
        """The method available for user to start the batch process of images stitching in `initfnames`

        Returns:
            array: The result of stitching images in `initfnames`, image.
        """
        if self.br_corr:
            self.images = self._correct_brightness()

        self.img_l = self.images[0]

        for i in range(1, len(self.images)):
            print('The image ', i+1, ' from ', len(self.images),' is stictching')

            self.img_r = self.images[i]

            mcol = np.min([self.img_r.shape[1], self.img_l.shape[1]])
            
            if self.mask_distance is None or self.mask_distance <= 0 or self.mask_distance > mcol:
                self.mask_distance = mcol
            print(self.mask_distance)
            self.l_kp, self.l_descr = self._calc_SIFT(self.img_l, active_side='right')
            self.r_kp, self.r_descr = self._calc_SIFT(self.img_r, active_side='left')
            print('Feauteres were found ')

            if self.dist_between_features <= 0:
                self.dist_between_features = mcol 

            self.matches = self._calc_matches()
            print('Pairs were found')

            self.M, _, _, _ = self._evaluaute_homography()
            img_stitch = self._combine_images()
            self.img_l = img_stitch
        return img_stitch

if __name__ == "__main__":

    # Create a list of images that will be stitched. The resulting image is MP393_FF0000+MP393_FF0001+MP393_FF0002+MP393_FF0003
    fname1 = 'imgs/MP393_FF0000.tiff'
    fname2 = 'imgs/MP393_FF00011.tiff'
    fname3 = 'imgs/MP393_FF0002.tiff'
    fname4 = 'imgs/MP393_FF0003.tiff'
    sname = 'imgs/result.png'
    fnames = [fname1, fname2, fname3, fname4]

    # Create an object of SEMImageStitcher class with required parameters and run stitch() method
    img_stitch = SEMImageStitcher(fnames, sname, True, 400, 500, [None, 2047, None, None]).stitch()

    # Show the result
    plt.imshow(img_stitch)
    plt.imsave(sname, img_stitch, dpi = 600)
    plt.show()    