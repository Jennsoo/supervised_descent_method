#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  16  2018

@author: jingyi
"""

import math
import cv2 as cv
import numpy as np
import copy

from structure import Shape
from viewer import show_points


class HOG(object):
    """
    A class to get HOG features of an image
    """
    def __init__(self, image, shape,
                 cells_per_side=4,
                 pixels_per_cell=2,
                 cells_per_block=2,
                 block_stride=1,
                 bin_size=8):
        """ Cell size and block size are both square.
        ex. cell_size = (pixels_per_cell * pixels_per_cell)
        ex. block_size = (cells_per_block * cells_per_block)
        Block stride by cells """
        # normalize image
        self.image = self.__normalization(image)
        self.shape = shape
        # get gradient magnitude and angle
        self.gimage, self.gimage_mag, self.gimage_ang = self.__get_general_gradient(self.image)

        self.cells_per_side = cells_per_side
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_stride = block_stride
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size

    @staticmethod
    def __normalization(image):
        """ Normalizing the image, gamma = 0.5 """
        image = image.astype(float)
        return np.sqrt(image)

    @staticmethod
    def __get_general_gradient(image):
        """ Getting gradient info of magnitude and angle """
        dx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
        dy = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
        absx = cv.convertScaleAbs(dx)
        absy = cv.convertScaleAbs(dy)
        # gradient image
        gimage = cv.addWeighted(absx, 0.5, absy, 0.5, 0)
        # gradient magnitude
        gimage_mag = cv.addWeighted(dx, 0.5, dy, 0.5, 0)
        # gradient angle
        gimage_ang = cv.phase(dx, dy, angleInDegrees=True)
        return gimage, abs(gimage_mag), gimage_ang

    @staticmethod
    def __get_closest_bin(angle, angle_unit, bin_size):
        """ Given an angle, returning the closest bin it belongs to """
        idx = int(angle / angle_unit)
        mod = angle % angle_unit
        return idx % bin_size, (idx+1) % bin_size, mod

    @staticmethod
    def __get_cell_rect(point, pixels_per_cell, cells_per_side):
        """ Getting rect of the local image which is to be extracted """
        # left top coordinate of the extracted image
        lt_x = int(point.x - pixels_per_cell * cells_per_side)
        lt_y = int(point.y - pixels_per_cell * cells_per_side)
        # right bottom coordinate of the extracted image
        rb_x = int(point.x + pixels_per_cell * cells_per_side)
        rb_y = int(point.y + pixels_per_cell * cells_per_side)
        rect = (lt_x, lt_y, rb_x, rb_y)
        return rect

    def __get_cell_gradient(self, cell_mag, cell_ang):
        """ Getting gradient info per cell """
        # initialize to 0
        histogram = [0]*self.bin_size
        for row in range(cell_mag.shape[0]):
            for col in range(cell_mag.shape[1]):
                strength = cell_mag[row, col]
                angle = cell_ang[row, col]
                # which bin it is close to, use bilinear interpolation
                min, max, mod = self.__get_closest_bin(angle, self.angle_unit, self.bin_size)
                histogram[min] += (strength*(1-(mod/self.angle_unit)))
                histogram[max] += (strength*(mod/self.angle_unit))
        return histogram

    def extract(self):
        """ Extracting HOG features, to speed up the feature extraction,
        only use local features around landmarks """
        # extract features around landmarks
        cells = np.zeros((self.shape.num_pts,       # amounts of landmarks
                          self.cells_per_side*2,    # amounts of cells per row
                          self.cells_per_side*2,    # amounts of cells per col
                          self.bin_size))           # amounts of bins per cell
        for p, pt in enumerate(self.shape.pts):
            x0, y0, x1, y1 = self.__get_cell_rect(pt, self.pixels_per_cell, self.cells_per_side)
            # cv.rectangle(self.image, (x0, y0), (x1, y1), (255, 0, 0), 2)
            # local image to be extracted
            gimg_mag = self.gimage_mag[y0: y1, x0: x1]
            gimg_ang = self.gimage_ang[y0: y1, x0: x1]
            for i in range(cells.shape[1]):
                for j in range(cells.shape[2]):
                    # gradient magnitude per cell
                    cell_mag = gimg_mag[i*self.pixels_per_cell: (i+1)*self.pixels_per_cell,
                               j * self.pixels_per_cell: (j + 1) * self.pixels_per_cell]
                    # gradient angle per cell
                    cell_ang = gimg_ang[i * self.pixels_per_cell: (i + 1) * self.pixels_per_cell,
                               j * self.pixels_per_cell: (j + 1) * self.pixels_per_cell]
                    cells[p, i, j] = self.__get_cell_gradient(cell_mag, cell_ang)
        '''
        # global features
        # divide image with cells
        cells = np.zeros((int(height/self.pixels_per_cell),
                         int(width/self.pixels_per_cell),
                         self.bin_size))
        # get gradient info of every cell
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                # gradient magnitude per cell
                cell_mag = self.gimage_mag[i*self.pixels_per_cell: (i+1)*self.pixels_per_cell,
                           j*self.pixels_per_cell: (j+1)*self.pixels_per_cell]
                # gradient angle per cell
                cell_ang = self.gimage_ang[i*self.pixels_per_cell: (i+1)*self.pixels_per_cell,
                           j*self.pixels_per_cell: (j+1)*self.pixels_per_cell]
                cells[i, j] = self.__get_cell_gradient(cell_mag, cell_ang)
        '''
        # connect cells to blocks
        if self.cells_per_side + self.block_stride >= self.cells_per_side*2:
            assert "The block size or block stride are too big."
        # blocks per row per col
        n_blocks = int((self.cells_per_side * 2 - self.cells_per_block) / self.block_stride + 1)
        blocks = np.zeros((self.shape.num_pts, n_blocks, n_blocks, self.cells_per_block**2*self.bin_size))
        for p, pt in enumerate(self.shape.pts):
            for i in range(blocks.shape[1]):
                for j in range(blocks.shape[2]):
                    start_cell_row = i * self.block_stride
                    start_cell_col = j * self.block_stride
                    cell = cells[p,
                           start_cell_row: (start_cell_row+self.cells_per_block),
                           start_cell_col: (start_cell_col+self.cells_per_block), :]
                    block = cell.flatten()
                    blocks[p, i, j, :] = block
        # render on an image
        #hog_image = self.render(cells)
        return blocks.flatten()

    def render(self, cells):
        """ Rendering the HOG image """
        height, width = self.image.shape[:2]
        hog_image = np.zeros((height, width))
        # max magnitude of all
        max_mag = cells.max()
        # regard square cells as circles
        radius = self.pixels_per_cell / 2
        for p, pt in enumerate(self.shape.pts):
            for i in range(cells.shape[1]):
                for j in range(cells.shape[2]):
                    cell_grad = cells[p, i, j]
                    cell_grad /= max_mag
                    new_rect = self.__get_cell_rect(pt, self.pixels_per_cell, self.cells_per_side)
                    angle = 0
                    angle_unit = 360 / self.bin_size
                    for mag in cell_grad:
                        radian = math.radians(angle)
                        x0 = int(i * self.pixels_per_cell + radius * mag * math.cos(radian)) + new_rect[0]
                        y0 = int(j * self.pixels_per_cell + radius * mag * math.sin(radian)) + new_rect[1]
                        x1 = int(i * self.pixels_per_cell - radius * mag * math.cos(radian)) + new_rect[0]
                        y1 = int(j * self.pixels_per_cell - radius * mag * math.sin(radian)) + new_rect[1]
                        cv.line(hog_image, (x0, y0), (x1, y1), int(255 * math.sqrt(mag)))
                        angle += angle_unit
        '''
        # global features
        for i in range(cells.shape[1]):
            for j in range(cells.shape[2]):
                cell_grad = cells[i, j]
                cell_grad = cell_grad / max_mag
                angle = 0
                angle_unit = 360 / self.bin_size
                for mag in cell_grad:
                    radian = math.radians(angle)
                    x0 = int(i*self.pixels_per_cell + radius*mag*math.cos(radian))
                    y0 = int(j*self.pixels_per_cell + radius*mag*math.sin(radian))
                    x1 = int(i*self.pixels_per_cell - radius*mag*math.cos(radian))
                    y1 = int(j*self.pixels_per_cell - radius*mag*math.sin(radian))
                    cv.line(hog_image, (y0, x0), (y1, x1), int(255 * math.sqrt(mag)))
                    angle += angle_unit
        '''
        return hog_image
    