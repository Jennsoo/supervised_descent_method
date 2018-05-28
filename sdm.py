#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wes Apr  10  2018

@author: jingyi
"""

import numpy as np
import mat4py
from scipy import io
import cv2 as cv
import math
import time
import os
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.externals import joblib
import dlib
import copy

from loader import ImagesReader
from loader import PointsReader
from hog import HOG
from structure import Shape
from viewer import show_points

TRAIN_PTS_PATH = "data/300w/train"
TRAIN_IMAGE_PATH = "data/300w/train"

TEST_IMAGE_PATH = "data/300w/test"
TEST_PTS_PATH = "data/300w/test"


class SupervisedDescentMethod(object):
    """
    A class to perform supervised descent method
    """
    def __init__(self, train_or_test,
                 with_file,
                 iterates=3,
                 alpha=1000,
                 new_size=(400, 400),
                 extend_rate=4):
        self.train_or_test = train_or_test
        self.with_file = with_file
        self.iterates = iterates
        self.alpha = alpha
        self.new_size = new_size
        self.extend_rate = extend_rate

        self.prepare()

        self.detector = dlib.get_frontal_face_detector()

    def prepare(self):
        """ Doing preparations before processing """
        if self.with_file:
            self.images = self.__load_images()
            self.shapes = self.__load_shapes()
            self.cvt2gray()

    def cvt2gray(self):
        """ Converting all images to gray """
        grays = []
        for img in self.images:
            grays.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        self.grays = grays

    def __load_images(self):
        """ Loading images from png in the training set """
        if self.train_or_test == "train":
            print("Loading training images...")
            imgs = ImagesReader.read_images_dictionary(TRAIN_IMAGE_PATH, "png", gray=False)
        else:
            print("Loading testing images...")
            imgs = ImagesReader.read_images_dictionary(TEST_IMAGE_PATH, "png", gray=False)
        return imgs

    def __load_shapes(self):
        """ Loading shapes from pts in the training set """
        if self.train_or_test == "train":
            print("Loading training pts...")
            shapes = PointsReader.read_points_dictionary(TRAIN_PTS_PATH)
        else:
            print("Loading testing pts...")
            shapes = PointsReader.read_points_dictionary(TEST_PTS_PATH)
        return shapes

    def set_single_image(self, image):
        """ Only using one image as test """
        self.images = [image]
        self.grays = [cv.cvtColor(image, cv.COLOR_BGR2GRAY)]

    '''@staticmethod
    def __load_ground_truth_boxes():
        """ Loading ground truth boxes from mat file and return.
        Ground truth boxes are like (x0, y0, x1, y1)"""
        print("Loading training ground truth bounding boxes...")
        bbox = mat4py.loadmat(TRAIN_BOXES_PATH)['bounding_boxes']
        bbgt = [bbox[i]['bb_ground_truth'] for i in range(len(bbox))]
        return bbgt'''

    def __get_dlib_rect(self, image):
        """ Getting rect of face with haar cascade """
        rect = []
        #face_cascade = cv.CascadeClassifier("data/haarcascade_frontalface_default.xml")
        # haar rects
        #faces = face_cascade.detectMultiScale(image, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5))

        faces = self.detector(image, 1)
        return faces

    @staticmethod
    def __crop(image, rect, extend_rate):
        """ Cropping the face from images using ground truth boxes,
        and extend by (width/extend_rate, height/extend_rate)
        :param image: initial image
        :param rect: initial bounding box rect
        :param extend_rate: the extend rate of initial image
        ----------------------------------------------------
        :return extended rect based on initial rect
        """
        x0, y0, x1, y1 = rect
        # width, height of ground truth bounding box
        w = x1 - x0
        h = y1 - y0
        # initial size
        iheight, iwidth = image.shape[:2]
        new_x0 = max(x0 - w / extend_rate, 0)
        new_y0 = max(y0 - h / extend_rate, 0)
        new_x1 = min(x1 + w / extend_rate, iwidth)
        new_y1 = min(y1 + h / extend_rate, iheight)
        new_rect = (new_x0, new_y0, new_x1, new_y1)
        return new_rect

    @staticmethod
    def __resize(image, new_size):
        """ Resizing to a unified size (400*400)
        :param image: initial image
        :param new_size: the new size the image will resize to
        ------------------------------------------------------
        :return resized image
        """
        new_image = cv.resize(image, new_size)
        return new_image

    def __recompute_shape(self, image, shape, rect):
        """ Once image is changed, points coordinates should also be recomputed
        :param image: initial image
        :param shape: initial shape
        :param rect: initial bounding box
        ---------------------------------------------------
        :return recomputed shapes coordination by extension
        """
        x0, y0, x1, y1 = rect
        # compute resize rate
        height, width = image.shape[:2]
        scale_x = self.new_size[0] / width
        scale_y = self.new_size[1] / height
        # recompute shape
        for pt in shape.pts:
            pt.x = (pt.x - x0 + (x1 - x0) / self.extend_rate) * scale_x
            pt.y = (pt.y - y0 + (y1 - y0) / self.extend_rate) * scale_y

    def __crop_and_resize(self, rects):
        """ Merging all functions of crop and resize """
        for idx in range(len(self.grays)):
            #rect = self.bboxes[idx]
            rect = rects[idx]
            # get extend ground truth bounding box
            new_rect = self.__crop(self.grays[idx], rect, self.extend_rate)
            # crop image
            cropped = self.grays[idx][int(new_rect[1]): int(new_rect[3]), int(new_rect[0]): int(new_rect[2])]
            # resize image
            self.grays[idx] = self.__resize(cropped, self.new_size)
            # recompute shape
            self.__recompute_shape(cropped, self.shapes[idx], rect)

    def __get_mean_shape(self):
        """ Getting mean shape from all shapes """
        mean = self.shapes[0]
        for shape in self.shapes[1:]:
            mean += shape
        return mean / len(self.shapes)

    def __get_detection(self):
        """ Training face detection with Haar cascade """
        dlib_rects = []
        order_list = []
        print("Calculating face detectors of training images with haar cascade...")
        for idx in range(len(self.grays)):
            # dlib rects
            faces = self.__get_dlib_rect(self.grays[idx])
            # if haar detector returns rects, rects should be an instance of np.ndarray
            # otherwise, rects is an instance of tuple
            #if isinstance(faces, np.ndarray):
            if len(faces) > 0:
                # get center of ground truth shape,
                # add the shape to list whose center is within haar box.
                # because there are images with multiple faces in the training set
                cs = self.shapes[idx].get_centroid()
                # the distance between true shape center and haar rect center
                offset = []
                # traverse all faces detected by dlib, and pick out the closest to the true shape
                for fidx in range(len(faces)):
                    rect = faces[fidx]
                    cr = ((rect.left()+rect.right()) / 2, (rect.top()+rect.bottom()) / 2)
                    dist = math.sqrt((cs[0]-cr[0])**2 + (cs[1]-cr[1])**2)
                    offset.append(dist)
                # get the rect closest to the true shape
                cidx = offset.index(min(offset))
                #x, y, w, h = faces[cidx]
                dlib_rects.append((faces[cidx].left(), faces[cidx].top(), faces[cidx].right(), faces[cidx].bottom()))
                print("Already completed", idx+1, "images.")
                order_list.append(idx)
            else:
                print("Dlib detector gets no faces.")
        print("Done.")
        print(np.array(dlib_rects).shape[0], "images are valid.")
        return order_list, dlib_rects

    def __reload(self, orders):
        """ Reloading images and shapes according to the results from detection """
        self.images = [self.images[idx] for idx in orders]
        self.grays = [self.grays[idx] for idx in orders]
        self.shapes = [self.shapes[idx] for idx in orders]

    def train_landmarks(self):
        """ Training face landmarks with Lasso function """
        # crop and resize training images
        orders, detect_box = self.__get_detection()
        self.__reload(orders)
        self.__crop_and_resize(detect_box)

        hog = []
        shapes = []
        for idx in range(len(self.grays)):
            print("Calculating ", idx, "th HOG features of training images...")
            # get hog features
            hog_descriptor = HOG(self.grays[idx], self.shapes[idx])
            h = hog_descriptor.extract()
            hog.append(h)
            # get shape vector list
            s = Shape.get_vector(self.shapes[idx])
            shapes.append(s)

        # true hog features and true shapes
        hog_star = np.array(hog)
        shapes_star = np.array(shapes)

        # get mean shape as x0
        #pdm = PointDistributionModel(self.shapes)
        #x0 = pdm.mean
        x0 = self.__get_mean_shape().get_vector()
        shape_x = np.array([x0.tolist()]*len(self.grays))

        # parameters we need
        R = []
        b = []
        # training
        for i in range(self.iterates):
            # delta shape vector
            delta_x = shapes_star - shape_x
            # hog features of computed shapes
            hog_x = np.zeros_like(hog_star)
            for j in range(len(self.grays)):
                # get hog features
                hog_descriptor = HOG(self.grays[j], Shape.turn_back_to_point(shape_x[j]))
                h = hog_descriptor.extract()
                hog_x[j, :] = h

            # linear regression
            if self.alpha == 0:
                reg = LinearRegression(fit_intercept=False)
            else:
                #reg = LinearRegression()
                #reg = SVR()
                #reg = Ridge(alpha=self.alpha)
                reg = Lasso(alpha=self.alpha)
            print("Calculating with Linear Regression...")
            reg.fit(hog_x, delta_x)
            R.append(reg.coef_.T)
            b.append(reg.intercept_.T)

            shape_x = shape_x + np.matmul(hog_x, R[i]) + b[i]

        # x0 = x0.tolist()
        io.savemat("./data/train_sdm", {"R": R, "b": b, "i": x0})

    @staticmethod
    def evaluate(shape_current, shape_true):
        """ Evaluating results """
        return np.linalg.norm(shape_current-shape_true)/np.linalg.norm(shape_true)

    def test(self):
        """ Testing images with known mat files """
        if os.path.exists("./data/train_sdm.mat"):
            mat = io.loadmat("./data/train_sdm.mat")
            R = mat['R']
            b = mat['b']
            shape_x = mat['i']

            # accuracy = 0
            for idx, img in enumerate(self.grays):
                img = copy.deepcopy(self.grays[idx])
                shape_x = mat['i']
                faces = self.__get_dlib_rect(img)[0]
                rect = [faces.left(), faces.top(), faces.right(), faces.bottom()]
                # get extend ground truth bounding box
                new_rect = self.__crop(img, rect, self.extend_rate)
                # crop image
                cropped = img[int(new_rect[1]): int(new_rect[3]), int(new_rect[0]): int(new_rect[2])]
                # resize image
                img = self.__resize(cropped, self.new_size)
                # self.__recompute_shape(cropped, self.shapes[idx], rect)

                for i in range(1):
                    shape = Shape.turn_back_to_point(shape_x)
                    hog_descriptor = HOG(img, shape)
                    hog_x = hog_descriptor.extract()
                    shape_x = shape_x + np.matmul(hog_x, R[i, :]) + b[i, :]

                # accuracy += self.evaluate(shape_x, self.shapes[idx].get_vector())
                # print("Already completed", idx + 1, "images.")

                height, width = cropped.shape[:2]
                scale_x = self.new_size[0] / width
                scale_y = self.new_size[1] / height

                shape = Shape.turn_back_to_point(shape_x)
                for pt in shape.pts:
                    pt.x = pt.x / scale_x - (rect[2]-rect[0]) / self.extend_rate + rect[0]
                    pt.y = pt.y / scale_y - (rect[3]-rect[1]) / self.extend_rate + rect[1]

            # print("Accuracy:", accuracy/len(self.shapes))
                return shape
        return
