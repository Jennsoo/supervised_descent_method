#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wes Apr  10  2018

@author: jingyi
"""

import cv2 as cv

from sdm import SupervisedDescentMethod
from viewer import show_points


def main():
    sdm = SupervisedDescentMethod("test", with_file=False)

    image = cv.imread("data/test.png")
    sdm.set_single_image(image)
    shape = sdm.test()

    show_points(image, shape)
    cv.imshow("test", image)
    cv.waitKey()


if __name__ == "__main__":
    main()