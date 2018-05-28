#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wes Apr  10  2018

@author: jingyi
"""

from sdm import SupervisedDescentMethod


def main():
    sdm = SupervisedDescentMethod("train", with_file=True)
    sdm.train_landmarks()


if __name__ == "__main__":
    main()
