# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:44:37 2021
这个是检测滤波滤除背景噪声的效果，感觉pyrMean这个均值漂移滤波效果比较好。
@author: 70951
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

origineImage = cv2.imread('C:/Users/70951/Desktop/python_experiment/cv2/material_library/test.jpg')
 # 图像灰度化    
 #image = cv2.imread('test.jpg',0)
image = origineImage
cv2.imshow('gray',image)
 #去噪
image1 = cv2.GaussianBlur(image, (3, 3), 0)
image2 = cv2.pyrMeanShiftFiltering(image, 10, 10)
image3 = cv2.blur(image, (2, 2))
cv2.imshow('gussian  blur',image1)
cv2.imshow('mean  blur',image2)
cv2.imshow('11111111',image3)
image1 = cv2.cvtColor(image1 ,cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image2 ,cv2.COLOR_BGR2GRAY)
image3 = cv2.cvtColor(image3 ,cv2.COLOR_BGR2GRAY)
 # 将图片二值化
retval1, img1 = cv2.threshold(image1,120,255,cv2.THRESH_BINARY_INV)
retval2, img2 = cv2.threshold(image2,120,255,cv2.THRESH_BINARY_INV)
retval3, img3 = cv2.threshold(image3,120,255,cv2.THRESH_BINARY_INV)
cv2.imshow('gussian',img1)
cv2.imshow('mean',img2)
cv2.imshow('1111111',img3)
 # # 腐蚀