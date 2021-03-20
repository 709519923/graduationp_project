# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:11:10 2021

@author: 70951
"""
import imutils
import cv2
import pandas as pd

# 加载图片
file_path = 'C:\\Users\\70951\\Desktop\\python_experiment\\cv2'
image = cv2.imread('C:/Users/70951/Desktop/python_experiment/cv2/test.jpg')
image = cv2.resize(image, (500, 500))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 灰度
#blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 5x5的内核的高斯平滑
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1] # 阈值化，阈值化后形状被表示成黑色背景上的白色前景。
cv2.imshow("Image", thresh)
# 在阈值图像中查找轮廓
# 找到白色对应的边界点的集合
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("Image", cnts)
cnts = imutils.grab_contours(cnts)




# 计算轮廓中心
for c in cnts:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])


    # 在图像上绘制形状的轮廓和中心
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    # 展示图片
    cv2.imshow("Image", image)
    with open(file_path, 'w') as w_obj:
        w_obj.write(str(cX) +' ' + str(cY)+ "\n")
    print(cX, cY)
    cv2.waitKey(0)
