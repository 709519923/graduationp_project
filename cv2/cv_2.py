# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:01:47 2021

@author: 70951
"""
import cv2
import numpy as np
file_path = 'C:\\Users\\70951\\Desktop\\python_experiment\\cv2'
'''水平投影'''
def getHProjection(image):
    hProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    (h,w)=image.shape 
    #长度与图像高度一致的数组
    h_ = [0]*h
    #循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y,x] == 255:
                h_[y]+=1
    #绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y,x] = 255
    cv2.imshow('hProjection2',hProjection)
 
    return h_
 
if __name__ == "__main__":
    #读入原始图像
    origineImage = cv2.imread(file_path + "/material_library/machine_character.jpg")
    # 图像灰度化    
    #image = cv2.imread('test.jpg',0)
    image = cv2.cvtColor(origineImage,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',image)
    # 将图片二值化
    retval, img = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('binary',img)
    #水平投影
    H = getHProjection(img)


#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt
file_path = 'C:\\Users\\70951\\Desktop\\python_experiment\\cv2'

'''水平投影'''
def getHProjection(image):
    hProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    (h,w)=image.shape 
    #长度与图像高度一致的数组
    h_ = [0]*h
    #循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y,x] == 255:
                h_[y]+=1
    #绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y,x] = 255
    cv2.imshow('hProjection2',hProjection)
 
    return h_

'''垂直投影'''
def getVProjection(image):
    vProjection = np.zeros(image.shape,np.uint8);
    #图像高与宽
    (h,w) = image.shape
    #长度与图像宽度一致的数组
    w_ = [0]*w
    #循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y,x] == 255:
                w_[x]+=1
    #绘制垂直平投影图像
    for x in range(w):
        for y in range(h-w_[x],h):
            vProjection[y,x] = 255
    #cv2.imshow('vProjection',vProjection)
    return w_
 
if __name__ == "__main__":
    #读入原始图像
    origineImage = cv2.imread('C:/Users/70951/Desktop/python_experiment/cv2/material_library/test1.jpg')
    # 图像灰度化    
    #image = cv2.imread('test.jpg',0)
    image = cv2.cvtColor(origineImage,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',image)
    image = cv2.blur(image, (2, 2))
    #去噪
   # image = cv2.GaussianBlur(image, (3, 3), 0)
    cv2.imshow('gussian  blur',image)
    # 将图片二值化
    retval, img = cv2.threshold(image,120,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('binary',img)
    # # 腐蚀
    # kernel = np.ones((2, 2), np.uint8)  # 卷积核
    # img_erode = cv2.erode(img ,kernel,iterations=1)
    # cv2.imshow('erode',img_erode)
    # dilation = cv2.dilate(img_erode, kernel, iterations=5)  # 膨胀
    # cv2.imshow('dilate',dilation)


    # # 闭运算
    # closing = cv2.morphologyEx(img_erode, cv2.MORPH_CLOSE, kernel)  # 闭运算
    # cv2.imshow('closing',closing)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)  # 闭运算
    # cv2.imshow('closing1',closing)
    #图像高与宽
    (h,w)=img.shape 
    Position = []
    #水平投影
    H = getHProjection(img)
 
    start = 0
    H_Start = []
    H_End = []
    #根据水平投影获取垂直分割位置
    for i in range(len(H)):
        if H[i] > 0 and start ==0:
            H_Start.append(i)
            start = 1   #投影起点标记
        if H[i] <= 0 and start == 1:
            H_End.append(i)
            start = 0   #投影终点标记
    #分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(H_Start)):
        #获取行图像
        cropImg = img[H_Start[i]:H_End[i], 0:w]   # 冒号表示这一段，范围的意思
        #cv2.imshow('cropImg',cropImg)
        #对行图像进行垂直投影
        W = getVProjection(cropImg)
        Wstart = 0
        Wend = 0
        W_Start = 0
        W_End = 0
        for j in range(len(W)):
            if W[j] > 0 and Wstart ==0:
                W_Start =j
                Wstart = 1
                Wend=0
            if W[j] <= 0 and Wstart == 1:
                W_End =j
                Wstart = 0
                Wend=1
            if Wend == 1:
                Position.append([W_Start,H_Start[i],W_End,H_End[i]])
                Wend =0
#%%                
    # 1. 现在是要写出宽度统计 yes
    # 2. 小于1/15的面积的不要，减少噪声 
    # 3. 宽度大于3的，并且小于1/2max的，让其与右边的这个合并, 其他照常输出,解决 yes
    # 4. 每次的截切宽度，必须以行切割之后每行的宽度为准，而不是全部的宽度
    # 函数要求，传入 
    cutting_Width = [] #用于统计每一个字符识别框的宽度
    cutting_image = [] #裁剪下来的图片
    Position_New = [] #字符位置数组，New表示已经加入过滤算法将偏旁之类的识别框合并
    flag = -1
    for i in range(len(Position)):
        print(Position[i][2] - Position[i][0])
        cutting_Width.append(Position[i][2] - Position[i][0])
    plt.hist(cutting_Width) 
    for i in range(len(Position)):
        if flag == i:
            continue
        if cutting_Width[i] > 3 and cutting_Width[i] < 1/2 * max(cutting_Width): #这个1/2可能可以改大一点，如3/4
            #这个主要是要得到i，i可以给到Position  
            #Position[0][0]是左上角，第一个索引表示第几个框框
            if i+1<len(Position):
                Position[i][2] =  Position[i+1][2]
            Position_New.append(Position[i])
            flag = i+1
           # print('i = ' , i)
        else: 
            Position_New.append(Position[i])
           # print('i = ' , i)
    #根据确定的位置分割字符
    # for m in range(len(Position)):
    #     cv2.rectangle(origineImage, (Position[m][0],Position[m][1]), (Position[m][2],Position[m][3]), (255 ,0 ,0), 1)
    # #利用算法过滤之后分割字符
    # for m in range(len(Position_New)):
    #     cv2.rectangle(origineImage, (Position_New[m][0],Position_New[m][1]), (Position_New[m][2],Position_New[m][3]), (0 ,0 ,255), 1)    
    # cv2.imshow('image',origineImage)
    #写图片数据
    for m in range(len(Position_New)):
        img111 = origineImage[Position_New[m][1]:Position_New[m][3], Position_New[m][0]:Position_New[m][2]]
        #cv2.imshow('ak', img111)
        #cv2.imwrite('./cutting image/cutting image'+str(m)+'.jpg',origineImage[Position_New[m][1]:Position_New[m][3],Position_New[m][0]:Position_New[m][2]])
    for m in range(len(Position_New)):
        cutting_image.append(cv2.resize(origineImage[Position_New[m][1]:Position_New[m][3],Position_New[m][0]:Position_New[m][2]],(64,64),))  # 为图片重新指定尺寸
    cv2.waitKey(0)
    
#%% 
#调整图片大小,要取消边框的画，上面的程序就不能话框，或者用新的图片代替
#调用模型，开始预测
import matplotlib.pyplot as plt
cutting_image[1] = cv2.cvtColor(cutting_image[1],cv2.COLOR_BGR2GRAY)
plt.imshow(cutting_image[1],cmap ='gray')
t, cutting_image[1] = cv2.threshold(cutting_image[1],120,255,cv2.THRESH_BINARY_INV)
plt.imshow(cutting_image[1],cmap ='gray')
#cutting_image[1] = np.reshape(cutting_image[1], (64,64)).astype(np.uint8)
#%%
#图片反色
def inverse_color(image):

    height,width = image.shape
    img2 = image.copy()

    for i in range(height):
        for j in range(width):
            img2[i,j] = (255-image[i,j]) 
    return img2

import tensorflow as tf
network = tf.keras.models.load_model('C:/Users/70951/Desktop/python_experiment/cv2/model/model.h5')
network.summary()
#cutting_image = np.array(cutting_image)
cutting_image[1] = inverse_color(cutting_image[1])
cutting_image[1] = np.reshape(cutting_image[1], (1,64,64,1)).astype(np.float16)
a = network.predict(cutting_image[1])
a = np.argmax(a)
code_to_character = ['零', '一', '二', '三', '四', '五', '六', '七',
                 '八', '九', '十', '百', '千', '万', '亿']
print(code_to_character[a])
type(cutting_image)
