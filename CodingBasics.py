# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:08:56 2023

@author: Morteza
"""

import cv2
import numpy as np
  
# Open the image files.
img1_color = cv2.imread("C:/Users/Morteza/Desktop/YouTube/coding/07ronaldo.jpg")
print("The size of the smaller image is",img1_color.shape)

cv2.imshow('Smaller picture', img1_color) 

  
cv2.waitKey(0) 
cv2.destroyAllWindows()  

#%%
#Big Picture
import cv2
import numpy as np
  
# Open the image files.
img1_color = cv2.imread("C:/Users/Morteza/Desktop/YouTube/coding/07ronaldoBig.jpg")
print("The size of the bigger image is",img1_color.shape)

cv2.imshow('Bigger picture', img1_color) 

  
cv2.waitKey(0) 
cv2.destroyAllWindows()  

#%%
import numpy as np

mat1 = np.array([[1, 1, 1],[1 ,1, 1],[1, 1, 1]])
mat2 = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])

Out = np.matmul(mat1,mat2)
print ("The result of np.matmul is", Out)

Out = mat1*mat2
print ("The result of * is", Out)

#%%

import skimage.io
import cv2

img = skimage.io.imread('C:/Users/Morteza/Desktop/YouTube/coding/YouTube/PuR6m.png')
cv2.imwrite('C:/Users/Morteza/Desktop/YouTube/coding/YouTube/sample_color_corrected.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 
cv2.imwrite('C:/Users/Morteza/Desktop/YouTube/coding/YouTube/sample_Not_color_corrected.png',img)


img = cv2.imread('C:/Users/Morteza/Desktop/YouTube/coding/YouTube/PuR6m.png')  
cv2.imwrite('C:/Users/Morteza/Desktop/YouTube/coding/YouTube/sample_readIn_OpenCV.png',img)  