import constants
from interfaces import DataLoaderInterface

import zope.interface
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from tqdm import tqdm 
import matplotlib.pyplot as plt
import os
import cv2
import imutils

@zope.interface.implementer(DataLoaderInterface)
class CellTypeDataLoader():
    def __init__(self) -> None:
        pass
        
    def findEdges(self, image):
        # find edges in image
        gray = cv2.GaussianBlur(image, (1, 1), 0)
        edged = cv2.Canny(gray, 100, 400)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        return edged

    def getImgContours(self, edged):
        # find contours in the edge map
        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x))
        return contours

    def getBoxes(self, contours, orig):
        # get the boxes
        boxes = []
        centers = []
        for contour in contours:
            box = cv2.minAreaRect(contour)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            (tl, tr, br, bl) = box
            if (dist.euclidean(tl, bl)) > 0 and (dist.euclidean(tl, tr)) > 0:
                boxes.append(box)
        return boxes

    def load_data(self):
        
        datasets = ['./dataset2-master/dataset2-master/images/TRAIN','./dataset2-master/dataset2-master/images/TEST' ]
        images = []
        labels = []
        imageNames = []

        # iterate through training and test sets
        count =0
        for dataset in datasets:

            # iterate through folders in each dataset
            for folder in os.listdir(dataset):

                if folder in ['EOSINOPHIL']: label = 0
                elif folder in ['LYMPHOCYTE']: label = 1
                elif folder in ['MONOCYTE']: label = 2
                elif folder in ['NEUTROPHIL']: label = 3

                # iterate through each image in folder
                for file in tqdm(os.listdir(os.path.join(dataset, folder))):

                    # get pathname of each image
                    img_path = os.path.join(os.path.join(dataset, folder), file)
                    

                    # Open 
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # add padding to the image to better detect cell at the edge
                    image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[198, 203, 208])
                    
                    #thresholding the image to get the target cell
                    image1 = cv2.inRange(image,(80, 80, 180),(180, 170, 245))
                    
                    # openning errosion then dilation
                    kernel = np.ones((3, 3), np.uint8)
                    kernel1 = np.ones((5, 5), np.uint8)
                    img_erosion = cv2.erode(image1, kernel, iterations=2)
                    image1 = cv2.dilate(img_erosion, kernel1, iterations=5)
                    
                    #detecting the blood cell
                    edgedImage = self.findEdges(image1)
                    edgedContours = self.getImgContours(edgedImage)
                    edgedBoxes =  self.getBoxes(edgedContours, image.copy())
                    if len(edgedBoxes)==0:
                        count +=1
                        continue
                    # get the large box and get its cordinate
                    last = edgedBoxes[-1]
                    max_x = int(max(last[:,0]))
                    min_x = int( min(last[:,0]))
                    max_y = int(max(last[:,1]))
                    min_y = int(min(last[:,1]))
                    
                    # draw the contour and fill it 
                    mask = np.zeros_like(image)
                    cv2.drawContours(mask, edgedContours, len(edgedContours)-1, (255,255,255), -1) 
                    
                    # any pixel but the pixels inside the contour is zero
                    image[mask==0] = 0
                    
                    # extract th blood cell
                    image = image[min_y:max_y, min_x:max_x]

                    if (np.size(image)==0):
                        count +=1
                        continue
                    # resize th image
                    image = cv2.resize(image, constants.image_size)

                    # Append the image and its corresponding label to the output
                    images.append(image)
                    labels.append(label)
                    imageNames.append(file)
        print(len(images), len(images), len(imageNames))
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')


        return images, labels, imageNames

