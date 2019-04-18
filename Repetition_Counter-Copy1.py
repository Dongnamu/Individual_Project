#!/usr/bin/env python
# coding: utf-8

# In[7]:


from pprint import pprint
from matplotlib import pyplot as plt
import cv2
import copy
import operator
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.optimizers import Adam
from time import time
import json
import os
import warnings
import math

warnings.filterwarnings(action='ignore')


# In[8]:


def generateRandomPoint():
    return np.random.randint(1080, size = (25, 2))


# In[9]:


def generateRandomLength():
    return np.random.randint(50, size = 1)


# In[10]:


def getNewLocations(index, distances, scaler, original, finalPoints, first):
    k = 0
    
    while k < 30:
        for j in range(len(distances)):
            if k == 30:
                break
                
            tmp = copy.deepcopy(original)
            
            for x in range(original.shape[0]):
                if index < 2:
                    tmp[x][index] = tmp[x][index] + (distances[j] * scaler)
                
                if index == 2:
                    tmp[x][0] = tmp[x][0] + distances[j]
                    tmp[x][1] = tmp[x][1] + distances[j]
                
                if index == 3:
                    tmp[x][0] = tmp[x][0] + distances[j]
                    tmp[x][1] = tmp[x][1] - distances[j]
                
                if index == 4:
                    tmp[x][0] = tmp[x][0] - distances[j]
                    tmp[x][1] = tmp[x][1] + distances[j]
                
                if index == 5:
                    tmp[x][0] = tmp[x][0] - distances[j]
                    tmp[x][1] = tmp[x][1] - distances[j]
                    
            tmp = StandardScaler().fit_transform(tmp)
            if not first:
                finalPoints = np.concatenate((finalPoints, tmp), axis = 0)
            else:
                finalPoints = copy.deepcopy(tmp)
                first = False
                
            k += 1

    return finalPoints, first


# In[11]:


def generateTrainingData():
    
    finalPoints = []
    y_points = []
    
    first = True
    
    frequencies = np.linspace(3, 15, 13, dtype=int)
    
    k = 0
    l  = 0
    
    for j in range(10000):
        original = generateRandomPoint()
        distance = generateRandomLength()
        
        l += 1
        
        for i in range(len(frequencies)):
            frequency = frequencies[i]

            div = int(math.floor(frequency / 2)) + frequency % 2
            
            first_half = np.linspace(0, distance, div, dtype = int)
            second_half = np.linspace(distance, 0, div, dtype = int)
            
            if frequency % 2 == 1:
                second_half = second_half[1:]
            else:
                first_half = first_half[:-1]

            distances = np.concatenate((first_half, second_half), axis = 0)            

            finalPoints, first = getNewLocations(0, distances, 1, original, finalPoints, first)

            finalPoints, first = getNewLocations(0, distances, -1, original, finalPoints, first)

            finalPoints, first = getNewLocations(1, distances, 1, original, finalPoints, first)

            finalPoints, first = getNewLocations(1, distances, -1, original, finalPoints, first)
            
            finalPoints, first = getNewLocations(2, distances, -1, original, finalPoints, first)

            finalPoints, first = getNewLocations(3, distances, -1, original, finalPoints, first)

            finalPoints, first = getNewLocations(4, distances, -1, original, finalPoints, first)

            finalPoints, first = getNewLocations(5, distances, -1, original, finalPoints, first)

            y_points = np.append(y_points, frequency)
            y_points = np.append(y_points, frequency)
            y_points = np.append(y_points, frequency)
            y_points = np.append(y_points, frequency)
            y_points = np.append(y_points, frequency)
            y_points = np.append(y_points, frequency)
            y_points = np.append(y_points, frequency)
            y_points = np.append(y_points, frequency)
                
#     return finalPoints.reshape(13 * l * 8, 30, 50, 1), y_points
    return finalPoints.tolist(), y_points
    


# In[12]:


x, y= generateTrainingData()


# In[ ]:


y.shape


# In[ ]:


x_dict = dict(enumerate(x))


# In[ ]:


print(x_dict[0])


# In[ ]:


with open('RepetitionData2.json', 'w') as fp:
    json.dump(x_dict, fp)


# In[ ]:


x = [1,2,4,5]
x = x[1:]
x


# In[ ]:




