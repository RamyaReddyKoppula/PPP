# importing required packages

#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import IPython
import csv
import pandas as pd

import matplotlib #scientific plots
import matplotlib.pyplot as plt

import numpy as np # numerical programming

import sklearn #scikit-learn
import math #Mathematical functions
import pylab 
import time
import pprint as pp
from mpl_toolkits.mplot3d import Axes3D #we use this to plot 3d
from matplotlib.ticker import NullFormatter #No labels on the ticks.
from IPython.display import Image
#from __future__ import division
first_array=np.genfromtxt('data_want.csv',delimiter=',')
SFE_data = first_array
SFE_data = np.array(SFE_data, dtype='float')
#SFE-(1-error),each rows
lower = SFE_data[:,10]*(1 - SFE_data[:,11]) 
higher = SFE_data[:,10]*(1 + SFE_data[:,11])
lowhigh = np.column_stack((lower,higher))
SFE_data = np.column_stack((SFE_data,lowhigh))
print(SFE_data.shape)
print(SFE_data[0,:])
SFE_classes = np.zeros(SFE_data.shape[0])

SFE_classes[SFE_data[:,10] <= 20] = 1
SFE_classes[(SFE_data[:,10] > 20) & (SFE_data[:,10] <= 45)] = 2
SFE_classes[(SFE_data[:,10] > 45)] = 3

#check number of entries for each class 
print((SFE_classes == 1).sum())
print((SFE_classes == 2).sum())
print((SFE_classes == 3).sum())

#check if only the needed classes there
print((SFE_classes == 1).sum() + (SFE_classes == 2).sum() + (SFE_classes == 3).sum()) 
SFE_classes.shape
SFE_data.shape
data_SFE=np.genfromtxt('data_nor.csv',delimiter=',')
# save to csv file
np.savetxt('data_nor.csv', SFE_data[:,:10], delimiter=',')

#normalization and standardization
from sklearn import preprocessing
#normalization = preprocessing.normalize(SFE_data[:,:10], axis=0)
scaler = preprocessing.MinMaxScaler()
normalization = scaler.fit_transform(SFE_data[:,:10])
print(normalization[0])


standardization = preprocessing.scale(SFE_data[:,:10], axis=0)
