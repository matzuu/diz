from cmath import log
import string
import sys
from tokenize import String

from torch import Tensor
sys.path.append('../FedAdapt')
import config

from skimage.morphology import disk
from skimage.filters.rank import entropy
    
import pickle
import numpy as np
import matplotlib.pyplot as plt
import skimage
import pandas as pd
import array as arr
import scipy

def unpickleBytes(file: string) -> pd.DataFrame:
    with open(file , 'rb') as f:
        datares1 = pickle.load(f, encoding='bytes')
        f.close()
    return datares1

def unpickle(file: string) -> pd.DataFrame:
    with open(file , 'rb') as f:
        datares1 = pickle.load(f)
        f.close()
    return datares1

def get_avg_entropy_of_img_batches(list_of_tensorIndexes: list) -> list: 
    '''
    img_list: list of tensor of indexes of images
    returns: list/avg entropy of imgs
    '''
    df = pd.DataFrame

    df =  unpickle("./results/entropy_cifar10_df.pkl")
    entropy_slice = df['Entropy']
    entropy_list = []
    for tensorIdxList in list_of_tensorIndexes:
        entropy_list.append(np.mean(list(entropy_slice.loc[tensorIdxList])))
    return entropy_list

    
def get_entropy_of_aggregated_img_batches(list_of_tensorIndexes: list) -> list:
    dict1 = unpickleBytes(config.dataset_path + "cifar-10-batches-py/data_batch_1")    
    images1 = np.reshape(dict1[b'data'],(10000,3,32,32)) #array is 10000,3,32,32, need it as 10000,32,32,3
    images1 = images1.transpose(0,2,3,1)

    entropy_list = []
    for tensorIdxList in list_of_tensorIndexes:
        batch_matrix_imgs = []#np.array([])
        for idx in tensorIdxList:
            batch_matrix_imgs.append(images1[int(idx)]) #append images of the batch into a 4d array (Idx,Heigh,Width,RGB)
        batch_entropy= skimage.measure.shannon_entropy(np.stack(batch_matrix_imgs, axis=0))
        entropy_list.append(batch_entropy)
    return entropy_list

def get_splitLayer_entropy(split_layer_lists: list) -> float:
    '''splits_layer_lists #list of lists [[],[],[]]'''
    pd_series = pd.Series(split_layer_lists)
    counts = pd_series.value_counts()
    entropy = scipy.stats.entropy(counts)
    return entropy

def get_all_splitLayer_surprise(split_layer_lists) -> list:
    '''returns a list of a surprise for each individual element'''
    pd_series = pd.Series(split_layer_lists)
    countsIndex = pd_series.value_counts().index.tolist()
    countsValue = pd_series.value_counts().values.tolist()
    split_layer_surprise_list = []
    total_elems = len(split_layer_lists)
    for splitLayer in split_layer_lists:
        #probability:
        val = countsValue[countsIndex.index(splitLayer)]
        prob = val/total_elems
        split_layer_surprise_list.append(log(1 / prob))
    return split_layer_surprise_list

if __name__ == "__main__":

    list1 = [1,2,3,4]
    list2 = [4,5,6,7]
    list_of_lists = [list1,list2,list2]
    #get_avg_entropy_of_img_batches(list_of_lists)
    #get_entropy_of_aggregated_img_batches(list_of_lists)
    #get_splitLayer_entropy([[1,2,3],[3,4,1],[1,1,1],[1,2,3]])
    #get_all_splitLayer_surprise(list_of_lists)
    print ("DONE")