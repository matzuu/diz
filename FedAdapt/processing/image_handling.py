

import sys
sys.path.append('../FedAdapt')
import config

from skimage.morphology import disk
from skimage.filters.rank import entropy
    
import pickle
import numpy as np
import matplotlib.pyplot as plt
import skimage
import pandas as pd


def unpickle(file):
    with open(file , 'rb') as f:
        datares1 = pickle.load(f, encoding='bytes')
        f.close()
    return datares1

def first_get_entropy_attempt():
    dict1 = unpickle(config.dataset_path + "cifar-10-batches-py/data_batch_1")
    
    images1 = np.reshape(dict1[b'data'],(10000,3,32,32)) #array is 10000,3,32,32, need it as 10000,32,32,3
    images1 = images1.transpose(0,2,3,1) # change to 10000,32,32,3   i.e move index 1(3) to 3rd intex, and move the 2nd & 3rd into 1&2; index 0 (10000) should stay the same

    labels1 = dict1[b'labels']
    #####################
    img = images1[457]
    grayscale_img = skimage.color.rgb2gray(img)

    entropy_of_img = skimage.measure.shannon_entropy(img)
    entropy_of_grayimg = skimage.measure.shannon_entropy(grayscale_img)

    entropy_img = entropy(grayscale_img, disk(3))
    #plt.imshow(entropy_img)
    #plt.show()

    
    fig, ((ax0, ax1,ax2)) = plt.subplots(nrows=1, ncols=3) # , figsize=(10, 4)
    #(ax00, ax01,ax02),(ax10, ax11,ax12),(ax20, ax21,ax22) TODO wanted to change3x3

    ax0.imshow(img)
    ax0.set_title("Original")
    ax1.imshow(grayscale_img, cmap='gray')
    ax1.set_title("Grayscale Orig")
    ax2.imshow(entropy_img, cmap='viridis')
    ax2.set_title("Local entropy")
    ax0.set_xlabel("entropy:"+str(entropy_of_img))
    ax1.set_xlabel("entropy:"+str(entropy_of_grayimg))
    

    fig.tight_layout()
    fig.show()




if __name__ == "__main__":
    dict1 = unpickle(config.dataset_path + "cifar-10-batches-py/data_batch_1")
    
    images1 = np.reshape(dict1[b'data'],(10000,3,32,32)) #array is 10000,3,32,32, need it as 10000,32,32,3
    images1 = images1.transpose(0,2,3,1) # change to 10000,32,32,3   i.e move index 1(3) to 3rd intex, and move the 2nd & 3rd into 1&2; index 0 (10000) should stay the same
    indexes = range(10000)
    labels1 = dict1[b'labels']
    #####################
    ### Calc Entropy

    entropy_of_img_list = []

    for img in images1:
        entropy_of_img = skimage.measure.shannon_entropy(img)
        entropy_of_img_list.append(entropy_of_img)


    #####################
    ### PANDAS DF

    data = {
        "Index": indexes,
        "Labels": labels1,
        "Entropy": entropy_of_img_list
    }

    df = pd.DataFrame(data)
    aux = df.loc[0]

    avg_entropy_per_label = []

    aux = df.groupby('Labels')['Entropy'].mean()
    print(aux)

    df.to_pickle('./results/entropy_cifar10_df.pkl')
    ###
    
    # img = images1[457]
    # grayscale_img = skimage.color.rgb2gray(img)

    # entropy_of_img = skimage.measure.shannon_entropy(img)
    # entropy_of_grayimg = skimage.measure.shannon_entropy(grayscale_img)

    # entropy_img = entropy(grayscale_img, disk(3))
    # #plt.imshow(entropy_img)
    # #plt.show()

    
    # fig, ((ax0, ax1,ax2)) = plt.subplots(nrows=1, ncols=3) # , figsize=(10, 4)
    # #(ax00, ax01,ax02),(ax10, ax11,ax12),(ax20, ax21,ax22) TODO wanted to change3x3

    # ax0.imshow(img)
    # ax0.set_title("Original")
    # ax1.imshow(grayscale_img, cmap='gray')
    # ax1.set_title("Grayscale Orig")
    # ax2.imshow(entropy_img, cmap='viridis')
    # ax2.set_title("Local entropy")
    # ax0.set_xlabel("entropy:"+str(entropy_of_img))
    # ax1.set_xlabel("entropy:"+str(entropy_of_grayimg))
    

    # fig.tight_layout()
    # fig.show()


    print("TEST")