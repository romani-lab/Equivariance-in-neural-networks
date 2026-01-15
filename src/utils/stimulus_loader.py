from torchvision import transforms
import numpy as np
from PIL import Image
import skimage
import scipy.io as io
from skimage.transform import resize
from skimage.util import img_as_ubyte


def load_img_sorted(stim_root, stim_name):
    """
    Load images from a stimulus file, sort them into training and testing sets, and make them 300x300 pixels.
    Only works for stimulus file stim_05-08-23.

    Parameters
    ----------
    stim_root : str
        path to the stimulus file
    stim_name : str
        name of the stimulus file

    Returns
    -------
    img_all_transfo : np.ndarray
        array of size n_all x 300 x 300
    """    
    if stim_name=='stim_05-08-23':
        dat = io.loadmat(stim_root / (stim_name+'.mat'), squeeze_me=True)
        idx_train, idx_test = find_sort_idx(stim_root, stim_name)   
        img_all = dat['img']
        img_train = np.transpose(img_all[:,:,idx_train],[2,0,1])
        img_test = np.transpose(img_all[:,:,idx_test],[2,0,1])
        img_all_sorted = np.concatenate((img_train,img_test),axis=0)
        #resize images
        img_all_square = np.zeros((img_all_sorted.shape[0],300,300), dtype=np.uint8)
        for i in range(img_all_sorted.shape[0]):
            #images go over 3 screens, each with 4:3 aspect ratio
            #take square centered img on left screen (rest is gray)
            img_all_square[i] = img_as_ubyte(resize(img_all_sorted[i],(300,1200))[:,50:350])
    else:
        raise ValueError('Can only handle stimulus file stim_05-08-23')

    return img_all_square


def standard_img_transform(img_all, size2):
    """
    Standard image transformations for ANNs.

    Parameters
    ----------
    img_all : np.ndarray
        array of size n_all x n_y x n_x where
        n_all is the number of images
        n_y and n_x are the dimensions of the images
    size2 : int
        half width of the image, used for scaling down 
    
    Returns
    -------
    img_all_transfo : np.ndarray
        array of size n_all x 3 x 224 x 224
    """
    #scale down image
    transform2 = transforms.Resize((2*size2,2*size2))
    #standard image transformations for ANNs
    transform = transforms.Compose([           
     transforms.Resize(256),                   
     transforms.CenterCrop(224),               
     transforms.ToTensor(),                    
     transforms.Normalize(                     
     mean=[0.485, 0.456, 0.406],                
     std=[0.229, 0.224, 0.225]                  
     )])

    #apply transformations to each image
    img_all_transfo = np.zeros((img_all.shape[0],3,224,224),dtype=np.float32)
    for i in range(img_all.shape[0]):
        img_i = Image.fromarray( skimage.color.gray2rgb(img_all[i]) )
        #make gray image with scaled down image in the center
        if size2 < 112:
            img_i_scaled = transform2(img_i)
            img_i_inserted = Image.new('L', (256, 256), 150)
            img_i_inserted.paste(img_i_scaled, (128-size2,128-size2))
            img_i_inserted = img_i_inserted.convert('RGB')
        else:
            img_i_inserted = img_i
        img_all_transfo[i] = transform(img_i_inserted)

    return img_all_transfo


def find_n_trials(stim_name):
    """
    Define the number of training and testing trials in the stimulus file.
    Only works for stimulus file stim_05-08-23.

    Parameters
    ----------
    stim_name : str
        name of the stimulus file

    Returns
    -------
    n_trials_train : int
        number of training trials
    n_trials_test : int
        number of testing trials
    """    
    if stim_name=='stim_05-08-23':
        n_trials_train = 2
        n_trials_test = 80
    else:
        raise ValueError('Can only handle stimulus file stim_05-08-23')

    return n_trials_train, n_trials_test  


def find_n_imgs(stim_name):
    """
    Define the number of angles, training images, and testing images in the stimulus file.
    Only works for stimulus file stim_05-08-23.

    Parameters
    ----------
    stim_name : str
        name of the stimulus file

    Returns
    -------
    n_rot : int
        number of angles
    n_train : int
        number of training images
    n_test : int
        number of testing images
    """
    if stim_name=='stim_05-08-23':
        n_rot = 15
        n_train = 2*45
        n_test = 2*1
    else:
        raise ValueError('Can only handle stimulus file stim_05-08-23')

    return n_rot, n_train, n_test    


def find_sort_idx(stim_root, stim_name):
    """
    Find the indices of the training and testing images in the stimulus file.
    Only works for stimulus file stim_05-08-23.

    Parameters
    ----------
    stim_root : str
        path to the stimulus file
    stim_name : str
        name of the stimulus file

    Returns
    -------
    idx_train : np.array
        array of indices for the training images
    idx_test : np.array
        array of indices for the testing images
    """
    #Load stim data
    dat = io.loadmat( stim_root / (stim_name+'.mat'), squeeze_me=True)
    n_rot, n_train, n_test = find_n_imgs(stim_name)
    if stim_name=='stim_05-08-23':
        #file names
        file_names = [s[0] for s in dat['db'][:n_rot*(n_train+n_test)]]
        #stimulus names w/ monitor suffix (first all left then all front)
        stim_names = np.zeros(n_train+n_test,dtype=object)
        #get stimulus names w/o monitor suffix (take all left names)
        ct = 0
        for s in file_names:
            if '_1_left' in s:
                stim_names[ct] = s[:-7]
                ct+=1
        #sort indices for train and test imgs
        idx_train = np.zeros(n_train*n_rot,dtype=int)
        idx_test = np.zeros(n_test*n_rot,dtype=int)
        for j in range(n_rot):
            for i in range(n_train):
                idx_train[n_rot*i+j] = file_names.index(stim_names[n_test+i]+'_'+str(j+1)+'_left')
            for i in range(n_test):
                idx_test[n_rot*i+j] = file_names.index(stim_names[i]+'_'+str(j+1)+'_left')
    else:
        raise ValueError('Can only handle stimulus file stim_05-08-23')
        
    return idx_train, idx_test