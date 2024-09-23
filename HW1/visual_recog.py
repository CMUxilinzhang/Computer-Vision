import os, math
from os.path import join
from copy import copy
from tqdm import tqdm
import numpy as np
import util
import visual_words
from sklearn.metrics import confusion_matrix
from PIL import Image
from networkx import intersection
from distutils.command.build import build
from os import supports_dir_fd
import multiprocessing


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    # iterate
    K = opts.K


    # part 1. get the histogram of the wordmap
    hist, bin_edges = np.histogram(wordmap, bins=K, range=(0, K))
    #print(' The boundary of the histogram is ", bin_edges)


    # part 2. L1-normalize the histogram
    hist = hist / np.sum(hist)

    return hist


def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
    # iterate
    K = opts.K
    L = opts.L

    # part 1. prepare a list
    hist_all = []


    # part 2. use for loop to calculate the histogram of every cell in every chop method
    for l in range(L + 1): # chop
        num_chop = 2 ** l
        cell_height = math.floor( wordmap.shape[0] / num_chop )
        cell_width = math.floor( wordmap.shape[1] / num_chop )
        if l == 0 or l == 1 :
            weight = 2 ** (-L)
        else:
            weight = 2 ** (l - L - 1)

        for i in range(num_chop):   # row
            for j in range(num_chop):   # column
                chop_window = wordmap[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

                hist, edge = np.histogram(chop_window, bins=K, range=(0, K))

                hist = hist / np.sum(hist)
                # followings are visualization of testing the chop function but no longer needed
                #print("👉循环过程中第",l+1,"种分割方式","第[",i,",",j,"]个chop窗口的histogram")
                #print(hist)
                #print(sum(hist))
                hist_all.append(hist * weight)   # 这里hist是一个np类型的一维数组，所以他*weight的意思就是所有元素都*weight，注意python的普通列表*数字得到的结果就是复制列表


    # part 3. L1-Normalize
    hist_all = np.concatenate(hist_all)
    hist_all = hist_all / np.sum(hist_all)
    return hist_all


    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
    # load_img is a function wrote by myself which can return a normalized image array in <util.py>
    img = util.load_img(img_path)

    wordmap = visual_words.get_visual_words(opts, img, dictionary)

    feature = get_feature_from_wordmap_SPM(opts, wordmap)  #这个函数的返回值是一个长为K*Σ(4**l - 1)的histo

    return feature


def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    # iterate
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L


    # part 1. read file
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    #print(train_files)
    train_files = [os.path.join(data_dir, img_path) for img_path in train_files]
    #没有上面这一行的话，train_files.txt里面没有aquarium，desert···等文件夹前面的路经了，所以就会默认在code这个文件夹里面找，所以需要在train_files这个列表每个元素前面加上date_dir，或者直接把那些图片文件夹都复制到code文件夹下面

    # part 2. append the trained features
    train_features = []  # prepare a list for all the trained features
    print("Progress of training set feature extraction：")
    for img_path in tqdm(train_files):
        train_feature_single = get_image_feature(opts, img_path, dictionary)
        train_features.append(train_feature_single)

    train_features = np.vstack(train_features)

    # part 3. save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
         features=train_features,
         labels=train_labels,
         dictionary=dictionary,
         SPM_layer_num=SPM_layer_num)
    print("build_recognition_system has been successfully implemented")
    print("<trained_system.npz> has been saved in: ", out_dir)


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''
    overlap = np.minimum(word_hist, histograms).sum(axis=1)   # calculate the overlap , word_hist is a 1D array，histograms is 2D

    distance = 1 - overlap  # already normalized, so overlap can deduct by 1

    return distance


    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    # iterate the path of the <dictionary.npy>
    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    train_features = trained_system['features']
    train_labels = trained_system['labels']

    # part 1. using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']
    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_files = [os.path.join(data_dir, file) for file in test_files]                    ###
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    predict_labels = []

    print('Progress of getting predicted_labels：')
    for img_path in tqdm(test_files):
        test_feature = get_image_feature(opts, img_path, dictionary)
        distances = distance_to_set(test_feature, train_features)
        closest = np.argmin(distances)  # closest is just an index
        #if(img_path == 'aquarium/sun_afwnlpclpshcueip.jpg'):    #the first image to be tested
            #print('The index of aquarium/sun_afwnlpclpshcueip.jpg is ', closest)
        predicted_label = train_labels[closest]
        predict_labels.append(predicted_label)

    # part 2. transform the predicted label to array
    predicted_labels = np.array(predict_labels)
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

    # part 3. output the confusion matrix and accuracy
    print("Confusion Matrix:", '\n', conf_matrix)
    
    print("Accuracy:", accuracy)

    misclassified = np.where(predicted_labels != test_labels)[0]
    print("Misclassified images:", '\n', misclassified)

    return conf_matrix, accuracy
