import os
import multiprocessing
from os.path import join, isfile
from tqdm import tqdm
import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
from sklearn.cluster import KMeans
import util
from opts import get_opts


def extract_filter_responses(opts, img):

    '''
    Extracts the filter responses for the given image.

    [input]
    * opts   : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    # iterate
    filter_scales = opts.filter_scales


    # part 1: Convert the image to Lab color space
    if img.ndim == 2:
       img = np.repeat(img[:, :, np.newaxis], 3, 2)

    img = img.astype(np.float32)
    img_lab = skimage.color.rgb2lab(img)
    #print(img_lab)

    # part 2: Define the filter bank
    filter_scales = opts.filter_scales
    num_scales = len(filter_scales)
    num_filters = 4    # 4 types of filters (Gaussian, LoG, DoG_x, DoG_y)
    H, W, C = img.shape


    # part 3: Prepare the output array for filter responses
    filter_responses = np.zeros((H, W, C * num_filters * num_scales))


    # part 4: Loop over the scales and filters
    layer = 0   # iterate the parameter layer outside the loop!!!
    for scale in filter_scales:
        for i in range(C):  # Apply to each channel (number of Lab image channel is 3)
            img_channel = img_lab[:, :, i]
            #img_channel = img[:,:,i]

            # Gaussian
            filter_responses[:, :, layer] = scipy.ndimage.gaussian_filter(img_channel, scale, [0, 0])  #这里可以不写order，default is [0,0]
            layer += 1

            # Laplacian of Gaussian
            filter_responses[:, :, layer] = scipy.ndimage.gaussian_laplace(img_channel, scale)
            layer += 1

            # Derivative of Gaussian(x and y)
            filter_responses[:, :, layer] = scipy.ndimage.gaussian_filter(img_channel, scale, [1, 0]) # derivative in x direction
            layer += 1
            filter_responses[:, :, layer] = scipy.ndimage.gaussian_filter(img_channel, scale, [0, 1]) # derivative in y direction
            layer += 1

    return filter_responses


def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    opts = get_opts()
    data_dir = opts.data_dir
    opts, img_path, temp_file_path = args

    # part 1. Load
    img = Image.open(join(data_dir, img_path))
    img = img.convert('RGB')
    img = np.array(img)

    # part 2. filter responses
    filter_responses = extract_filter_responses(opts, img)

    # part 3. sample
    H, W, C = filter_responses.shape
    alpha = opts.alpha
    sample_indices = np.random.choice(H * W, alpha, replace=False)

    # part 4. Reshape
    sampled_responses = filter_responses.reshape(-1, C)[sample_indices]

    # part 5. Save the sampled responses
    np.save(temp_file_path, sampled_responses)


def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''
    # iterate
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    n_cpu = util.get_num_CPU()  # 这个本来是为了传参给kmeans函数，但是现在版本的scikit-learn库里面的kmeans已经可以自动检测有几个cpu了，所以不用传参了

    # part 1.Load the training images
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # train_files = open("data/train_files.txt").read().splitlines()
    '''
    ##this is the for loop when I get stuck in file reading error previously but no longer needed
    for file in train_files:
        file_path = join(data_dir, file) 
        if isfile(file_path):             #  ATTENTION: isfile()can only accept the string instead of a list
            print(f"{file} is a valid file.")
        else:
            print(f"{file} is not a valid file or does not exist.")
    '''
    # part 2. For each image, extract filter responses and sample responses (α*T).
    all_responses = []
    alpha = opts.alpha  # Number of sampled pixels per image

    print("Progress of computing dictionary")
    for img_path in tqdm(train_files):
        img = Image.open(join(data_dir,img_path))   # Open the image file
        img = img.convert('RGB')                    # Ensure image is in RGB format
        img = np.array(img).astype(np.float32)/255  # Convert image to numpy array
        filter_responses = extract_filter_responses(opts, img)  # Extract responses  get the responses with the shape of（H*W, 3F）

        # Randomly sample alpha pixels
        H, W, C = filter_responses.shape
        sample_indices = np.random.choice(H * W, alpha, replace=False)
        sampled_responses = filter_responses.reshape(-1, C)[sample_indices]
        all_responses.append(sampled_responses)

    # part 3. Stack all the sampled responses into a single matrix (αT × 3F)
    all_responses = np.vstack(all_responses)  # this function can make all_responses to be one 2D array
    #print("The shape of all_responses[] is ", all_responses.shape)


    # part 4. Run K-means clustering on the sampled filter responses
    K = opts.K  # Number of clusters
    kmeans = KMeans(n_clusters=K).fit(all_responses)


    # part 5. Save the cluster centers as the visual word dictionary
    dictionary = kmeans.cluster_centers_


    # part 6. Save the dictionary to a file for future use
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    # np.savetxt('dictionary.txt', dictionary)      # visualize the dictionary centers but no longer needed
    if(isfile(os.path.join(out_dir, 'dictionary.npy'))):
        print("Dictionary has been saved successfully")
        print("<dictionary.npy> has been saved in ", out_dir)

    return dictionary


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    # part 1. get the filter_responses
    filter_responses = extract_filter_responses(opts, img)
    #print("滤波响应矩阵", filter_responses)


    # part 2. prepare the 2D responses for cdist
    H, W, C = filter_responses.shape
    filter_responses = filter_responses.reshape(-1, C)   # reshape filter_response to (H*W,3F),  the shape of dictionary is(k, 3F)


    # part 3. calculate the Euclidean distance and using index to represent the minimum distance
    dis = scipy.spatial.distance.cdist(filter_responses, dictionary, metric='euclidean')   # the shape of "dis" is(H*W, k) 这个cdist函数只能接收二维数组
    #print("the shape of the Euclidean matrix: ", dis.shape)
    #print("partial content of the 'dis' matrix: ", dis[:8, :10])
    closest_dis = np.argmin(dis, axis=1)  # Shape: (H*W,1) 返回的是最小值的索引
    wordmap = closest_dis.reshape(H, W)

    return wordmap
