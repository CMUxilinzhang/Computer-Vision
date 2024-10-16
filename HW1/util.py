import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from PIL import Image
import cv2

def get_num_CPU():
    '''
    Counts the number of CPUs available in the machine.
    '''
    return multiprocessing.cpu_count()

def display_filter_responses(opts, response_maps):
    '''
    Visualizes the filter response maps.

    [input]
    * response_maps: a numpy.ndarray of shape (H,W,3F)
    '''
    
    n_scale = len(opts.filter_scales)
    plt.figure(1)
    
    for i in range(n_scale*4):
        plt.subplot(n_scale, 4, i+1)
        resp = response_maps[:, :, i*3:i*3 + 3]
        resp_min = resp.min(axis=(0,1), keepdims=True)
        resp_max = resp.max(axis=(0,1), keepdims=True)
        resp = (resp - resp_min)/(resp_max - resp_min)
        plt.imshow(resp)
        plt.axis("off")

    plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,wspace=0.05,hspace=0.05)
    plt.show()

def visualize_wordmap(wordmap, out_path=None):
    plt.figure(2)
    plt.axis('equal')
    plt.axis('off')
    plt.imshow(wordmap)
    plt.show()
    if out_path:
        plt.savefig(out_path, pad_inches=0)

def load_img(img_path):

    img = Image.open(img_path)
    img = np.array(img)
    # if image is gray, convert it to RGB
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    # ensure the image is a float type
    img = img.astype(np.float32) / 255.0
    return img


def resize_image_with_aspect_ratio(image, max_size=(256, 256)):
    max_width, max_height = max_size

    # part 1: the origin shape of image
    h, w = image.shape[:2]

    # part 2: calculate the w / h
    aspect_ratio = w / h

    # part 3: check the image and choose the scale method
    if aspect_ratio > 1:  # wider image
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
    else:  # higher image
        new_height = max_height
        new_width = int(new_height * aspect_ratio)

    # part 4: resize the image according to the ratio
    resized_img = cv2.resize(image, (new_width, new_height))

    return resized_img

def subtract_mean_color(image):
    mean_color = np.mean(image, axis=(0, 1))
    img_subtracted = image - mean_color
    return img_subtracted