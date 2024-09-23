from os.path import join
import numpy as np
from PIL import Image
import util
import visual_words
import visual_recog
from opts import get_opts
import matplotlib.pyplot as plt

def main():
    opts = get_opts()
    out_dir = opts.out_dir
    ## Q1.1
    img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses)

    ## Q1.2
    n_cpu = util.get_num_CPU()
    # part 1. generate and save dictionary.npy
    #visual_words.compute_dictionary(opts, n_worker = n_cpu)

    # part 2. load the <.npy> files
    dictionary = np.load(join(out_dir,'dictionary.npy'))
    print("The shape of Dictionary is:", dictionary.shape, '\n')   # ensure the shape of cluster centers array is correct
    #print("Dictionary content:", dictionary)                # print the content of the dictionary. JUST A TEST
    
    ## Q1.3
    img_path = join(opts.data_dir, 'aquarium/sun_asgtepdmsxsrqqvy.jpg')
    img_path2 = join(opts.data_dir, 'kitchen/sun_aaqhazmhbhefhakh.jpg')    # the pathway of the rest two images
    img_path3 = join(opts.data_dir, 'highway/sun_aagkjhignpmigxkv.jpg')
    img_paths = [img_path, img_path2, img_path3]                           # combine them to an array
    # print(img_paths)

    for i in img_paths:                                      # show the wordmap of three images
        img = Image.open(i)
        img = np.array(img).astype(np.float32)/255
        dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
        wordmap = visual_words.get_visual_words(opts, img, dictionary)
        #print("Dictionary shape:", dictionary.shape)
        #print("Dictionary content:", dictionary)
        #print(np.unique(wordmap, return_counts=True))
        util.visualize_wordmap(wordmap)

    '''
    img = Image.open(img_path)     #
    img = np.array(img).astype(np.float32)/255
    dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    util.visualize_wordmap(wordmap)
    '''
    ## Q2.1-2.4
    # Test whether hist is correct and explore the content of it
    #hist = visual_recog.get_feature_from_wordmap(opts, wordmap)
    #print("Histogram shape:", hist.shape)
    #print("Histogram content:", hist)
    #total_sum = sum(hist)                         # use sum() to test if the hist is normalized  # result : 1

    # Test whether hist_all is correct
    #hist_all = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    #print("SPM feature shape:", hist_all.shape)
    #print("SPM feature shape:", hist_all)

    n_cpu = util.get_num_CPU()
    #visual_recog.build_recognition_system(opts, n_worker = n_cpu)                               ### generate and save trained_system.npz

    ## Q2.5
    n_cpu = util.get_num_CPU()
    visual_recog.evaluate_recognition_system(opts, n_worker = n_cpu)


    # Below is the origin code in template, but I have integrated it into the evaluate_recognition_system function
    # print(conf)
    # print(accuracy)
    # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')

if __name__ == '__main__':
    main()
