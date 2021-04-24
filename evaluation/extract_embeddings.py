import os
import sys
import imp
import argparse
import time
import math
import numpy as np

from utils import utils
from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from network import Network
from evaluation.lfw import LFWTest


def main(args):
    #paths = Dataset(args.dataset_path)['abspath']
    
    ijbc_meta = np.load(args.meta_path)
    paths = [os.path.join(args.input_path, img_name.split('/')[-1]) for img_name in ijbc_meta['img_names']]
    print('%d images to load.' % len(paths))
    assert(len(paths)>0)

    # Load model files and config file
    network = Network()
    network.load_model(args.model_path) 
    images = preprocess(paths, network.config, False)
    print('N_images', images.shape)

    # Run forward pass to calculate embeddings
    mu, sigma_sq = network.extract_feature(images, 128, verbose=True)
    if args.nosigma:
        feat_pfe = mu
    else:
        feat_pfe = np.concatenate([mu, sigma_sq], axis=1)
    print('N_features', feat_pfe.shape)
    np.save(args.output_path, feat_pfe)
    #np.save(args.output_path.replace('.npy', '_paths.npy'), paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="The path to the pre-trained model directory",
                        type=str)
    parser.add_argument("--input_path", help="The path to the LFW dataset directory",
                        type=str, default='data/lfw_mtcnncaffe_aligned')
    parser.add_argument("--meta_path", help="The path to the LFW dataset directory",
                        type=str, default='data/lfw_mtcnncaffe_aligned')
    parser.add_argument("--output_path", help="The path to the numpy file to save",
                        type=str, default='embeds.npy')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=128)
    parser.add_argument("--nosigma", action="store_true", help="Don't save uncertainty estimation")
    args = parser.parse_args()
    print(args)
    if '.npy' not in args.output_path:
        print('output_path should contain .npy')
        exit(-1)
    main(args)
