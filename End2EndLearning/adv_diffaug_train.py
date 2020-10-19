import sys
import os
import time
import argparse
import shutil
import numpy as np

import tensorflow as tf

ROOT_DIR = os.path.abspath("../")
print('PLATFORM_ROOT_DIR ', ROOT_DIR)

sys.path.insert(0, './library/')

def parse_args():
    parser = argparse.ArgumentParser(description='Adv Diff Aug')
    parser.add_argument('--data_root', default='/Data/udacityA_nvidiaB/',
                        type=str, help='prefix used to define output path')
    parser.add_argument('--output_name', default='diffaug_results',
                        type=str, help='prefix used to define output path')
    parser.add_argument('--augments', default=None, type=list, 
                    help='augment list')
    parser.add_argument('--stepsize', default=None, type=float, 
                    help='fgsm step size')
    parser.add_argument('--nrepeat', default=None, type=int, 
                    help='# steps')
    parser.add_argument('--eps', default=None, type=float, 
                    help='adversarial step: projection radias')
    parser.add_argument('--gpu', default=0, type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    if (args.gpu != None):
		os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
	print("CUDA_VISIBLE_DEVICES " + os.environ["CUDA_VISIBLE_DEVICES"])

