### This script is the main training file.

import sys
import os

ROOT_DIR = os.path.abspath("../")
print('PLATFORM_ROOT_DIR ', ROOT_DIR)

sys.path.insert(0, './library/')

from learning import train_dnn

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from train import train_network
from test import test_network

DATASET_ROOT = ROOT_DIR + "/Data/udacityA_nvidiaB/"
TRAIN_OUTPUT_ROOT = ROOT_DIR + "/Data/udacityA_nvidiaB_results/train_results/"
TEST_OUTPUT_ROOT = ROOT_DIR + "/Data/udacityA_nvidiaB_results/test_results/"

if not os.path.exists(TRAIN_OUTPUT_ROOT):
	os.mkdir(TRAIN_OUTPUT_ROOT)

if not os.path.exists(TEST_OUTPUT_ROOT):
	os.mkdir(TEST_OUTPUT_ROOT)

def unit_test_for_style():
	TRAIN_LIST = ["trainA", "trainA_fake_GAN", "trainA_fake_color", "trainB", "trainB_fake_GAN", "trainB_fake_color"]
	'''
	for train_folder in TRAIN_LIST:
		imagePath = DATASET_ROOT + train_folder + "/"
		if "trainA" in train_folder:
			labelPath = DATASET_ROOT + "labelsA_train.csv"
		else:
			labelPath = DATASET_ROOT + "labelsB_train.csv"
		outputPath = TRAIN_OUTPUT_ROOT + train_folder + "/"
		train_network(imagePath, labelPath, outputPath)
	'''

	for train_folder in TRAIN_LIST:
		for val_folder in TRAIN_LIST:
			modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/model-final.h5"
			val_folder = val_folder.replace("train", "val")

			imagePath = DATASET_ROOT + val_folder + "/"
			if "valA" in val_folder:
				labelPath = DATASET_ROOT + "labelsA_val.csv"
			else:
				labelPath = DATASET_ROOT + "labelsB_val.csv"
			outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + val_folder + ")/test_result.txt"
			test_network(modelPath, imagePath, labelPath, outputPath)

def unit_test_for_quality():
	TRAIN_LIST_LIST = [["trainB", "trainB_blur_1", "trainB_blur_2", "trainB_blur_3"],
					["trainB", "trainB_noise_1", "trainB_noise_2", "trainB_noise_3"],
					["trainB", "trainB_distort_1", "trainB_distort_2", "trainB_distort_3"]]

	for TRAIN_LIST in TRAIN_LIST_LIST:
		for train_folder in TRAIN_LIST:
			if train_folder == "trainB":
				continue
			imagePath = DATASET_ROOT + train_folder + "/"
			if "trainA" in train_folder:
				labelPath = DATASET_ROOT + "labelsA_train.csv"
			else:
				labelPath = DATASET_ROOT + "labelsB_train.csv"
			outputPath = TRAIN_OUTPUT_ROOT + train_folder + "/"
			train_network(imagePath, labelPath, outputPath)

	for TRAIN_LIST in TRAIN_LIST_LIST:
		for train_folder in TRAIN_LIST:
			for val_folder in TRAIN_LIST:
				modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/model-final.h5"
				val_folder = val_folder.replace("train", "val")

				imagePath = DATASET_ROOT + val_folder + "/"
				if "valA" in val_folder:
					labelPath = DATASET_ROOT + "labelsA_val.csv"
				else:
					labelPath = DATASET_ROOT + "labelsB_val.csv"
				outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + val_folder + ")/test_result.txt"
				test_network(modelPath, imagePath, labelPath, outputPath)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='batch train test')
	parser.add_argument('--gpu_id', required=False, metavar="gpu_id", help='gpu id (0/1)')
	args = parser.parse_args()

	if (args.gpu_id != None):
		os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)

	unit_test_for_style()
	#unit_test_for_quality()