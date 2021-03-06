### This script is the main training file.

import sys
import os
from test import test_network

ROOT_DIR = os.path.abspath("../")
print('PLATFORM_ROOT_DIR ', ROOT_DIR)

sys.path.insert(0, './library/')

from learning import train_dnn_multi

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


def train_network(imagePath, labelPath, outputPath, modelPath = "", trainRatio = 1.0, partialPreModel = False, reinitHeader = False, 
	BN_flag=0, imagePath_advp=[], labelPath_advp=[], trainRatio_advp = 1.0, reinitBN = False, classification = False, netType=1,
	augments=None, adv_step=0.2, n_repeats=3, eps=0.5, before_relu=False, resume=0):
	train_network_multi([imagePath], [labelPath], outputPath, modelPath, trainRatio, partialPreModel, reinitHeader, BN_flag, 
		[imagePath_advp], [labelPath_advp], trainRatio_advp, reinitBN, classification, netType, 
		augments=augments, adv_step=adv_step, n_repeats=n_repeats, eps=eps, before_relu=before_relu, resume=resume)

def train_network_multi(imagePath_list, labelPath_list, outputPath, modelPath = "", trainRatio = 1.0, partialPreModel = False, reinitHeader = False, 
	BN_flag=0, imagePath_list_advp=[], labelPath_list_advp=[], trainRatio_advp = 1.0, reinitBN = False, classification = False, netType=1, pack_flag=False,
	augments=None, adv_step=0.2, n_repeats=3, eps=0.5, before_relu=False, resume=0):
	print('Image folder: ' + str(imagePath_list))
	print('Label file: ' + str(labelPath_list))
	print('Output folder: ' + outputPath)

	if not os.path.exists(outputPath):
		os.mkdir(outputPath)

	## flags
	fRandomDistort = False
	fThreeCameras = False  # set to True if using Udacity data set
	fClassifier = classification
	flags = [fRandomDistort, fThreeCameras, fClassifier]
	
	## parameters
	batchSize = 128
	nEpoch = 1000
	nClass = 49        # only used if fClassifier = True
	nFramesSample = 5  # only used for LSTMs
	nRep = 1
	specs = [batchSize, nEpoch, nClass, nFramesSample, nRep]
	
	## train
    ## NOTE: paths must have forward slash (/) character at the end
    
	#netType = netType        # 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o, 5: GAN
	train_dnn_multi(imagePath_list, labelPath_list, outputPath, netType, flags, specs, modelPath, trainRatio, partialPreModel, reinitHeader, 
		BN_flag, imagePath_list_advp, labelPath_list_advp, trainRatio_advp, reinitBN, pack_flag, 
		augments=augments, adv_step=adv_step, n_repeats=n_repeats, eps=eps, before_relu=before_relu, resume=resume)

# def train_network_diffAug(model, nEpoch, augments, trainGenerator, validGenerator):
# 	net(tf.ones((1, 66, 200, 3)))
# 	net.compile(h_optimizer=tf.keras.optimizers.Adam(1e-4), loss_fn=tf.keras.losses.MeanSquaredError(), h_metrics=mean_accuracy_tf)
# 	val_ma_tracker = tf.keras.metrics.Mean(name="val_ma")
# 	for epoch in range(epochs):
#     	print("\n Train Epoch: [{}/{}]".format(epoch,nEpoch))
#     	start_time = time.time()

# 		# iterate over different augmentations
# 		for aug in augments:
#     		# Iterate over the batches of the dataset.
#     		for step, (x_batch_train, y_batch_train) in enumerate(trainGenerator):
#     		    mloss, ma = model.train_step((x_batch_train, y_batch_train), aug)
# 			print("augmentation: {} \t loss_tracker: {%.4f} \t ma_tracker: {%.4f}\n".format(aug, float(mloss), float(ma)))
		
# 		# validation
# 		for x_batch_val, y_batch_val in validGenerator:
# 			val_ma = model.test_step((x_batch_val, y_batch_val))
# 			val_ma_tracker.update_state(val_ma)
# 		print("\n Val Epoch: [{}/{}] \t ma: {%.4f}\n".format(epoch, nEpoch, float(val_ma_tracker.result())))
# 		print("Time taken: {%.2f}".format(time.time() - start_time))


def train_network_multi_factor_search(imagePath, labelPath, outputPath, modelPath = "", trainRatio = 1.0, partialPreModel = False, reinitHeader = False, 
	BN_flag=0, imagePath_list_advp=[], labelPath_list_advp=[], trainRatio_advp = 1.0, reinitBN = False, classification = False, netType=1):
	print('Image folder: ' + str(imagePath))
	print('Label file: ' + str(labelPath))
	print('Output folder: ' + outputPath)

	if not os.path.exists(outputPath):
		os.mkdir(outputPath)

	## flags
	fRandomDistort = False
	fThreeCameras = False  # set to True if using Udacity data set
	fClassifier = classification
	flags = [fRandomDistort, fThreeCameras, fClassifier]
	
	## parameters
	nRound = 20
	nEpoch = 50
	batchSize = 128
	nClass = 49        # only used if fClassifier = True
	nFramesSample = 5  # only used for LSTMs
	nRep = 1
	specs = [batchSize, nEpoch, nClass, nFramesSample, nRep]

	blur_level = 1
	noise_level = 1
	distortion_level = 1

	G_level = 1
	S_level = 1
	Y_level = 1

	imagePath0 = imagePath[0:-1]

	val_ratio = 0.1
	f = open(outputPath+"factor_level_choices.txt",'w')
	for rid in range(nRound):
		blur_imagePath = imagePath0+'_blur_'+str(blur_level)+'/'
		noise_imagePath = imagePath0+'_noise_'+str(noise_level)+'/'
		distortion_imagePath = imagePath0+'_distort_'+str(distortion_level)+'/'
		G_imagePath = imagePath0+'_G_darker/' if G_level == 1 else imagePath0+'_G_lighter/'
		S_imagePath = imagePath0+'_S_darker/' if S_level == 1 else imagePath0+'_S_lighter/'
		Y_imagePath = imagePath0+'_Y_luma_darker/' if Y_level == 1 else imagePath0+'_Y_luma_lighter/'

		imagePath_list = [imagePath, blur_imagePath, noise_imagePath, distortion_imagePath, G_imagePath, S_imagePath, Y_imagePath]
		
		labelPath_list = [labelPath] * len(imagePath_list)

		#Noise only
		#imagePath_list = [imagePath, imagePath0+'_noise_'+str(noise_level)+'/']
		#labelPath_list = [labelPath, labelPath]

		train_dnn_multi(imagePath_list, labelPath_list, outputPath, netType, flags, specs, modelPath, trainRatio, partialPreModel, reinitHeader, 
			BN_flag, imagePath_list_advp, labelPath_list_advp, trainRatio_advp, reinitBN)

		modelPath = outputPath + "model-final.h5"
		valOutputPath = ""

		print('blur MAs:')
		MA_min = 1
		for new_blur_level in range(1,6):
			blurImagePath = imagePath0+'_blur_'+str(new_blur_level)+'/'
			MA = test_network(modelPath, blurImagePath, labelPath, valOutputPath, BN_flag=BN_flag, ratio=val_ratio)
			print(new_blur_level, ': ', MA)
			if MA_min > MA:
				MA_min = MA
				blur_level = new_blur_level

		print('noise MAs:')
		MA_min = 1
		for new_noise_level in range(1,6):
			noiseImagePath = imagePath0+'_noise_'+str(new_noise_level)+'/'
			MA = test_network(modelPath, noiseImagePath, labelPath, valOutputPath, BN_flag=BN_flag, ratio=val_ratio)
			print(new_noise_level, ': ', MA)
			if MA_min > MA:
				MA_min = MA
				noise_level = new_noise_level

		print('distort MAs:')
		MA_min = 1
		for new_distort_level in range(1,6):
			distortImagePath = imagePath0+'_distort_'+str(new_distort_level)+'/'
			MA = test_network(modelPath, distortImagePath, labelPath, valOutputPath, BN_flag=BN_flag, ratio=val_ratio)
			print(new_distort_level, ': ', MA)
			if MA_min > MA:
				MA_min = MA
				distortion_level = new_distort_level

		print('G MAs:')
		MA_min = 1
		for new_G_level in range(1,3):
			G_imagePath = imagePath0+'_G_darker/' if G_level == 1 else imagePath0+'_G_lighter/'
			MA = test_network(modelPath, G_imagePath, labelPath, valOutputPath, BN_flag=BN_flag, ratio=val_ratio)
			print(new_G_level, ': ', MA)
			if MA_min > MA:
				MA_min = MA
				G_level = new_G_level

		print('S MAs:')
		MA_min = 1
		for new_S_level in range(1,3):
			S_imagePath = imagePath0+'_S_darker/' if S_level == 1 else imagePath0+'_S_lighter/'
			MA = test_network(modelPath, S_imagePath, labelPath, valOutputPath, BN_flag=BN_flag, ratio=val_ratio)
			print(new_S_level, ': ', MA)
			if MA_min > MA:
				MA_min = MA
				S_level = new_S_level

		print('Y MAs:')
		MA_min = 1
		for new_Y_level in range(1,3):
			Y_imagePath = imagePath0+'_Y_luma_darker/' if Y_level == 1 else imagePath0+'_Y_luma_lighter/'
			MA = test_network(modelPath, Y_imagePath, labelPath, valOutputPath, BN_flag=BN_flag, ratio=val_ratio)
			print(new_Y_level, ': ', MA)
			if MA_min > MA:
				MA_min = MA
				Y_level = new_Y_level

		print('new blur level: ', blur_level)
		print('new noise level: ', noise_level)
		print('new distort level: ', distortion_level)
		print('new G channel level: ', G_level)
		print('new S channel level: ', S_level)
		print('new Y channel level: ', Y_level)
		f.write("round no: "+str(rid)+"\n")
		f.write("new blur level: "+str(blur_level)+"\n")
		f.write("new noise level: "+str(noise_level)+"\n")
		f.write("new distort level: "+str(distortion_level)+"\n\n")
		f.write("new G channel level: "+str(G_level)+"\n")
		f.write("new S channel level: "+str(S_level)+"\n")
		f.write("new Y channel level: "+str(Y_level)+"\n\n")
		f.flush()
	f.close()


if __name__ == "__main__":

	import argparse

    # Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Train CNN to predict steering angle.')
	parser.add_argument('--image_folder_path', required=False,
						metavar="/path/to/image/folder/",
						help='/path/to/image/folder/')
	parser.add_argument('--label_file_path', required=False,
						metavar="/path/to/label/file",
						help="/path/to/label/file")
	parser.add_argument('--output_path', required=False,
						metavar="/path/to/output/folder/",
						help="/path/to/output/folder/")
	args = parser.parse_args()


	data_root = ROOT_DIR + '/Data/'

    # NVIDIA dataset 
	trainPath = data_root + 'udacityA_nvidiaB/'
    
    #image folder path
	imagePath = trainPath + 'trainB/'
	if args.image_folder_path != None:
		imagePath = args.image_folder_path

	#label file path
	labelPath = trainPath + 'labelsB_train.csv'
	if args.label_file_path != None:
		labelPath = args.label_file_path

	outputPath = data_root + 'udacityA_nvidiaB_results/training_models/'
	if args.output_path != None:
		outputPath = args.output_path

	train_network(imagePath, labelPath, outputPath)
	