# This script is for learning.

import cv2  
import os
import shutil
import functools
import numpy as np

import tensorflow as tf
import keras

from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from utilities import resize_image, random_distort, load_train_data, load_train_data_multi, load_train_data_multi_pack
from networks import net_lstm, create_nvidia_network, GAN_Nvidia, mean_accuracy, mean_accuracy_tf
import time
import ntpath



def gen_train_data_lstm_m2m(xList, yList, batchSize, nFramesSample):
	## get feature dimensions
	featureSample = resize_image(cv2.imread(xList[0]))
	
	## specify X and y shapes
	X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
	y = np.empty((batchSize, nFramesSample, 1))
	
	## generate training data
	sampleCount = 0
	while True:  
		for i in range(0, len(yList) - nFramesSample):
			## create a sample that has multiple frames
			for k in range(nFramesSample):
				X[sampleCount,k] = resize_image(cv2.imread(xList[i + k]))
				y[sampleCount,k] = yList[i + k]
				
			sampleCount += 1
				
			## yield a batch when #samples = batchSize
			if(sampleCount == batchSize):
				yield (X, y)
				X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
				y = np.empty((batchSize, nFramesSample, 1))
				sampleCount = 0
				
				
				
def gen_train_data_lstm_m2o(xList, yList, batchSize, nFramesSample):
	## get feature dimensions
	featureSample = resize_image(cv2.imread(xList[0]))
	
	## specify X and y shapes
	X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
	y = np.empty((batchSize, 1))
	
	## generate training data
	sampleCount = 0
	while True:  
		for i in range(0, len(yList) - nFramesSample):
			## create a sample that has multiple frames
			for k in range(nFramesSample):
				X[sampleCount,k] = resize_image(cv2.imread(xList[i + k]))
				
			y[sampleCount] = yList[i + nFramesSample]
			sampleCount += 1
				
			## yield a batch when #samples = batchSize
			if(sampleCount == batchSize):
				yield (X, y)
				X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
				y = np.empty((batchSize, 1))
				sampleCount = 0

# class ImageDataset(tf.data.Dataset):
	# def _generator(xList, yList, batchSize):
		# xList, yList = shuffle(xList, yList)
		# X,y = ([],[])
		# while True:       
			# for i in range(len(yList)):
				# image_path = xList[i]
				# if not os.path.isfile(image_path):
					# image_path = image_path.replace(".jpg", "_fake.png")
				# img = resize_image(cv2.imread(image_path))
				# angle = yList[i]
				# X.append(img)
				# y.append(angle)
# 
				# when a batch is ready, yield, and prepare for the next batch
				# if len(X) == batchSize:
					# yield (np.array(X), np.array(y))
					# X, y = ([],[])
					# xList, yList = shuffle(xList, yList)
# 
	# def __new__(cls, xList, yList, batchSize):
		# return tf.data.Dataset.from_generator(
			# cls._generator,
			# output_types = (tf.float32, tf.float32),
			# args = (xList, yList, batchSize)
		# )

def load_images_from_path(path):
	image = tf.io.read_file(path)
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.resize(image, [66, 200], method='area')
	img = image[..., ::-1]
	return img
						
def gen_train_data_random(xList, yList, batchSize, fRandomDistort = False, fFlip = False):
	xList, yList = shuffle(xList, yList)
	X,y = ([],[])
	while True:       
		for i in range(len(yList)):
			image_path = xList[i]
			if not os.path.isfile(image_path):
				image_path = image_path.replace(".jpg", "_fake.png")
			img = resize_image(cv2.imread(image_path))
			angle = yList[i]
			if fRandomDistort:
				print('######### Applying random distortion #########')
				img, angle = random_distort(img, angle)
			X.append(img)
			y.append(angle)
			
			## when a batch is ready, yield, and prepare for the next batch
			if len(X) == batchSize:
				yield (np.array(X), np.array(y))
				X, y = ([],[])
				xList, yList = shuffle(xList, yList)
				
			## flip an image horizontally along its corresponding steering angle
			if fFlip:
				angleThreshold = 0.33
				if abs(angle) > angleThreshold:
					X.append(cv2.flip(img, 1))
					y.append(angle * -1)
					if len(X) == batchSize:
						yield (np.array(X), np.array(y))
						X, y = ([],[])
						xList, yList = shuffle(xList, yList)

def gen_train_data_random_pack_channel(xList, yList, batchSize, fRandomDistort = False, fFlip = False):
	xList, yList = shuffle(xList, yList)
	X,y = ([],[])
	while True:       
		for i in range(len(yList)):
			for j in range(len(xList[i])):
				image_path = xList[i][j]
				if not os.path.isfile(image_path):
					image_path = image_path.replace(".jpg", "_fake.png")
				img_1 = resize_image(cv2.imread(image_path))
				if j == 0:
					img = img_1
				else:
					img = np.concatenate((img, img_1), axis=2)

			#noise = np.random.uniform(low=0, high=255, size=(img.shape[0], img.shape[1], 1))
			#img = np.concatenate((img, noise), axis=2)

			angle = yList[i][0]
			if fRandomDistort:
				print('######### Applying random distortion #########')
				img, angle = random_distort(img, angle)


			X.append(img)
			y.append(angle)
			
			## when a batch is ready, yield, and prepare for the next batch
			if len(X) == batchSize:
				yield (np.array(X), np.array(y))
				X, y = ([],[])
				xList, yList = shuffle(xList, yList)
				
			## flip an image horizontally along its corresponding steering angle
			if fFlip:
				angleThreshold = 0.33
				if abs(angle) > angleThreshold:
					X.append(cv2.flip(img, 1))
					y.append(angle * -1)
					if len(X) == batchSize:
						yield (np.array(X), np.array(y))
						X, y = ([],[])
						xList, yList = shuffle(xList, yList)

def gen_train_data_random_AdvProp(xList, yList, xList_advp, yList_advp, batchSize, fRandomDistort = False, fFlip = False):
	xList, yList = shuffle(xList, yList)
	X,y = ([],[])
	X_advp, y_advp = ([],[])
	while True:       
		for i in range(max(len(yList), len(yList_advp))):
			image_path = xList[i%len(xList)]
			if not os.path.isfile(image_path):
				image_path = image_path.replace(".jpg", "_fake.png")
			img = resize_image(cv2.imread(image_path))
			angle = yList[i%len(yList)]

			image_path_advp = xList_advp[i%len(xList_advp)]
			if not os.path.isfile(image_path_advp):
				image_path_advp = image_path_advp.replace(".jpg", "_fake.png")
			img_advp = resize_image(cv2.imread(image_path_advp))
			angle_advp = yList_advp[i%len(yList_advp)]

			if fRandomDistort:
				print('######### Applying random distortion #########')
				img, angle = random_distort(img, angle)
				img_advp, angle_advp = random_distort(img_advp, angle_advp)


			#X.append([img, img_advp])
			#y.append([angle, angle_advp])
			X.append(img)
			y.append(angle)
			X_advp.append(img_advp)
			y_advp.append(angle_advp)

			
			## when a batch is ready, yield, and prepare for the next batch
			if len(X) == batchSize:
				#yield (np.array(X), np.array(y))
				#yield (np.array([X, X_advp]), np.array([y, y_advp]))
				yield [np.array(X), np.array(X_advp)], [np.array(y), np.array(y_advp)]
				X, y = ([],[])
				X_advp, y_advp = ([],[])
				xList, yList = shuffle(xList, yList)
				xList_advp, yList_advp = shuffle(xList_advp, yList_advp)
				
			## flip an image horizontally along its corresponding steering angle
			if fFlip:
				angleThreshold = 0.33
				if abs(angle) > angleThreshold:
					X.append(cv2.flip(img, 1))
					y.append(angle * -1)
					if len(X) == batchSize:
						yield (np.array(X), np.array(y))
						X, y = ([],[])
						xList, yList = shuffle(xList, yList)


def normalize_value(value_list):
	THRESH_HOLDS = 24
	value_list = np.clip(value_list, -THRESH_HOLDS, THRESH_HOLDS)
	value_list = value_list + THRESH_HOLDS + 0.5
	return value_list

def train_dnn_multi(imageDir_list, labelPath_list, outputPath, netType, flags, specs, modelPath = "", 
	trainRatio = 1.0, partialPreModel = False, reinitHeader = False, 
	BN_flag=0, imageDir_list_advp=[], labelPath_list_advp=[], trainRatio_advp = 1.0, reinitBN = False, pack_flag=False,
	augments=None, adv_step=0.2, n_repeats=3, eps=0.5, before_relu=False, resume=0):
	
	## assigning variables
	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier    = flags[2]
	batchSize 	   = specs[0]
	nEpoch 		   = specs[1]
	nClass         = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	## prepare the data
	#xList, yList = load_train_data_multi(imageDir_list, labelPath_list, nRep, fThreeCameras, trainRatio, specialFilter=True)
	if not pack_flag:
		xList, yList = load_train_data_multi(imageDir_list, labelPath_list, nRep, fThreeCameras, trainRatio)
	else:
		xList, yList = load_train_data_multi_pack(imageDir_list, labelPath_list, nRep, fThreeCameras, trainRatio)

	xTrainList, xValidList = train_test_split(np.array(xList), test_size=0.1, random_state=42)
	yTrainList, yValidList = train_test_split(np.array(yList), test_size=0.1, random_state=42)

	if BN_flag == 2:
		xList_advp, yList_advp = load_train_data_multi(imageDir_list_advp, labelPath_list_advp, nRep, fThreeCameras, trainRatio_advp)

		xTrainList_advp, xValidList_advp = train_test_split(np.array(xList_advp), test_size=0.1, random_state=42)
		yTrainList_advp, yValidList_advp = train_test_split(np.array(yList_advp), test_size=0.1, random_state=42)
	
	## change the data format if necessary
	if fClassifier:
		print('\n######### Classification #########')
		yTrainList = normalize_value(yTrainList)
		yTrainList = to_categorical(yTrainList, num_classes = nClass)
		yValidList = normalize_value(yValidList)
		yValidList = to_categorical(yValidList, num_classes = nClass)
	else:
		print('\n######### Regression #########')
		
	print('Train data:', xTrainList.shape, yTrainList.shape)
	print('Valid data:', xValidList.shape, yValidList.shape)
	print('##############################\n')
	
	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o, 5: GAN
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		nChannel = 3
		if pack_flag:
			nChannel = 3*len(imageDir_list)
		if augments is not None:
			net = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel, 
							augments=augments, adv_step=adv_step, n_repeats=n_repeats, eps=eps)
			net(tf.ones((1, 66, 200, 3)))
			print("DiffAug model created")
		else:
			if BN_flag == 3:
				net = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel, 
							adv_step=adv_step, n_repeats=n_repeats, eps=eps, before_relu=before_relu)
				print("AdvBN model created")
			else:
				net = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel)
		if BN_flag <= 1 or BN_flag == 3:
			if not pack_flag:
				trainGenerator = gen_train_data_random(xTrainList, yTrainList, batchSize)
				validGenerator = gen_train_data_random(xValidList, yValidList, batchSize)
			else:
				trainGenerator = gen_train_data_random_pack_channel(xTrainList, yTrainList, batchSize)
				validGenerator = gen_train_data_random_pack_channel(xValidList, yValidList, batchSize)
		elif BN_flag == 2:
			trainGenerator = gen_train_data_random_AdvProp(xTrainList, yTrainList, xTrainList_advp, yTrainList_advp, batchSize)
			validGenerator = gen_train_data_random_AdvProp(xValidList, yValidList, xValidList_advp, yValidList_advp, batchSize)

	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
		trainGenerator = gen_train_data_lstm_m2o(xTrainList, yTrainList, batchSize, nFramesSample)
		validGenerator = gen_train_data_lstm_m2o(xValidList, yValidList, batchSize, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)
		trainGenerator = gen_train_data_lstm_m2m(xTrainList, yTrainList, batchSize, nFramesSample)
		validGenerator = gen_train_data_lstm_m2m(xValidList, yValidList, batchSize, nFramesSample)
	elif netType == 5:
		net = GAN_Nvidia()
		trainGenerator = gen_train_data_random(xTrainList, yTrainList, batchSize)
		validGenerator = gen_train_data_random(xValidList, yValidList, batchSize)

	if modelPath != "":
		print("loading pretrain modelPath: ", modelPath)
		if BN_flag == 3:
			net(tf.ones((1, 66, 200, 3)))
		net.load_weights(modelPath)
		# print(partialPreModel)
		if partialPreModel:
			print("partial PreModel activate")
			#net_untrain = net_nvidia(fClassifier, nClass)
			start_layer_id=8
			for i in range(start_layer_id):
				net.layers[i].trainable = False
			#for i in range(start_layer_id, len(net.layers)):
				#net.layers[i].set_weights(net_untrain.layers[i].get_weights())
			#	net.layers[i].trainable = False
			net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=[mean_accuracy])
		if reinitHeader:
			print("reinit header activate")
			net_untrain = create_nvidia_network(BN_flag, fClassifier, nClass)
			net.layers[-1].set_weights(net_untrain.layers[-1].get_weights())
			net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=[mean_accuracy])
		if reinitBN:
			net_untrain = create_nvidia_network(BN_flag, fClassifier, nClass)
			BN_layer_ids = [3, 6, 9, 12, 15, 19, 22, 25]
			for id in BN_layer_ids:
				net.layers[id].set_weights(net_untrain.layers[id].get_weights())
			net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=[mean_accuracy])
			#net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
		if BN_flag == 3:
			print('load weight for AdvBN fine-tuning')
			# step = tf.Variable(0, trainable=False)
			# boundaries = [200, 400]
			# values = [1e-5, 5e-6, 1e-6]
			# learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
				# boundaries, values)
			# Later, whenever we perform an optimization step, we pass in the step.
			# learning_rate = learning_rate_fn(step)
			net.compile(h_optimizer=tf.keras.optimizers.Adam(1e-5), loss_fn=tf.keras.losses.MeanSquaredError(), h_metrics=mean_accuracy_tf)
			print(net.summary())
			print('trainable weight:')
			for var in net.trainable_weights:
				print(var.name)


	## setup outputs
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	#else:
	#	shutil.rmtree(outputPath)
	#	os.makedirs(outputPath)

	modelLog = ModelCheckpoint(outputPath + 'model{epoch:02d}.h5', monitor='val_loss', save_best_only=True)
	lossLog  = CSVLogger(outputPath + 'loss-log', append=True, separator=',')
	#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=outputPath+"logs/")

	## train
	if netType != 5:
		nTrainStep = int(len(yTrainList)/batchSize)
		nValidStep = int(len(yValidList)/batchSize)
		if augments is not None:
			x_path = tf.data.Dataset.from_tensor_slices(xTrainList)
			xDataset = x_path.map(load_images_from_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			yDataset = tf.data.Dataset.from_tensor_slices(yTrainList)
			trainDataset = tf.data.Dataset.zip((xDataset, yDataset))
			batchTrainDataset = trainDataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True).batch(batchSize, drop_remainder=True)

			val_x_path = tf.data.Dataset.from_tensor_slices(xValidList)
			val_xDataset = val_x_path.map(load_images_from_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
			val_yDataset = tf.data.Dataset.from_tensor_slices(yValidList)
			valDataset = tf.data.Dataset.zip((val_xDataset, val_yDataset))
			batchValDataset = valDataset.batch(batchSize, drop_remainder=False)
			
			modelLog_tf = tf.keras.callbacks.ModelCheckpoint(
					outputPath + 'checkpoint/cp-weights-{epoch:02d}.ckpt', save_best_only=False,
					save_weights_only=True, save_freq=100)
			lossLog_tf = tf.keras.callbacks.CSVLogger(outputPath + 'train-log', append=True, separator='\t')
			loss_log = open(outputPath + 'loss-log', "w")

			# net.compile(h_optimizer=tf.keras.optimizers.Adam(1e-4), loss_fn=tf.keras.losses.MeanSquaredError(), h_metrics=mean_accuracy_tf)
			optimizer = tf.keras.optimizers.Adam(1e-3)
			loss_fn = tf.keras.losses.MeanSquaredError()

			val_ma_tracker = tf.keras.metrics.Mean(name="val_ma")
			val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
			for epoch in range(resume, nEpoch):
			# for epoch in range(1):
				print("\n Train Epoch: [{}/{}]".format(epoch, nEpoch))
				loss_log.write("\n Train Epoch: [{}/{}]".format(epoch,nEpoch))
				start_time = time.time()

				# iterate over different augmentations
				for aug in augments:
					# Iterate over the batches of the dataset.
					for step, (x_batch_train, y_batch_train) in enumerate(batchTrainDataset):
						# mloss = net.my_train_step((x_batch_train, y_batch_train), aug)
						# print(x_batch_train[:10])
						# print(y_batch_train[:10])
						input = tf.cast(x_batch_train, dtype=tf.float32)
						target = tf.cast(y_batch_train, dtype=tf.float32)
						
						if net.delta is None:
							net.delta = tf.Variable(tf.zeros([1]))
						else:
							net.delta.assign(tf.zeros([1]))

						for _iter in range(net.n_repeats):
							with tf.GradientTape() as tape:
								if aug == '1': # gaussian blur
									aug_op = getattr(net, "blur")
									param = net.delta * 100
									param_min = 0.0
								elif aug == '2': # gaussian noise
									aug_op = getattr(net, "gaussian")
									dist = tfp.distributions.Normal(0, net.delta)
									param = dist.sample([66, 200, 3])
									param_min = 0.0
								# elif aug == '3': #distortion
								elif aug in ['R', 'G', 'B', 'H', 'S', 'V']: 
									aug_op = getattr(net, "color_" + aug)
									param = net.delta
									param_min = -net.eps
								elif aug == '7':
									aug_op = getattr(net, "saturation")
									param = net.delta + 1
									param_min = -net.eps
								elif aug == '8':
									aug_op = getattr(net, "contrast")
									param = net.delta + 1
									param_min = -net.eps
								elif aug == '9':
									aug_op = getattr(net, "brightness")
									param = net.delta
									param_min = -net.eps
								elif aug == 'N':
									aug_op = None
									break
								else:
									print("augmentation not defined")

								x = aug_op(input, param)
								output = net(x)
								# loss = loss_fn(output, target)
								loss = net.mse(output, target)

							grad = tape.gradient(loss, [net.delta])[0]
							net.delta.assign_add(net.adv_step * tf.keras.backend.sign(grad))
							net.delta.assign(tf.keras.backend.clip(net.delta, min_value=param_min, max_value=net.eps))
			
						if aug_op is not None:
							x = aug_op(input, param)
						else:
							x = input
						with tf.GradientTape() as tape:
							output = net(x)
							loss = loss_fn(output, target)
							# loss = net.mse(output, target)

						grads = tape.gradient(loss, net.model.trainable_weights)
						optimizer.apply_gradients(zip(grads, net.model.trainable_weights))

						net.train_loss_tracker.update_state(loss)

					mloss = net.train_loss_tracker.result()
					net.train_loss_tracker.reset_states()
					print("augmentation: {} \t loss_tracker: {:.4f} ".format(aug, float(mloss)))
					loss_log.write("augmentation: {} \t loss_tracker: {:.4f} ".format(aug, float(mloss)))
					# net.augments = aug
					# net.fit(x=batchTrainDataset, shuffle=True, \
					# 	steps_per_epoch=None, epochs=nEpoch, initial_epoch=0, \
					# 	verbose=2,  callbacks=[lossLog_tf, modelLog_tf], validation_data=batchValDataset, validation_steps=None)
				# validation
				for step, (x_batch_val, y_batch_val) in enumerate(batchValDataset):
					input = tf.cast(x_batch_val, dtype=tf.float32)
					target = tf.cast(y_batch_val, dtype=tf.float32)

					output = net(input)
					val_loss = loss_fn(target, output)
					# val_loss = net.mse(target, output)
					val_loss_tracker.update_state(val_loss)
					thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]
					total_acc = 0
					prediction_error = tf.math.abs(output-target)
					# tf.print(prediction_error, tf.shape(prediction_error))

					for thresh_hold in thresh_holds:
						acc = tf.where(prediction_error < thresh_hold, 1., 0.)
						acc = tf.math.reduce_mean(acc)
						total_acc += acc

					val_ma = total_acc / len(thresh_holds)
					val_ma_tracker.update_state(val_ma)

				print("\n Val Epoch: [{}/{}] \t loss: {:.4f} \t ma: {:.4f} \n".format(epoch, nEpoch, float(val_loss_tracker.result()), float(val_ma_tracker.result())))
				loss_log.write("\n Val Epoch: [{}/{}] \t loss: {:.4f} \t ma: {:.4f}\n".format(epoch, nEpoch, float(val_loss_tracker.result()), float(val_ma_tracker.result())))
				print("Time taken: {:.2f}s".format(time.time() - start_time))
				loss_log.write("Time taken: {:.2f}s".format(time.time() - start_time))
				val_ma_tracker.reset_states()
				val_loss_tracker.reset_states()

				if (epoch + 1) % 100 == 0:
					net.save_weights(filepath= outputPath + 'checkpoint/cp-weights-{:02d}.ckpt'.format(epoch))

		else:	
			if BN_flag == 3:
				modelLog_tf = tf.keras.callbacks.ModelCheckpoint(
					outputPath + 'checkpoint/cp-weights-{epoch:02d}.ckpt', monitor='val_loss', save_best_only=False,
					save_weights_only=True, save_freq=100)
				lossLog_tf  = tf.keras.callbacks.CSVLogger(outputPath + 'loss-log', append=True, separator=',')
				net.fit(x=trainGenerator, steps_per_epoch=nTrainStep, epochs=nEpoch, \
				verbose=2, callbacks=[modelLog_tf,lossLog_tf], validation_data=validGenerator, validation_steps=nValidStep)
			else:
				net.fit_generator(trainGenerator, steps_per_epoch=nTrainStep, epochs=nEpoch, \
				verbose=2, callbacks=[modelLog,lossLog], validation_data=validGenerator, validation_steps=nValidStep)
	else:
		for [x_batch, y_batch] in trainGenerator:
			print(x_batch.shape)
			print(y_batch.shape)
			print(y_batch[0])
			input_img = tf.convert_to_tensor(x_batch, dtype=tf.float32)
			print(input_img)
			cv2.imshow("input", input_img[0].eval(session=tf.compat.v1.Session())/255)
			imgs = net.g(input_img)

			cv2.imshow("output", imgs[0].eval(session=tf.compat.v1.Session())/255)
			cv2.waitKey(0)

	net.save(outputPath + 'model-final.h5')
	print(net.summary())
	

	
def test_dnn_multi(modelPath, imageDir_list, labelPath_list, outputPath, netType, flags, specs, BN_flag=0, pathID=0, ratio=1, pack_flag=False,
	diffAug=False, adv_step=0.2, n_repeats=3, eps=0.5, before_relu=False):
	
	## assigning variables
# 	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier    = flags[2]
# 	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
	nClass         = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	# print('\n\n\n')
	# print('********************************************')
	
	if fClassifier:
		print('Classification......')
	else:
		print('Regression......')

	### retrieve the test data
	if not pack_flag:
		testFeatures, testLabels = load_train_data_multi(imageDir_list, labelPath_list, nRep, fThreeCameras, ratio=ratio)
	else:
		testFeatures, testLabels = load_train_data_multi_pack(imageDir_list, labelPath_list, nRep, fThreeCameras, ratio=ratio)

	#testFeatures, testLabels = load_train_data(imageDir, labelPath, nRep, fThreeCameras, ratio=ratio)
	testFeatures = np.array(testFeatures)
	testLabels = np.array(testLabels)
	if pack_flag:
		testLabels = testLabels[:,0]
	n = len(testLabels)

	if fClassifier:
		print('\n######### Classification #########')
		testLabels = normalize_value(testLabels)
		testLabels = to_categorical(testLabels, num_classes = nClass)
	
	# print(testFeatures)
	print('The number of tested data: ' + str(testLabels.shape))
	print('********************************************')
	testData = []
	if not pack_flag:
		for i in range(len(testLabels)):
			image_path = testFeatures[i]
			if not os.path.isfile(image_path):
				print("image not exist: ", image_path)
				image_path = image_path.replace(".jpg", "_fake.png")
			img = resize_image(cv2.imread(image_path))
			testData.append(img)
	else:
		for i in range(len(testFeatures)):
			for j in range(len(testFeatures[i])):
				image_path = testFeatures[i][j]
				if not os.path.isfile(image_path):
					image_path = image_path.replace(".jpg", "_fake.png")
				img_1 = resize_image(cv2.imread(image_path))
				if j == 0:
					img = img_1
				else:
					img = np.concatenate((img, img_1), axis=2)

			#noise = np.random.uniform(low=0, high=255, size=(img.shape[0], img.shape[1], 1))
			#img = np.concatenate((img, noise), axis=2)
			testData.append(img)

	testData = np.array(testData)
	nChannel=3
	if pack_flag:
		testFeatures = testFeatures[:,0]
		nChannel = 3*len(imageDir_list)


	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
		if diffAug:
			print("test DiffAug_Net")
			net = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel, 
					augments="", adv_step=0., n_repeats=0, eps=0.)
			net(tf.ones((1, 66, 200, 3)))
			net.summary()
		else:
			if BN_flag == 3:
				net = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel, 
							adv_step=adv_step, n_repeats=n_repeats, eps=eps, before_relu=before_relu)
				net(tf.ones((1, 66, 200, 3)))
			else:
				net = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel)
	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)

	#print(net.layers[3].get_weights())
	# print(net.summary())
	
	## load model weights
	if modelPath != "":
		net.load_weights(modelPath)
		print("finished loading weight")
	
	if BN_flag == 3:
		inp = [net.featureX.input, net.head.input]
	else:
		if diffAug:
			inp = net.model.input
		else:
			inp = net.input                                           # input placeholder

	if BN_flag == 0:
		if diffAug:
			outputs = [layer.get_output_at(-1) for layer in net.model.layers]
		else:
			outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		print(outputs)
		last_conv_id = 10
	elif BN_flag == 1:
		outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 15
	elif BN_flag == 2:
		#independent_layer_ids = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 24, 25, 28, 29, 32, 33]
		BN_layer_ids = [4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 25, 26, 29, 30, 33, 34]
		outputs = []
		for i in range(len(net.layers)):
			if i == 0 or i == 1:
				continue
			layer = net.layers[i]
			if i in BN_layer_ids:
				outputs.append(layer.get_output_at(0))
			else:
				outputs.append(layer.get_output_at(0))
				outputs.append(layer.get_output_at(1))
		last_conv_id = 22
	elif BN_flag == 3:
		outputs_1 = [layer.get_output_at(-1) for layer in net.featureX.layers] 
		outputs_2 = [layer.get_output_at(-1) for layer in net.head.layers] 
		outputs = outputs_1[1:] + outputs_2[1:]
		last_conv_id = 10


	if BN_flag == 3:
		functor = K.function(inp, outputs )   # evaluation function
	else:
		functor = K.function([inp], outputs )   # evaluation function
	
	### predict and output
	if BN_flag <= 1:
		layer_outs = functor(testData)
		predictResults = layer_outs[-1]
	elif BN_flag == 3:
		layer_outs = functor([testData, net.featureX(testData)])
		predictResults = layer_outs[-1]
	else:
		layer_outs = functor([testData, testData])
		predictResults = layer_outs[-2+pathID]
	#predictResults = net.predict(testData)
	#score, acc = net.evaluate(testData, testLabels)

	BN_means = []
	BN_stds = []
	for layer_id in range(len(layer_outs)):
		#if layer_id not in [1, 3, 5, 7, 9]:
		#	continue

		layer_out = layer_outs[layer_id]
		#print(layer_out.shape)
		if layer_id <= last_conv_id:
			for i in range(layer_out.shape[3]):
				BN_means.append(np.mean(layer_out[:,:,:,i]))
				BN_stds.append(np.std(layer_out[:,:,:,i]))
		else:
			BN_means.append(np.mean(layer_out[:,:]))
			BN_stds.append(np.std(layer_out[:,:]))

	if outputPath != "":
		f_BN = open(outputPath.replace(ntpath.basename(outputPath), "BN_means.txt"),'w')
		#print(BN_means)
		#print(BN_stds)
		for mean in BN_means:
			f_BN.write("{:.5f}\n".format(mean))
		f_BN.close()
		f_BN = open(outputPath.replace(ntpath.basename(outputPath), "BN_stds.txt"),'w')
		for std in BN_stds:
			f_BN.write("{:.5f}\n".format(std))
		f_BN.close()

	print(outputPath)
	if outputPath != "":
		f = open(outputPath,'w')
	if fClassifier:
		predictResults = predictResults.argmax(axis=1)
		testLabels = testLabels.argmax(axis=1)
		correct_count = n - np.count_nonzero(predictResults-testLabels)
		print("accuracy: ", correct_count / (float)(n))

		if outputPath != "":
			f.write("accuracy: {:.5f}\n\n".format(correct_count / (float)(n)))
			f.write("{:^12} {:^12} {:^12} {:^12}\n".format("prediction", "groundtruth", "difference", "input"))
		
		for p in range(len(predictResults)):
	# 		if fClassifier:
	#  			f.write(str(np.argmax(p)))
	#  			print(np.argmax(p))
	# 		else: 
			# for regression
			imgName = os.path.basename(testFeatures[p])
			prediction = predictResults[p]
			groundTruth = testLabels[p]

			if outputPath != "":
				f.write("{:^12.0f} {:^12.0f} {:^12.0f} {:^12}".format(prediction, groundTruth, prediction-groundTruth, imgName))
				f.write('\n')
	else:
		prediction_error = predictResults.flatten() - testLabels
		mse_loss = np.mean(np.square(prediction_error))
		print("mse loss: " + str(mse_loss))

		if outputPath != "":
			f.write("mse loss: {:.5f}\n".format(mse_loss))

		#thresh_holds = [0.01, 0.033, 0.1, 0.33, 1, 3.3]
		thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]
		#thresh_holds = [1, 2, 4, 8]
		acc_list = []


		for thresh_hold in thresh_holds:
			if outputPath != "":
				image_fail_cases_folder = os.path.dirname(outputPath)+'/fail_cases_'+str(thresh_hold)
				if not os.path.exists(image_fail_cases_folder):
					os.mkdir(image_fail_cases_folder)

			acc = np.sum(np.abs(prediction_error) < thresh_hold) / len(testLabels)
			acc_list.append(acc)
			# print("accuracy (+-" + str(thresh_hold) + "): " + str(acc))

			if outputPath != "":
				f.write("accuracy (+-{:.3f}): {:.5f}\n".format(thresh_hold, acc))

				f_img_list_succ = open(outputPath.replace(ntpath.basename(outputPath), "img_list_"+str(thresh_hold)+"_succ.txt"),'w')
				f_img_list_fail = open(outputPath.replace(ntpath.basename(outputPath), "img_list_"+str(thresh_hold)+"_fail.txt"),'w')
				img_list_succ = testFeatures[np.abs(prediction_error) < thresh_hold]

				fail_flag = (np.abs(prediction_error) >= thresh_hold)

				#img_list_fail = testFeatures[np.abs(prediction_error) >= thresh_hold]
				#img_fail = testData[np.abs(prediction_error) >= thresh_hold]
				#print(len(img_list_succ))
				#print(len(img_list_fail))
				for img_file in img_list_succ:
					f_img_list_succ.write(str(img_file))
					f_img_list_succ.write('\n')

				for i in range(len(fail_flag)):
					if fail_flag[i] == True:
						img_file = testFeatures[i]
						f_img_list_fail.write(str(img_file))
						f_img_list_fail.write('\n')
						'''
						img = cv2.imread(img_file)
						#img_path = image_fail_cases_folder + "/gt_" + str(testLabels[i]) + "_pred_" + str(predictResults.flatten()[i]) + "_diff_" + str(prediction_error[i]) + "_" + os.path.basename(img_file)
						img_path = image_fail_cases_folder + "/gt_" + "{:.3f}".format(testLabels[i]) + \
							"_pred_" + "{:.3f}".format(predictResults.flatten()[i]) + \
							"_diff_" + "{:.3f}".format(prediction_error[i]) + \
							"_" + os.path.basename(img_file)
						cv2.imwrite(img_path, img)
						'''


				f_img_list_succ.close()
				f_img_list_fail.close()

		print("mean accuracy: " + str(np.mean(acc_list)))
		if outputPath != "":
			f.write("mean accuracy: {:.5f}\n\n".format(np.mean(acc_list)))

		
			f.write("{:^12} {:^12} {:^12} {:^12}\n".format("prediction", "groundtruth", "difference", "input"))
			
			for p in range(len(predictResults)):
		# 		if fClassifier:
		#  			f.write(str(np.argmax(p)))
		#  			print(np.argmax(p))
		# 		else: 
				# for regression
				imgName = os.path.basename(testFeatures[p])
				prediction = predictResults[p][0]
				groundTruth = testLabels[p]
				f.write("{:^12.3f} {:^12.3f} {:^12.3f} {:^12}".format(prediction, groundTruth, prediction-groundTruth, imgName))
				f.write('\n')

	if outputPath != "":
		f.close()

# 	for i in range(len(testLabels)):
# 		print([str('%.4f' % float(j)) for j in predictResults[i]])

			
	# print('********************************************')
	# print('\n\n\n')

	K.clear_session()

	return np.mean(acc_list)



def test_dnn_visualize(modelPath, imageDir, labelPath, outputPath, netType, flags, specs, BN_flag=0, pathID=0, radius=5):
	
	## assigning variables
# 	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier    = flags[2]
# 	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
	nClass         = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	step = 2
	
	print('\n\n\n')
	print('********************************************')
	
	if fClassifier:
		print('Classification......')
	else:
		print('Regression......')

	### retrieve the test data
	testFeatures, testLabels = load_train_data(imageDir, labelPath, nRep, fThreeCameras)
	testFeatures = np.array(testFeatures)
	testLabels = np.array(testLabels)
	n = len(testLabels)

	if fClassifier:
		print('\n######### Classification #########')
		testLabels = normalize_value(testLabels)
		testLabels = to_categorical(testLabels, num_classes = nClass)
	
	print(testFeatures)
	print('The number of tested data: ' + str(testLabels.shape))
	print('********************************************')

	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		net = create_nvidia_network(BN_flag, fClassifier, nClass)
	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)

	#print(net.layers[3].get_weights())
	print(net.summary())
	
	## load model weights
	if modelPath != "":
		net.load_weights(modelPath)

	inp = net.input                                           # input placeholder

	if BN_flag == 0:
		outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 10
	elif BN_flag == 1:
		outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 15
	elif BN_flag == 2:
		#independent_layer_ids = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 24, 25, 28, 29, 32, 33]
		BN_layer_ids = [4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 25, 26, 29, 30, 33, 34]
		outputs = []
		for i in range(len(net.layers)):
			if i == 0 or i == 1:
				continue
			layer = net.layers[i]
			if i in BN_layer_ids:
				outputs.append(layer.get_output_at(0))
			else:
				outputs.append(layer.get_output_at(0))
				outputs.append(layer.get_output_at(1))
		last_conv_id = 22


	functor = K.function([inp], outputs )   # evaluation function
	
	for i in range(len(testLabels)):
		image_path = testFeatures[i]
		label = testLabels[i]
		if not os.path.isfile(image_path):
			image_path = image_path.replace(".jpg", "_fake.png")

		img_ori = cv2.imread(image_path)
		img = resize_image(img_ori)

		rows = (int)((img.shape[0]-1)/step) + 1
		cols = (int)((img.shape[1]-1)/step) + 1

		testData = []
		for i in range(rows):
			for j in range(cols):
				img1 = img.copy()
				st_r = np.clip(i*step-radius, 0, img.shape[0]-1)
				ed_r = np.clip(i*step+radius, 0, img.shape[0]-1)
				st_c = np.clip(j*step-radius, 0, img.shape[1]-1)
				ed_c = np.clip(j*step+radius, 0, img.shape[1]-1)
				img1[st_r:ed_r+1, st_c:ed_c+1, :] = 0
				testData.append(img1)
		testData.append(img)
		testData = np.array(testData)

		### predict and output
		if BN_flag <= 1:
			layer_outs = functor(testData)
			predictResults = layer_outs[-1]
		else:
			layer_outs = functor([testData, testData])
			predictResults = layer_outs[-2+pathID]
		#predictResults = net.predict(testData)
		#score, acc = net.evaluate(testData, testLabels)

		abs_diff = np.abs(predictResults.flatten() - label)
		max_diff = np.max(abs_diff)

		heat_map = np.reshape(np.abs(np.array(predictResults.flatten())[:(rows*cols)] - label), (rows, cols))
		heat_map = heat_map / np.max(heat_map)
		#cv2.imshow("heat_map_small", heat_map)

		heat_map = cv2.resize(heat_map,(img_ori.shape[1],img_ori.shape[0]), interpolation = cv2.INTER_AREA)
		#cv2.imshow("heat_map", heat_map)
		#cv2.imshow("img_ori", img_ori)

		img_ori[:,:,0] = np.multiply(img_ori[:,:,0], heat_map)
		img_ori[:,:,1] = np.multiply(img_ori[:,:,1], heat_map)
		img_ori[:,:,2] = np.multiply(img_ori[:,:,2], heat_map)
		#cv2.imshow("combine_img", img_ori)

		outputFolder = os.path.dirname(outputPath)
		imageName = os.path.basename(image_path)
		val_folder = imageName.replace(".jpg", "")
		val_folder = imageName.replace(".png", "")
		#outputImagePath = outputFolder + '/' + imageName + '_(gt_' + str(label) + ')_(error_' + str(abs_diff[-1]) + ')_(max_diff_' + str(max_diff) + ').jpg'
		outputImagePath = outputFolder + '/' + imageName + '_(gt_{:.3f})_(error_{:.3f})_(max_diff_{:.3f}).jpg'.format(label, abs_diff[-1], max_diff)

		cv2.imwrite(outputImagePath, img_ori)
		#cv2.waitKey(1);
		
			
	print('********************************************')
	print('\n\n\n')
	


def visualize_dnn_on_image(modelPath, imagePath, label, outputPath, netType, flags, specs, radius=10, BN_flag=0, pathID=0):
	
	## assigning variables
# 	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier    = flags[2]
# 	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
	nClass         = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	print('\n\n\n')
	print('********************************************')
	
	if fClassifier:
		print('Classification......')
	else:
		print('Regression......')

	img = resize_image(cv2.imread(imagePath))
	#img = cv2.imread(imagePath)
	#img = cv2.resize(img,(200, 66), interpolation = cv2.INTER_AREA)
	#cv2.imshow("test", img)
	#cv2.waitKey(0)

	step = 2

	rows = (int)((img.shape[0]-1)/step) + 1
	cols = (int)((img.shape[1]-1)/step) + 1

	testData = []
	for i in range(rows):
		for j in range(cols):
			img1 = img.copy()
			st_r = np.clip(i*step-radius, 0, img.shape[0]-1)
			ed_r = np.clip(i*step+radius, 0, img.shape[0]-1)
			st_c = np.clip(j*step-radius, 0, img.shape[1]-1)
			ed_c = np.clip(j*step+radius, 0, img.shape[1]-1)
			img1[st_r:ed_r+1, st_c:ed_c+1, :] = 0
			testData.append(img1)

	testData = np.array(testData)

	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		net = create_nvidia_network(BN_flag, fClassifier, nClass)
	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)

	#print(net.layers[3].get_weights())
	print(net.summary())
	
	## load model weights
	if modelPath != "":
		net.load_weights(modelPath)

	inp = net.input                                           # input placeholder

	if BN_flag == 0:
		outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 10
	elif BN_flag == 1:
		outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 15
	elif BN_flag == 2:
		#independent_layer_ids = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 24, 25, 28, 29, 32, 33]
		BN_layer_ids = [4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 25, 26, 29, 30, 33, 34]
		outputs = []
		for i in range(len(net.layers)):
			if i == 0 or i == 1:
				continue
			layer = net.layers[i]
			if i in BN_layer_ids:
				outputs.append(layer.get_output_at(0))
			else:
				outputs.append(layer.get_output_at(0))
				outputs.append(layer.get_output_at(1))
		last_conv_id = 22


	functor = K.function([inp], outputs )   # evaluation function
	
	### predict and output
	if BN_flag <= 1:
		layer_outs = functor(testData)
		predictResults = layer_outs[-1]
	else:
		layer_outs = functor([testData, testData])
		predictResults = layer_outs[-2+pathID]
	#predictResults = net.predict(testData)
	#score, acc = net.evaluate(testData, testLabels)

	BN_means = []
	BN_stds = []
	for layer_id in range(len(layer_outs)):
		#if layer_id not in [1, 3, 5, 7, 9]:
		#	continue

		layer_out = layer_outs[layer_id]
		#print(layer_out.shape)
		if layer_id <= last_conv_id:
			for i in range(layer_out.shape[3]):
				BN_means.append(np.mean(layer_out[:,:,:,i]))
				BN_stds.append(np.std(layer_out[:,:,:,i]))
		else:
			BN_means.append(np.mean(layer_out[:,:]))
			BN_stds.append(np.std(layer_out[:,:]))

			
	f_BN = open(outputPath.replace(ntpath.basename(outputPath), "BN_means.txt"),'w')
	#print(BN_means)
	#print(BN_stds)
	for mean in BN_means:
		f_BN.write("{:.5f}\n".format(mean))
	f_BN.close()
	f_BN = open(outputPath.replace(ntpath.basename(outputPath), "BN_stds.txt"),'w')
	for std in BN_stds:
		f_BN.write("{:.5f}\n".format(std))
	f_BN.close()


	heat_map = np.reshape(np.abs(np.array(predictResults.flatten()) - label), (rows, cols))
	heat_map = heat_map / np.max(heat_map)
	cv2.imshow("heat_map_small", heat_map)

	img_ori = cv2.imread(imagePath)
	heat_map = cv2.resize(heat_map,(img_ori.shape[1],img_ori.shape[0]), interpolation = cv2.INTER_AREA)
	cv2.imshow("heat_map", heat_map)
	cv2.imshow("img_ori", img_ori)

	cv2.imshow("img_ori", img_ori)
	img_ori[:,:,0] = np.multiply(img_ori[:,:,0], heat_map)
	img_ori[:,:,1] = np.multiply(img_ori[:,:,1], heat_map)
	img_ori[:,:,2] = np.multiply(img_ori[:,:,2], heat_map)
	cv2.imshow("combine_img", img_ori)
	cv2.imwrite(outputPath, img_ori)
	cv2.waitKey(0);

# 	for i in range(len(testLabels)):
# 		print([str('%.4f' % float(j)) for j in predictResults[i]])

			
	print('********************************************')
	print('\n\n\n')



def read_float_list(file_name):
	x = []
	file_in = open(file_name, 'r')
	for y in file_in.read().split('\n'):
		if len(y) > 0:
			x.append(float(y))
	return x

def is_similar(val1, val2, val_thresh, percent_thresh):
	val_diff = abs(val1 - val2)
	percent_diff1 = abs(val_diff / val1)
	percent_diff2 = abs(val_diff / val2)

	if val_diff < val_thresh or (percent_diff1 < percent_thresh and percent_diff2 < percent_thresh):
		return True

	return False


def filter_dataset(modelPath, imageDir, labelPath, outputPath, netType, flags, specs, BN_flag=0, target_BN_folder="", filter_percent=0.1):
	

	## assigning variables
# 	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier    = flags[2]
# 	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
	nClass         = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	print('\n\n\n')
	print('********************************************')
	
	if fClassifier:
		print('Classification......')
	else:
		print('Regression......')

	### retrieve the test data
	testFeatures, testLabels = load_train_data(imageDir, labelPath, nRep, fThreeCameras)
	testFeatures = np.array(testFeatures)
	testLabels = np.array(testLabels)

	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		net = create_nvidia_network(BN_flag, fClassifier, nClass)
	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)

	#print(net.layers[3].get_weights())
	print(net.summary())
	
	## load model weights
	if modelPath != "":
		net.load_weights(modelPath)

	inp = net.input                                           # input placeholder

	if BN_flag == 0:
		outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 10
	elif BN_flag == 1:
		outputs = [layer.get_output_at(-1) for layer in net.layers]          # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 15
	elif BN_flag == 2:
		#independent_layer_ids = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 24, 25, 28, 29, 32, 33]
		BN_layer_ids = [4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 25, 26, 29, 30, 33, 34]
		outputs = []
		for i in range(len(net.layers)):
			if i == 0 or i == 1:
				continue
			layer = net.layers[i]
			if i in BN_layer_ids:
				outputs.append(layer.get_output_at(0))
			else:
				outputs.append(layer.get_output_at(0))
				outputs.append(layer.get_output_at(1))
		last_conv_id = 22


	functor = K.function([inp], outputs )   # evaluation function
		

	print(testFeatures)
	print('The number of tested data: ' + str(testLabels.shape))
	print('********************************************')
	f = open(outputPath,'w')
	similar_img_count = 0
	diff_score_list = []
	for id in range(len(testLabels)):
		image_path = testFeatures[id]
		label_value = testLabels[id]
		if not os.path.isfile(image_path):
			image_path = image_path.replace(".jpg", "_fake.png")
		img = resize_image(cv2.imread(image_path))
		testData=[img]

		testData = np.array(testData)

		### predict and output
		if BN_flag <= 1:
			layer_outs = functor(testData)
			predictResults = layer_outs[-1]
		else:
			layer_outs = functor([testData, testData])
			predictResults = layer_outs[-2]
		#predictResults = net.predict(testData)
		#score, acc = net.evaluate(testData, testLabels)


		BN_means = []
		BN_stds = []
		for layer_id in range(len(layer_outs)):
			#if layer_id not in [1, 3, 5, 7, 9]:
			#	continue

			layer_out = layer_outs[layer_id]
			if layer_id <= last_conv_id:
				for i in range(layer_out.shape[3]):
					BN_means.append(np.mean(layer_out[:,:,:,i]))
					BN_stds.append(np.std(layer_out[:,:,:,i]))
			else:
				BN_means.append(np.mean(layer_out[:,:]))
				BN_stds.append(np.std(layer_out[:,:]))

		BN_means_target = read_float_list(target_BN_folder + "/BN_means.txt")
		BN_stds_target = read_float_list(target_BN_folder + "/BN_stds.txt")

		SINGLE_FEATURE_PERCENT_THRESH = 1
		SINGLE_FEATURE_VALUE_THRESH = 3
		MODEL_PERCENT_THRESH = 0.5

		similar_feature_count = 0
		for i in range(len(BN_means)):
			mean_value_diff = abs(BN_means[i] - BN_means_target[i])
			mean_percent_diff = abs((BN_means[i] - BN_means_target[i])/BN_means_target[i])
			#print("base mean: ", BN_means_target[i], "  current mean: ", BN_means[i], "  mean_value_diff: ", mean_value_diff, "  mean_percent_diff: ", mean_percent_diff, "\n")
			
			std_value_diff = abs(BN_stds[i] - BN_stds_target[i])
			std_percent_diff = abs((BN_stds[i] - BN_stds_target[i])/BN_stds_target[i])
			#print("base std: ", BN_stds_target[i], "  current std: ", BN_stds[i], "  std_value_diff: ", std_value_diff, "  std_percent_diff: ", std_percent_diff, "\n")
			
			if is_similar(BN_means_target[i], BN_means[i], SINGLE_FEATURE_VALUE_THRESH, SINGLE_FEATURE_PERCENT_THRESH) and is_similar(BN_stds_target[i], BN_stds[i], SINGLE_FEATURE_VALUE_THRESH, SINGLE_FEATURE_PERCENT_THRESH):
				similar_feature_count += 1

		BN_means = np.array(BN_means)
		BN_means_target = np.array(BN_means_target)
		BN_stds = np.array(BN_stds)
		BN_stds_target = np.array(BN_stds_target)

		BN_mean_diff = BN_means - BN_means_target
		BN_std_diff = BN_stds - BN_stds_target

		'''
		mean_all_percent_1 = np.linalg.norm(BN_mean_diff) / np.linalg.norm(BN_means_target)
		mean_all_percent_2 = np.linalg.norm(BN_mean_diff) / np.linalg.norm(BN_means)
		std_all_percent_1 = np.linalg.norm(BN_std_diff) / np.linalg.norm(BN_stds_target)
		std_all_percent_2 = np.linalg.norm(BN_std_diff) / np.linalg.norm(BN_stds)

		#print("mean_all_percent_1 ", mean_all_percent_1, "mean_all_percent_2 ", mean_all_percent_2, "  std_all_percent_1 ", std_all_percent_1, "  std_all_percent_2 ", std_all_percent_2)
		
		# for honda
		MEAN_PERCENTAGE_THRESH = 1.2
		STD_PERCENTAGE_THRESH = 0.5

		# for udacity
		MEAN_PERCENTAGE_THRESH = 1.2
		STD_PERCENTAGE_THRESH = 0.8
		if mean_all_percent_1 < MEAN_PERCENTAGE_THRESH and mean_all_percent_2 < MEAN_PERCENTAGE_THRESH and std_all_percent_1 < MEAN_PERCENTAGE_THRESH and std_all_percent_2 < STD_PERCENTAGE_THRESH:
			f.write(os.path.basename(image_path) + ",,," + str(label_value) + "\n")
			similar_img_count += 1
		'''
		'''
		print(id, " ", similar_feature_count / len(BN_means))
		if similar_feature_count / len(BN_means) > MODEL_PERCENT_THRESH:
			f.write(os.path.basename(image_path) + ",,," + str(label_value) + "\n")
			similar_img_count += 1
		'''

		mean_std_ratio = np.linalg.norm(BN_means_target) / np.linalg.norm(BN_stds_target)
		diff_score = np.linalg.norm(BN_mean_diff) + mean_std_ratio * np.linalg.norm(BN_std_diff)
		#print("mean diff: ", np.linalg.norm(BN_mean_diff), "  std diff: ", np.linalg.norm(BN_std_diff), "  total: ", diff_score)
		#print("mean: ", np.linalg.norm(BN_means), "  std: ", np.linalg.norm(BN_stds))
		diff_score_list.append((id, diff_score, np.linalg.norm(BN_means), np.linalg.norm(BN_stds)))

	f.close()
		
	#print(diff_score_list)
	diff_score_list = sorted(diff_score_list, key=lambda diff_score: diff_score[1])
	#print(diff_score_list)

	f = open(outputPath,'w')
	for i in range(int(len(testLabels)*filter_percent)):
		id = diff_score_list[i][0]
		image_path = testFeatures[id]
		label_value = testLabels[id]
		#f.write(os.path.basename(image_path) + ",,," + str(label_value) + "\n")
		f.write(os.path.basename(image_path) + ",,," + str(label_value) + "," +str(diff_score_list[i][2]) + "," +str(diff_score_list[i][3])+"\n")
	f.close()

	print('similar_img_count ', int(len(testLabels)*filter_percent))
	print('total_count ', len(testLabels))
	print('ratio ', filter_percent)
	print('\n\n')

	
if __name__ == "__main__":
	print('\n')
	print("### This is the library file for testing. Please do not directly run it.")
	print('\n')
