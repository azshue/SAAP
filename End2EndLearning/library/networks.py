# This script is to specify different network architectures.

import numpy as np

from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers import Conv2D, Input, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import layers
from keras import activations
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import keras


def net_lstm(netType, nFramesSample):
	net = Sequential()
	
	if netType == 4:	## one-to-one
		net.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(66, 200, 3)))
	else:			   ## many-to-one or many-to-many
		net.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(nFramesSample, 66, 200, 3)))
	
	net.add(TimeDistributed(Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')))
	net.add(TimeDistributed(Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')))
	net.add(TimeDistributed(Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')))
	net.add(TimeDistributed(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001), activation='elu')))
	net.add(TimeDistributed(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001), activation='elu')))
	net.add(TimeDistributed(Flatten()))
	
	if netType == 3: 	## many-to-many
		net.add(LSTM(100, return_sequences=True))
		net.add(TimeDistributed(Dense(1)))
	else:			   ## many-to-one or one-to-one
		net.add(LSTM(100))
		net.add(Dense(1))
		
	net.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
	return net


def create_nvidia_network(BN_flag, fClassifier, nClass, nChannel=3, **kwargs):
	if BN_flag == 0:
		net = net_nvidia_1(fClassifier, nClass, nChannel)
	elif BN_flag == 1:
		net = net_nvidia_BN(fClassifier, nClass)
	elif BN_flag == 2:
		net = net_nvidia_AdvProp(fClassifier, nClass)
	elif BN_flag == 3:
		adv_step = kwargs['adv_step']
		n_repeats = kwargs['n_repeats']
		eps = kwargs['eps']
		before_relu = kwargs['before_relu']
		net = AdvBN_Net(fClassifier, nClass, adv_step, n_repeats, eps, before_relu)
	return net

		
'''
def net_nvidia(fClassifier, nClass):
	mainInput = Input(shape=(66,200,3))
	x1 = Lambda(lambda x: x/127.5 - 1.0)(mainInput)
	x1 = Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')(x1)
	x1 = Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')(x1)
	x1 = Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001), activation='elu')(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001), activation='elu')(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001), activation='elu')(x1)
	x2 = Flatten()(x1)
	z = Dense(100, kernel_regularizer=l2(0.001), activation='elu')(x2)
	z = Dense(50,  kernel_regularizer=l2(0.001), activation='elu')(z)
	z = Dense(10,  kernel_regularizer=l2(0.001), activation='elu')(z)
	if fClassifier:
		if nClass > 2:
			mainOutput = Dense(nClass, activation='softmax')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
		else:
			mainOutput = Dense(1, activation='sigmoid')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
	else:
		mainOutput = Dense(1)(z)
		net = Model(inputs = mainInput, outputs = mainOutput)
		net.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])
	return net
'''

def mean_accuracy(y_true, y_pred):
	
	thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]

	res_list = []
	for thresh_hold in thresh_holds:
		res_list.append(tf.math.reduce_mean(tf.to_float(tf.keras.backend.abs(y_true-y_pred) > thresh_hold)))

	MA = tf.math.reduce_mean(res_list)
	
	return MA

## For tensorflow >= 2.0
def mean_accuracy_tf(y_true, y_pred):
	
	thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]

	res_list = []
	for thresh_hold in thresh_holds:
		res_list.append(tf.math.reduce_mean(tf.cast(tf.keras.backend.abs(y_true-y_pred) > thresh_hold, dtype=tf.float32)))

	MA = tf.math.reduce_mean(res_list)
	
	return MA
	

def net_nvidia_1(fClassifier, nClass, nChannel=3):
	mainInput = Input(shape=(66,200,nChannel))
	x1 = Lambda(lambda x: x/127.5 - 1.0)(mainInput)
	x1 = Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x2 = Flatten()(x1)
	z = Dense(100, kernel_regularizer=l2(0.001))(x2)
	z = layers.Activation(activations.elu)(z)
	z = Dense(50,  kernel_regularizer=l2(0.001))(z)
	z = layers.Activation(activations.elu)(z)
	z = Dense(10,  kernel_regularizer=l2(0.001))(z)
	z = layers.Activation(activations.elu)(z)
	if fClassifier:
		if nClass > 2:
			mainOutput = Dense(nClass, activation='softmax')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
		else:
			mainOutput = Dense(1, activation='sigmoid')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
	else:
		mainOutput = Dense(1)(z)
		net = Model(inputs = mainInput, outputs = mainOutput)
		net.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=[mean_accuracy])
		#net.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

	#print(net.summary())
	return net


def net_nvidia_BN(fClassifier, nClass, lr=1e-4):
	mainInput = Input(shape=(66,200,3))
	x1 = Lambda(lambda x: x/127.5 - 1.0)(mainInput)
	x1 = Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = BatchNormalization()(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = BatchNormalization()(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = BatchNormalization()(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = BatchNormalization()(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = BatchNormalization()(x1)
	x1 = layers.Activation(activations.elu)(x1)
	x2 = Flatten()(x1)
	z = Dense(100, kernel_regularizer=l2(0.001))(x2)
	z = BatchNormalization()(z)
	z = layers.Activation(activations.elu)(z)
	z = Dense(50,  kernel_regularizer=l2(0.001))(z)
	z = BatchNormalization()(z)
	z = layers.Activation(activations.elu)(z)
	z = Dense(10,  kernel_regularizer=l2(0.001))(z)
	z = BatchNormalization()(z)
	z = layers.Activation(activations.elu)(z)
	if fClassifier:
		if nClass > 2:
			mainOutput = Dense(nClass, activation='softmax')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
		else:
			mainOutput = Dense(1, activation='sigmoid')(z)
			net = Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
	else:
		mainOutput = Dense(1)(z)
		net = Model(inputs = mainInput, outputs = mainOutput)
		net.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['accuracy'])
	return net


def net_nvidia_AdvProp(fClassifier, nClass):
	image_input_1 = Input(shape=(66,200,3), name='images_1')  # Variable-length sequence of ints
	image_input_2 = Input(shape=(66,200,3), name='images_2')  # Variable-length sequence of ints

	lambda0 = Lambda(lambda x: x/127.5 - 1.0)
	conv1 = Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	elu_c1 = layers.Activation(activations.elu)
	conv2 = Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	elu_c2 = layers.Activation(activations.elu)
	conv3 = Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
	elu_c3 = layers.Activation(activations.elu)
	conv4 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))
	elu_c4 = layers.Activation(activations.elu)
	conv5 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))
	elu_c5 = layers.Activation(activations.elu)

	flat = Flatten()

	dense1 = Dense(100, kernel_regularizer=l2(0.001))
	elu_d1 = layers.Activation(activations.elu)
	dense2 = Dense(50,  kernel_regularizer=l2(0.001))
	elu_d2 = layers.Activation(activations.elu)
	dense3 = Dense(10,  kernel_regularizer=l2(0.001))
	elu_d3 = layers.Activation(activations.elu)
	dense4 = Dense(1)

	x1 = lambda0(image_input_1)
	x1 = conv1(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_c1(x1)
	x1 = conv2(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_c2(x1)
	x1 = conv3(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_c3(x1)
	x1 = conv4(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_c4(x1)
	x1 = conv5(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_c5(x1)
	x1 = flat(x1)

	x1 = dense1(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_d1(x1)
	x1 = dense2(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_d2(x1)
	x1 = dense3(x1)
	x1 = BatchNormalization()(x1)
	x1 = elu_d3(x1)
	output1 = dense4(x1)

	x2 = lambda0(image_input_2)
	x2 = conv1(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_c1(x2)
	x2 = conv2(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_c2(x2)
	x2 = conv3(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_c3(x2)
	x2 = conv4(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_c4(x2)
	x2 = conv5(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_c5(x2)
	x2 = flat(x2)

	x2 = dense1(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_d1(x2)
	x2 = dense2(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_d2(x2)
	x2 = dense3(x2)
	x2 = BatchNormalization()(x2)
	x2 = elu_d3(x2)
	output2 = dense4(x2)

	net = Model(inputs=[image_input_1, image_input_2], outputs=[output1, output2], name='Nvidia_AdvProp')
	net.compile(optimizer=Adam(lr=1e-4),
				  loss=["mse", 'mse'],
				  loss_weights=[1, 1], metrics=['accuracy'])
	return net

'''
class Gaussian_noise_layer(layers.Layer):
	def __init__(self, initializer="he_normal", **kwargs):
		super(Gaussian_noise_layer, self).__init__(**kwargs)
		self.initializer = keras.initializers.get(initializer)

	def build(self, input_shape):
		self.std = self.add_weight(
			shape=[1],
			initializer=self.initializer,
			name="std",
			trainable=True,
		)

	def call(self, inputs):
		noise = tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=self.std*1000, dtype=tf.float32) 
		return inputs + noise
'''

class Gaussian_noise_layer(keras.layers.Layer):
	def __init__(self):
		super(Gaussian_noise_layer, self).__init__()
		w_init = tf.random_normal_initializer()
		initial_value=w_init(shape=(1,1), dtype="float32")
		self.std = tf.Variable(initial_value=initial_value,trainable=True)
		print(self.std)
		print('!!!!!!!!!!!!!!!!!!!')

	def call(self, inputs):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
		print(self.std.eval(session=tf.compat.v1.Session()))
		print('!!!!!!!!!!!!!!!!!!!')
		noise = tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=tf.reduce_sum(self.std).eval(session=tf.compat.v1.Session())[0], dtype=tf.float32) 
		return inputs + noise

'''
class Gaussian_noise_layer(keras.layers.Layer):
	def __init__(self):
		units=32
		input_dim=32
		super(Gaussian_noise_layer, self).__init__()
		w_init = tf.random_normal_initializer()
		self.w = tf.Variable(
			initial_value=w_init(shape=(input_dim, units), dtype="float32"),
			trainable=True,
		)
		b_init = tf.zeros_initializer()
		self.b = tf.Variable(
			initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
		)

	def call(self, inputs):
		return tf.matmul(inputs, self.w) + self.b
'''

class GAN_Nvidia():
	def __init__(self):
		self.img_rows = 66
		self.img_cols = 200
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)

		#optimizer = Adam(0.0002, 0.5)
		optimizer = Adam(lr=1e-4)

		# Build and compile the discriminator
		self.d = self.build_discriminators()
		self.d.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

		# Build the generator
		self.g = self.build_generators()

		# The generator takes noise as input and generated imgs
		z = Input(shape=self.img_shape)
		img_gene = self.g(z)

		# For the combined model we will only train the generators
		self.d.trainable = False

		# The valid takes generated images as input and determines validity
		valid = self.d(img_gene)

		# The combined model  (stacked generators and discriminators)
		# Trains generators to fool discriminators
		self.combined = Model(z, valid)
		self.combined.compile(optimizer=optimizer, loss=self.generator_loss, metrics=['accuracy'])

	def generator_loss(self, y_true, y_pred):
		mse = tf.keras.losses.MeanSquaredError()
		return -mse(y_true, y_pred)

	def gaussian_noise_layer(self, input_layer, std):
		noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
		return input_layer + noise

	def build_generators(self):

		mainInput = Input(shape=self.img_shape)
		mainOutput = Gaussian_noise_layer()(mainInput)

		model = Model(inputs = mainInput, outputs = mainOutput)

		print("********************** Generator Model *****************************")
		model.summary()

		return model

	def build_discriminators(self):

		mainInput = Input(shape=self.img_shape)

		x1 = Lambda(lambda x: x/127.5 - 1.0)(mainInput)
		x1 = Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
		x1 = layers.Activation(activations.elu)(x1)
		x1 = Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
		x1 = layers.Activation(activations.elu)(x1)
		x1 = Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
		x1 = layers.Activation(activations.elu)(x1)
		x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
		x1 = layers.Activation(activations.elu)(x1)
		x1 = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
		x1 = layers.Activation(activations.elu)(x1)
		x2 = Flatten()(x1)
		z = Dense(100, kernel_regularizer=l2(0.001))(x2)
		z = layers.Activation(activations.elu)(z)
		z = Dense(50,  kernel_regularizer=l2(0.001))(z)
		z = layers.Activation(activations.elu)(z)
		z = Dense(10,  kernel_regularizer=l2(0.001))(z)
		z = layers.Activation(activations.elu)(z)

		mainOutput = Dense(1)(z)

		model = Model(inputs = mainInput, outputs = mainOutput)
		print("********************** Discriminator Model *****************************")
		model.summary()

		return model


class FeatureX(tf.keras.layers.Layer):
	def __init__(self, before_relu, nChannel=3):
		super(FeatureX, self).__init__()
		self.input_layer = tf.keras.Input(shape=(66, 200, nChannel))
		self.conv1 = tf.keras.layers.Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
		self.conv2 = tf.keras.layers.Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
		self.before_relu = before_relu

	def call(self, input):
		input = tf.cast(input, tf.float32)
		x = tf.keras.layers.Lambda(lambda x: x/127.5 - 1.0)(input)
		x = self.conv1(x)
		x = tf.nn.elu(x)
		x = self.conv2(x)
		if self.before_relu:
			return x
		x = tf.nn.elu(x)
		return x

class Head(tf.keras.layers.Layer):
	def __init__(self, fClassifier, nClass):
		super(Head, self).__init__()
		self.conv3 = tf.keras.layers.Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))
		self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))
		self.conv5 = tf.keras.layers.Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))
		self.flat = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(100, kernel_regularizer=l2(0.001))
		self.dense2 = tf.keras.layers.Dense(50,  kernel_regularizer=l2(0.001))
		self.dense3 = tf.keras.layers.Dense(10,  kernel_regularizer=l2(0.001))
		self.dense_cls = tf.keras.layers.Dense(nClass, activation='softmax')
		self.dense = tf.keras.layers.Dense(1)
		self.fClassifier = fClassifier

	def call(self, input):
		input = tf.nn.elu(input)
		x = self.conv3(input)
		x = tf.nn.elu(x)
		x = self.conv4(x)
		x = tf.nn.elu(x)
		x = self.conv5(x)
		x = tf.nn.elu(x)
		x = self.flat(x)
		x = self.dense1(x)
		x = tf.nn.elu(x)
		x = self.dense2(x)
		x = tf.nn.elu(x)
		x = self.dense3(x)
		x = tf.nn.elu(x)
		if self.fClassifier:
			x = self.dense_cls(x)
		else:
			x = self.dense_cls(x)


class AdvBN_Net(tf.keras.Model):
	def __init__(self, fClassifier, nClass, adv_step, n_repeats, eps, before_relu, nChannel=3):
		super(AdvBN_Net, self).__init__()
		self.img_rows = 66
		self.img_cols = 200
		self.nChannel = nChannel
		self.img_shape = (self.img_rows, self.img_cols, self.nChannel)
		self.feature_shape = (14, 47, 36)

		self.fClassifier = fClassifier
		self.nClass = nClass
		self.adv_step = adv_step
		tf.print(n_repeats)
		self.n_repeats = int(n_repeats)
		self.eps = eps
		self.before_relu = before_relu
		self.noise_batch_mean = None
		self.noise_batch_std = None

		if fClassifier:
			self.output_dim = nClass
		else:
			self.output_dim = 1

		# self.featureX = FeatureX(before_relu)
		self.featureX = self.get_featureX()
		self.featureX.trainable = False
		# self.head = Head(fClassifier, nClass)
		self.head = self.get_head()

	def get_featureX(self):
		mainInput = tf.keras.Input(shape=self.img_shape)
		x = tf.keras.layers.Lambda(lambda x: x/127.5 - 1.0)(mainInput)
		x = tf.keras.layers.Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x)
		x = tf.nn.elu(x)
		x = tf.keras.layers.Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x)
		if self.before_relu:
			mainOutput = x
		else:
			mainOutput = tf.nn.elu(x)
		model = tf.keras.Model(inputs=mainInput, outputs=mainOutput)
		print("**********************  FeatureX *****************************")
		model.summary()

		return model

	def get_head(self):
		mainInput = tf.keras.Input(shape=self.feature_shape)
		x = tf.nn.elu(mainInput)
		x = tf.keras.layers.Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x)
		x = tf.nn.elu(x)
		x = tf.keras.layers.Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x)
		x = tf.nn.elu(x)
		x = tf.keras.layers.Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x)
		x = tf.nn.elu(x)
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(100, kernel_regularizer=l2(0.001))(x)
		x = tf.nn.elu(x)
		x = tf.keras.layers.Dense(50,  kernel_regularizer=l2(0.001))(x)
		x = tf.nn.elu(x)
		x = tf.keras.layers.Dense(10,  kernel_regularizer=l2(0.001))(x)
		x = tf.nn.elu(x)
		if self.fClassifier:
			mainOutput = tf.keras.layers.Dense(self.nClass, activation='softmax')(x)
		else:
			mainOutput = tf.keras.layers.Dense(1)(x)
		model = tf.keras.Model(inputs=mainInput, outputs=mainOutput)
		print("**********************  Head *****************************")
		model.summary()

		return model


	# def build(self, input_shape):
	# 	super(AdvBN_Net, self).build(input_shape)
	
	def compile(self, h_optimizer, loss_fn, h_metrics):
		super(AdvBN_Net, self).compile()
		self.h_optimizer = h_optimizer
		self.loss_fn = loss_fn
		self.h_metrics = h_metrics
		self.train_loss_tracker = tf.keras.metrics.Mean(name="loss")
		self.train_ma_tracker = tf.keras.metrics.Mean(name="ma_mean")
	
	def call(self, input):
		x = self.featureX(input)
		x = self.head(x)
		return x

	def calc_mean_std(self, feature, delta=1e-5):
		size = feature.get_shape().as_list()
		assert (len(size) == 4)
		C = size[3]
		feature = tf.transpose(feature, perm=[3, 0, 1, 2])
		feature = tf.reshape(feature, [C, -1])
		feat_mean = tf.keras.backend.mean(feature, axis=1)
		feat_mean = tf.reshape(feat_mean, [1, 1, 1, C])
		# feat_mean = feat.permute(1, 0, 2, 3).reshape(C, -1).mean(dim=1).view(1, C, 1, 1)
		# feat_var = feat.permute(1, 0, 2, 3).reshape(C, -1).var(dim=1) + delta
		feat_std = tf.keras.backend.std(feature, axis=1) + delta
		feat_std = tf.reshape(feat_std, [1, 1, 1, C])
		# feat_std = feat_var.sqrt().view(1, C, 1, 1)
		return feat_mean, feat_std

	def perturb(self, feature, target):
		# tf.print('perturb')
		size = feature.get_shape().as_list()
		noise_size = [1, 1, 1, size[3]]
		if self.noise_batch_mean is None:
			self.noise_batch_mean = tf.Variable(tf.ones(noise_size))
			self.noise_batch_std = tf.Variable(tf.ones(noise_size))
		else:
			self.noise_batch_mean.assign(tf.ones(noise_size))
			self.noise_batch_std.assign(tf.ones(noise_size))
		# noise_batch_mean = tf.ones(noise_size)
		# noise_batch_std = tf.ones(noise_size)

		ori_mean, ori_std = self.calc_mean_std(feature)
		normalized_feature = feature - ori_mean
		# tf.print(self.n_repeats)
		for _iter in range(self.n_repeats):
			with tf.GradientTape() as tape:
				new_mean = ori_mean * self.noise_batch_mean
				new_std = self.noise_batch_std
				adv_feature = normalized_feature * new_std + new_mean

				input_feature = adv_feature

				output = self.head(input_feature)
				loss = self.loss_fn(output, target)

			grads_mean, grads_std = tape.gradient(loss, [self.noise_batch_mean, self.noise_batch_std])

			print(grads_mean)
			self.noise_batch_mean.assign_add(self.adv_step * tf.keras.backend.sign(grads_mean))
			self.noise_batch_std.assign_add(self.adv_step * tf.keras.backend.sign(grads_std))
			self.noise_batch_mean.assign(tf.keras.backend.clip(self.noise_batch_mean, min_value=1-self.eps, max_value=1+self.eps))
			self.noise_batch_std.assign(tf.keras.backend.clip(self.noise_batch_std, min_value=1-self.eps, max_value=1+self.eps))
			# tf.print("nois_batch_mean", self.noise_batch_mean)

		new_mean = ori_mean * self.noise_batch_mean
		new_std = self.noise_batch_std
		adv_feature = normalized_feature * new_std + new_mean
		out_feature = adv_feature

		# tf.debugging.assert_none_equal(out_feature, feature)
		
		return out_feature

	def train_step(self, batch_data):
		input, target = batch_data
		# tf.print(target.get_shape())

		feature = self.featureX(input)

		clean_feature = feature
		input_feature = tf.identity(feature)
		adv_feature = self.perturb(input_feature, target)

		with tf.GradientTape() as tape:
			adv_output = self.head(adv_feature)
			clean_output = self.head(clean_feature)

			adv_loss = self.loss_fn(adv_output, target)
			clean_loss = self.loss_fn(clean_output, target)
			loss = adv_loss + clean_loss

		grads = tape.gradient(loss, self.head.trainable_weights)
		self.h_optimizer.apply_gradients(zip(grads, self.head.trainable_weights))
		ma = self.h_metrics(clean_output, target)

		self.train_ma_tracker.update_state(ma)
		self.train_loss_tracker.update_state(loss)

		return{
			"adv_loss": adv_loss,
			"clean_loss": clean_loss,
			"total_loss": loss,
			"loss_tracker": self.train_loss_tracker.result(),
			"mean_accuracy": ma,
			"ma_tracker": self.train_ma_tracker.result()
		}
	
	def test_step(self, batch_data):
		input, target = batch_data

		feature = self.featureX(input)
		output = self.head(feature)

		loss = self.loss_fn(output, target)
		ma = self.h_metrics(output, target)

		return{
			"loss": loss,
			"mean_accuracy": ma
			# m.name: m.result() for m in self.metric,
		}


	
if __name__ == "__main__":
	print('\n')
	print("### This is the file specifying different network architectures. Please do not run it directly.")
	print('\n')
	# import h5py
	# kv = h5py.File("./Data/udacityA_nvidiaB_results/train_results/trainB/model-final.h5", "r")

	# model = create_nvidia_network(BN_flag=3, fClassifier=False, nClass=49, nChannel=3, 
	# 					adv_step=0.2, n_repeats=6, eps=1.1, before_relu=False)
	
