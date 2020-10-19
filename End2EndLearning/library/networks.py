# This script is to specify different network architectures.

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import keras

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
	if 'augments' in kwargs:
		augments = kwargs['augments']
		adv_step = kwargs['adv_step']
		n_repeats = kwargs['n_repeats']
		eps = kwargs['eps']
		net = DiffAug_Net(fClassifier, nClass, augments, adv_step, n_repeats, eps)
	else:
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
		res_list.append(tf.math.reduce_mean(tf.cast(tf.keras.backend.abs(y_true-y_pred) < thresh_hold, dtype=tf.float32)))

	MA = tf.math.reduce_mean(res_list)
	
	return MA
	

def net_nvidia_1(fClassifier, nClass, nChannel=3):
	mainInput = tf.keras.Input(shape=(66,200,nChannel))
	x1 = tf.keras.layers.Lambda(lambda x: x/127.5 - 1.0)(mainInput)
	x1 = tf.keras.layers.Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = tf.nn.elu(x1)
	x1 = tf.keras.layers.Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = tf.nn.elu(x1)
	x1 = tf.keras.layers.Conv2D(48, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = tf.nn.elu(x1)
	x1 = tf.keras.layers.Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = tf.nn.elu(x1)
	x1 = tf.keras.layers.Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001))(x1)
	x1 = tf.nn.elu(x1)
	x2 = tf.keras.layers.Flatten()(x1)
	z = tf.keras.layers.Dense(100, kernel_regularizer=l2(0.001))(x2)
	z = tf.nn.elu(z)
	z = tf.keras.layers.Dense(50,  kernel_regularizer=l2(0.001))(z)
	z = tf.nn.elu(z)
	z = tf.keras.layers.Dense(10,  kernel_regularizer=l2(0.001))(z)
	z = tf.nn.elu(z)
	if fClassifier:
		if nClass > 2:
			mainOutput = tf.keras.layers.Dense(nClass, activation='softmax')(z)
			net = tf.keras.Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
		else:
			mainOutput = tf.keras.layers.Dense(1, activation='sigmoid')(z)
			net = tf.keras.Model(inputs = mainInput, outputs = mainOutput)
			net.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
	else:
		mainOutput = tf.keras.layers.Dense(1)(z)
		net = tf.keras.Model(inputs = mainInput, outputs = mainOutput)
		net.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=[mean_accuracy])
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


class DiffAug_Net(tf.keras.Model):
	def __init__(self, fClassifier, nClass, augments="", adv_step=0.2, n_repeats=3, eps=0.5, nChannel=3):
		super(DiffAug_Net, self).__init__()
		self.img_rows = 66
		self.img_cols = 200
		self.nChannel = nChannel
		self.img_shape = (self.img_rows, self.img_cols, self.nChannel)

		self.fClassifier = fClassifier
		self.nClass = nClass
		self.augments = augments
		self.adv_step = adv_step
		self.n_repeats = int(n_repeats)
		self.eps = eps

		self.delta = None

		if fClassifier:
			self.output_dim = nClass
		else:
			self.output_dim = 1

		self.model = self.get_model()

		self.train_loss_tracker = tf.keras.metrics.Mean(name="loss")
		self.train_ma_tracker = tf.keras.metrics.Mean(name="ma_mean")

	def get_model(self):
		mainInput = tf.keras.Input(shape=self.img_shape)
		x = tf.keras.layers.Lambda(lambda x: x/127.5 - 1.0)(mainInput)
		x = tf.keras.layers.Conv2D(24, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x)
		x = tf.nn.elu(x)
		x = tf.keras.layers.Conv2D(36, (5, 5), strides=(2,2), padding='valid', kernel_regularizer=l2(0.001))(x)
		x = tf.nn.elu(x)
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

		return model

	def bgr2hsv(self, img):
		out = tf.zeros_like(img)
		img = img / 255.0

		# v channel
		max_c = tf.reduce_max(img, axis=-1)

		out_v = max_c
		# s channel
		min_c = tf.reduce_min(img, axis=-1)
		delta = max_c - min_c
		out_s = tf.math.divide_no_nan(delta, max_c)
		out_s = tf.where(delta == 0.0, 0.0, out_s)
		# h channel
		tmp_1 = (img[..., 1] - img[..., 0]) / delta[...]
		out_1 = tf.where(img[..., 2] == max_c, tmp_1, 0)

		tmp_2 = 2.0 + (img[..., 0] - img[..., 2]) / delta[...]
		out_2 = tf.where(img[..., 1] == max_c, tmp_2, out_1)

		tmp_3 = 4.0 + (img[..., 2] - img[..., 1]) / delta[...]
		out_h = tf.where(img[..., 0] == max_c, tmp_3, out_2)

		out_h = (out_h / 6.0) % 1.0
		out_h = tf.where(delta == 0.0, 0.0, out_h)

		out = tf.stack([out_h, out_s, out_v], axis=-1)
		out = tf.where(tf.raw_ops.IsNan(x=out), 0.0, out)
		return out
	
	def hsv2bgr(self, img):
		hi = tf.raw_ops.Floor(x=img[..., 0] * 6)
		f = img[..., 0] * 6 - hi
		p = img[..., 2] * (1 - img[..., 1])
		q = img[..., 2] * (1 - f * img[..., 1])
		t = img[..., 2] * (1 - (1 - f) * img[..., 1])
		v = img[..., 2]

		hi = tf.cast(tf.stack([hi, hi, hi], axis=-1), tf.int32) % 6
		out_1 = tf.where(hi == 0, tf.stack([v, t, p], axis=-1), 0)
		out_2 = tf.where(hi == 1, tf.stack([q, v, p], axis=-1), out_1)
		out_3 = tf.where(hi == 2, tf.stack([p, v, t], axis=-1), out_2)
		out_4 = tf.where(hi == 3, tf.stack([p, q, v], axis=-1), out_3)
		out_5 = tf.where(hi == 4, tf.stack([t, p, v], axis=-1), out_4)
		out = tf.where(hi == 5, tf.stack([v, p, q], axis=-1), out_5)

		out = out[..., ::-1]
		out = out * 255.
		return out

	#1
	# https://github.com/tensorflow/addons/blob/v0.11.2/tensorflow_addons/image/filters.py#L226
	def blur(self, img, sigma, size=20):
		original_ndims = tfa.image.utils.get_ndims(img)
		img = tfa.image.utils.to_4D_image(img)

		x = tf.range(-size //2 + 1, size // 2 + 1)
		x = tf.cast(x ** 2, sigma.dtype)
		x = tf.exp(-x / (2.0 * (sigma ** 2)))
		x = x / tf.reduce_sum(x)
		x1 = tf.reshape(x, [size, 1])
		x2 = tf.reshape(x, [1, size])

		gaussian_kernel = tf.matmul(x1, x2)
		channels = tf.shape(img)[3]
		gaussian_kernel = tf.repeat(gaussian_kernel, channels)
		gaussian_kernel = tf.reshape(gaussian_kernel, [size, size, channels, 1])

		paddings = [[0, 0], 
					[(size - 1) // 2, size - 1 - (size - 1) // 2],
					[(size - 1) // 2, size - 1 - (size - 1) // 2],
					[0, 0]]
		img = tf.pad(img, paddings)

		output = tf.nn.depthwise_conv2d(
			input=img,
			filter=gaussian_kernel,
			strides=(1, 1, 1, 1),
			padding="VALID"
		)
		output = tfa.image.utils.from_4D_image(output, original_ndims)
		return output
	#2
	def gaussian(self, img, gaussian):
		img = img + 255.0 * tf.squeeze(gaussian)
		img =  tf.clip_by_value(img, 0, 255)
		return img
	#3
	# def distrot(self):

	#R
	def color_R(self, img, magnitude):
		output = tf.stack([img[..., 0], img[..., 1], img[..., 2] * (1 + magnitude)], axis=-1)
		img = tf.clip_by_value(output, 0, 255)
		return img
	#G
	def color_G(self, img, magnitude):
		output = tf.stack([img[..., 0], img[..., 1] * (1 + magnitude), img[..., 2]], axis=-1)
		img = tf.clip_by_value(output, 0, 255)
		return img
	#B
	def color_B(self, img, magnitude):
		output = tf.stack([img[..., 0] * (1 + magnitude), img[..., 1], img[..., 2]], axis=-1)
		img = tf.clip_by_value(output, 0, 255)
		return img
	#H
	def color_H(self, img, magnitude):
		hsv_img = self.bgr2hsv(img)
		output = tf.stack([hsv_img[..., 0] * (1 + magnitude), hsv_img[..., 1], hsv_img[..., 2]], axis=-1)
		output = tf.clip_by_value(output, 0, 1.0)
		img = self.hsv2bgr(output)
		return img
	#S
	def color_S(self, img, magnitude):
		hsv_img = self.bgr2hsv(img)
		output = tf.stack([hsv_img[..., 0], hsv_img[..., 1] * (1 + magnitude), hsv_img[..., 2]], axis=-1) 
		output = tf.clip_by_value(output, 0, 1.0)
		img = self.hsv2bgr(output)
		return img
	#V
	def color_V(self, img, magnitude):
		hsv_img = self.bgr2hsv(img)
		output = tf.stack([hsv_img[..., 0], hsv_img[..., 1], hsv_img[..., 2] * (1 + magnitude)], axis=-1) 
		output = tf.clip_by_value(output, 0, 1.0)
		img = self.hsv2bgr(output)
		return img
	#7
	def saturation(self, img, magnitude):
		mean = tf.reduce_mean(img, axis=3, keepdims=True)
		img = (img - mean) * magnitude + mean
		return img
	#8
	def contrast(self, img, magnitude):
		mean = tf.reduce_mean(img, axis=[1, 2, 3], keepdims=True)
		img = (img - mean) * magnitude + mean
		return img
	#9
	def brightness(self, img, magnitude):
		img = img + magnitude
		return img

	def compile(self, h_optimizer, loss_fn, h_metrics):
		super(DiffAug_Net, self).compile()
		self.h_optimizer = h_optimizer
		self.loss_fn = loss_fn
		self.h_metrics = h_metrics
		self.train_loss_tracker = tf.keras.metrics.Mean(name="loss")
		self.train_ma_tracker = tf.keras.metrics.Mean(name="ma_mean")

	def mse(self, y_true, y_pred):
		squared_difference = tf.square(y_true - y_pred)
		return tf.reduce_mean(squared_difference, axis=-1)

	def call(self, input):
		x = self.model(input)
		return x

	def my_train_step(self, batch_data, augs):
		input, target = batch_data
		input = tf.cast(input, dtype=tf.float32)
		target = tf.cast(target, dtype=tf.float32)

		# for aug in self.augments:
		for aug in augs:
			if self.delta is None:
				self.delta = tf.Variable(tf.zeros([1]))
			else:
				self.delta.assign(tf.zeros([1]))

			for _iter in range(self.n_repeats):
				with tf.GradientTape() as tape:
					if aug == '1': # gaussian blur
						aug_op = getattr(self, "blur")
						param = self.delta * 100
						param_min = 0.0
					elif aug == '2': # gaussian noise
						aug_op = getattr(self, "gaussian")
						dist = tfp.distributions.Normal(0, self.delta)
						param = dist.sample([66, 200, 3])
						param_min = 0.0
					# elif aug == '3': #distortion
					elif aug in ['R', 'G', 'B', 'H', 'S', 'V']: 
						aug_op = getattr(self, "color_" + aug)
						param = self.delta
						param_min = -self.eps
					elif aug == '7':
						aug_op = getattr(self, "saturation")
						param = self.delta + 1
						param_min = -self.eps
					elif aug == '8':
						aug_op = getattr(self, "contrast")
						param = self.delta + 1
						param_min = -self.eps
					elif aug == '9':
						aug_op = getattr(self, "brightness")
						param = self.delta
						param_min = -self.eps
					elif aug == 'N':
						aug_op = None
						break
					else:
						print("augmentation not defined")

					x = aug_op(input, param)
					output = self.model(x)
					# loss = self.loss_fn(output, target)
					loss = self.mse(output, target)

				grad = tape.gradient(loss, [self.delta])[0]
				# tf.print(grad)
				self.delta.assign_add(self.adv_step * tf.keras.backend.sign(grad))
				self.delta.assign(tf.keras.backend.clip(self.delta, min_value=param_min, max_value=self.eps))
			
			if aug_op is not None:
				x = aug_op(input, param)
			else:
				x = input
			with tf.GradientTape() as tape:
				output = self.model(x)
				# tf.print(tf.shape(output))
				# loss = self.loss_fn(output, target)
				# loss = self.loss_fn(target, output)
				loss = self.mse(output, target)

			grads = tape.gradient(loss, self.model.trainable_weights)
			self.h_optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

			self.train_loss_tracker.update_state(loss)

		# return {
		# 	"mloss": self.train_loss_tracker.result()
		# }
		return self.train_loss_tracker.result()

	def test_step(self, batch_data):
		input, target = batch_data
		input = tf.cast(input, dtype=tf.float32)
		target = tf.cast(target, dtype=tf.float32)

		output = self.model(input)
		loss = self.mse(target, output)
		# ma = self.h_metrics(output, target)

		thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]
		total_acc = 0
		prediction_error = tf.math.abs(output-target)
		# tf.print(prediction_error, tf.shape(prediction_error))

		for thresh_hold in thresh_holds:
			acc = tf.where(prediction_error < thresh_hold, 1., 0.)
			acc = tf.math.reduce_mean(acc)
			total_acc += acc

		ma = total_acc / len(thresh_holds)
		# tf.print("ma: ", ma)

		# return {
		# 	"loss": loss,
		# 	"ma": ma
		# }
		return loss, ma




class AdvBN_Net(tf.keras.Model):
	def __init__(self, fClassifier, nClass, adv_step=0.2, n_repeats=3, eps=0.5, before_relu=False, nChannel=3):
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
		# print("**********************  FeatureX *****************************")
		# model.summary()

		return model

	def get_head(self):
		mainInput = tf.keras.Input(shape=self.feature_shape)
		# x = tf.nn.elu(mainInput)
		x = mainInput
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
		# print("**********************  Head *****************************")
		# model.summary()

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
		out_feature = tf.nn.elu(adv_feature)

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
	
