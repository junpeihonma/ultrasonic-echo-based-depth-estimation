import argparse
import os
from util import util
import torch

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		""" Setting command line arguments """
		self.parser.add_argument('--dataset', default='TUS-Echo', type=str, help='TUS-Echo')
		self.parser.add_argument('--dataset_path', default='/TUS-Echo', type=str, help='TUS-Echo')
		self.parser.add_argument('--echo_fre', default='', type=str, help='audible or ultra or null') # Set to null when running multi-task learning.
		self.parser.add_argument('--test_model_type', default='last', type=str, help='last or best') # Choose whether to use the best model or the model at the end of learning at the time of testing.
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
		self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
		self.parser.add_argument('--audio_length', default=0.4, type=float, help='audio length, default 0.06s')
		self.parser.add_argument('--audio_normalize', type=bool, default=False, help='whether to normalize the audio')
		self.parser.add_argument('--image_transform', type=bool, default=True, help='whether to transform the image data')
		self.parser.add_argument('--image_resolution', default=128, type=int, help='the resolution of image for cropping')
		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		self.opt.mode = self.mode
		self.opt.isTrain = self.isTrain
		self.opt.enable_img_augmentation = self.enable_data_augmentation
		self.opt.enable_cropping = self.enable_cropping 
		self.opt.scenes = {}

		if self.opt.dataset == 'TUS-Echo':
			self.opt.audio_sampling_rate = 96000 # sampling frequency 
			self.opt.max_depth = 10000 # Maximum value of depth
			self.opt.max_spec = 6 # Maximum value of Spectrogram
			
		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			if str(k) == "dataset":
				print('%s: %s' % (str(k), str(v)))
			if str(k) == "dataset_path":
				print('%s: %s' % (str(k), str(v)))
			if str(k) == "echo_fre":
				print('%s: %s' % (str(k), str(v)))
			if str(k) == "batchSize":
				print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')

		# Where to save the training model
		if self.opt.echo_fre == 'audible':
			checkpoints_dir = 'trained_models/' + self.opt.dataset + '/1_20k'
		elif self.opt.echo_fre == 'ultra':
			checkpoints_dir = 'trained_models/' + self.opt.dataset + '/20k_48k'
		else:
			checkpoints_dir = 'trained_models/' + self.opt.dataset + '/multitask'
		
		self.opt.checkpoints_dir = checkpoints_dir

		util.mkdirs(checkpoints_dir)
		file_name = os.path.join(checkpoints_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')

		return self.opt
