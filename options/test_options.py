from .base_options import BaseOptions

class TestOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)

		#model arguments
		self.mode = "test"
		self.isTrain = False
		self.enable_data_augmentation = False
		self.enable_cropping = True
