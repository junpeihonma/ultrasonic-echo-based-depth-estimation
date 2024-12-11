from .base_options import BaseOptions

class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of displaying average loss')
		self.parser.add_argument('--niter', type=int, default=200, help='# of epochs to train')
		self.parser.add_argument('--learning_rate_decrease_itr', type=int, default=80, help='how often is the learning rate decreased by six percent')
		self.parser.add_argument('--decay_factor', type=float, default=0.94, help='learning rate decay factor')
		self.parser.add_argument('--validation_on', action='store_true', default="True",help='whether to test on validation set during training')
		self.parser.add_argument('--epoch_save_freq', type=int, default=100, help='frequency of saving intermediate models')

		#optimizer arguments
		self.parser.add_argument('--lr_audio', type=float, default=0.0001, help='learning rate for audio')
		self.parser.add_argument('--optimizer', default='adam', type=str, help='adam or sgd for optimization')
		self.parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
		self.parser.add_argument('--weight_decay', default=0.0005, type=float, help='weights regularizer')

		self.mode = "train"