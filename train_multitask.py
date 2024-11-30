import os
import torch
from options.train_options import TrainOptions
from models.models_multitask import ModelBuilder
from models.echo_based_model_multitask import EchoBasedModel
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from util.util import TextWrite, compute_errors
import numpy as np
from models import criterion 
import torchaudio.transforms as T
import random
import copy

def create_optimizer(nets, opt):
	param_groups = [{'params': nets.parameters(), 'lr': opt.lr_audio}]
	if opt.optimizer == 'sgd':
		return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
	elif opt.optimizer == 'adam':
		return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.94):
	""" decrease learning rate 6% every opt.learning_rate_decrease_itr epochs """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

def evaluate(model, loss_criterion, dataset_val, opt):
	losses = []
	errors = []
	with torch.no_grad():
		for i, val_data in enumerate(dataset_val):
			output = model.forward(val_data, "depth")
			pre_depth = output['pre_depth']
			depth_gt = output['depth_gt']
			
			loss = loss_criterion(pre_depth[depth_gt!=0], depth_gt[depth_gt!=0])
			losses.append(loss.item())
			for idx in range(pre_depth.shape[0]):
				errors.append(compute_errors(depth_gt[idx].cpu().numpy(), 
								pre_depth[idx].cpu().numpy()))
	
	mean_loss = sum(losses)/len(losses)
	mean_errors = np.array(errors).mean(0)	
	print('Loss: {:.3f}, RMSE: {:.3f}'.format(mean_loss, mean_errors[1])) 
	val_errors = {}
	val_errors['ABS_REL'], val_errors['RMSE'] = mean_errors[0], mean_errors[1]
	val_errors['DELTA1'] = mean_errors[2] 
	val_errors['DELTA2'] = mean_errors[3]
	val_errors['DELTA3'] = mean_errors[4]
	return mean_loss, val_errors 

def Mixup(x1, x2,alpha=0.5):
	""" Linear mixing of echoes """
    mixed_x = alpha * x1 + (1 - alpha) * x2
    return mixed_x

if __name__ == '__main__':
	# loss criterion
	loss_criterion_depth = criterion.LogDepthLoss()
	loss_criterion_spec = criterion.LogSpecLoss()
	opt = TrainOptions().parse()
	opt.device = torch.device("cuda")

	# Log the results
	loss_list = ['step', 'loss']
	err_list = ['step', 'RMSE', 'ABS_REL', 'DELTA1', 'DELTA2', 'DELTA3']

	train_loss_file = TextWrite(os.path.join(opt.checkpoints_dir, 'train_loss.csv'))
	train_loss_file.add_line_csv(loss_list)
	train_loss_file.write_line()

	train_loss_depth_file = TextWrite(os.path.join(opt.checkpoints_dir, 'train_loss_depth.csv'))
	train_loss_depth_file.add_line_csv(loss_list)
	train_loss_depth_file.write_line()

	train_loss_spec_file = TextWrite(os.path.join(opt.checkpoints_dir, 'train_loss_spec.csv'))
	train_loss_spec_file.add_line_csv(loss_list)
	train_loss_spec_file.write_line()

	val_loss_file = TextWrite(os.path.join(opt.checkpoints_dir, 'val_loss.csv'))
	val_loss_file.add_line_csv(loss_list)
	val_loss_file.write_line()

	val_error_file = TextWrite(os.path.join(opt.checkpoints_dir, 'val_error.csv'))
	val_error_file.add_line_csv(err_list)
	val_error_file.write_line()

	# dataLoader
	dataloader = CustomDatasetDataLoader()
	dataloader.initialize(opt)
	dataset = dataloader.load_data()
	dataset_size = len(dataloader)
	
	print('#train clips = %d' % dataset_size)
	audio_shape = [4, 257, 331]

	for data in dataset: 
		audio_shape = data["audio"].shape[1:]
		break

	# network builders
	builder = ModelBuilder()
	weights=''
	nets = builder.build_audio(audio_shape,weights)

	# construct echo based model
	model = EchoBasedModel(nets, opt)
	model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
	model.to(opt.device)

	dataloader = CustomDatasetDataLoader()
	dataloader.initialize(opt)
	dataset = dataloader.load_data()
	dataset_size = len(dataloader)

	if opt.validation_on:
		opt.mode = 'test'  
		dataloader_val = CustomDatasetDataLoader()
		dataloader_val.initialize(opt)
		dataset_val = dataloader_val.load_data()
		dataset_size_val = len(dataloader_val)
		opt.mode = 'train'

	optimizer = create_optimizer(nets, opt)

	# initialization
	total_steps = 0
	batch_loss = []
	batch_loss_depth = []
	batch_loss_spec = []
	best_rmse = float("inf")
	best_loss = float("inf")

	print("lamda:0.6=>0, mu:0.4=>0")

	for epoch in range(1, opt.niter+1): 
		torch.cuda.synchronize()
		batch_loss = []   

		for i, data in enumerate(dataset):

			total_steps += opt.batchSize

			""" depth prediction """
			model.zero_grad()
			output_depth = model.forward(data, "depth")
			pre_depth = output_depth['pre_depth']
			depth_gt = output_depth['depth_gt']

			loss_depth = loss_criterion_depth(pre_depth[depth_gt!=0], depth_gt[depth_gt!=0]) 

			""" spectrogram prediction """
			model.zero_grad()
			output_spec = model.forward(data, "spec")
			pre_spec = output_spec['pre_spec']
			spec_gt = data["audio_sub"].to(opt.device)

			loss_spec = loss_criterion_spec(pre_spec[spec_gt!=0], spec_gt[spec_gt!=0])

			""" mixup """
			alpha = random.random()
			data_mix = copy.deepcopy(data)
			model_mix = copy.deepcopy(model)

			# Mixup by channel
			for j in range(audio_shape[0]):
				data_mix["audio"][:,j,:,:] = Mixup(data["audio"][:,j,:,:], data["audio_sub"][:,j,:,:],alpha)

			model_mix.zero_grad()
			output = model_mix.forward(data_mix, "depth")
			pre_depth_mix = output['pre_depth']
			depth_gt = output['depth_gt']

			loss_mix = loss_criterion_depth(pre_depth_mix[depth_gt!=0], depth_gt[depth_gt!=0])
		

			""" calculation total loss """
			
			# Introduce hyperparameters and schedule them
			# lambda : 0.5 => 0
			# mu     : 0.5 => 0
			lamb =  0.5 - (epoch-1)/60
			if lamb < 0:
				lamb = 0
			mu =  0.5 - (epoch-1)/60
			if mu < 0:
				mu = 0

			loss = (1-lamb-mu) * loss_depth + lamb * loss_spec * 3 + mu * loss_mix
			batch_loss.append(loss.item())
			batch_loss_depth.append(loss_criterion_depth(pre_depth[depth_gt!=0], depth_gt[depth_gt!=0]).item())
			batch_loss_spec.append(loss_criterion_spec(pre_spec[spec_gt!=0], spec_gt[spec_gt!=0]).item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# total loss
			if(total_steps // opt.batchSize % int(dataset_size/opt.batchSize) == 0):
				avg_loss = sum(batch_loss) / len(batch_loss)
				batch_loss = []
				train_loss_file.add_line_csv([total_steps // opt.batchSize, avg_loss])
				train_loss_file.write_line()

			# depth loss
			if(total_steps // opt.batchSize % int(dataset_size/opt.batchSize) == 0):
				avg_loss = sum(batch_loss_depth) / len(batch_loss_depth)
				batch_loss_depth = []
				train_loss_depth_file.add_line_csv([total_steps // opt.batchSize, avg_loss])
				train_loss_depth_file.write_line()

			# spec loss
			if(total_steps // opt.batchSize % int(dataset_size/opt.batchSize) == 0):
				avg_loss = sum(batch_loss_spec) / len(batch_loss_spec)
				batch_loss_spec = []
				train_loss_spec_file.add_line_csv([total_steps // opt.batchSize, avg_loss])
				train_loss_spec_file.write_line()
			
			if(total_steps // opt.batchSize % int(dataset_size/opt.batchSize) == 0 and opt.validation_on):
				model.eval()
				opt.mode = 'test'
				print('epoch %d  : ' % (epoch), end="")
				val_loss, val_err = evaluate(model, loss_criterion_depth, dataset_val, opt)
				model.train()
				opt.mode = 'train'

				# save the model that achieves the smallest validation error
				if val_err['RMSE'] < best_rmse:
					best_rmse = val_err['RMSE']
					torch.save(nets.state_dict(), os.path.join(opt.checkpoints_dir, 'trained_best_model.pth'))

				# Logging the values for the val set
				val_loss_file.add_line_csv([total_steps // opt.batchSize, val_loss])
				val_loss_file.write_line()
			
				err_list = [total_steps // opt.batchSize, \
					val_err['RMSE'], val_err['ABS_REL'], \
					val_err['DELTA1'], val_err['DELTA2'], val_err['DELTA3']]
				val_error_file.add_line_csv(err_list)
				val_error_file.write_line()

		if epoch % opt.epoch_save_freq == 0:
			torch.save(nets.state_dict(), os.path.join(opt.checkpoints_dir, 'trained_model_epoch_'+str(epoch)+'.pth'))

		#decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
		if(opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0):
			decrease_learning_rate(optimizer, opt.decay_factor)

