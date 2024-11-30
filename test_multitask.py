
import os 
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from options.test_options import TestOptions
import torchvision.transforms as transforms
from models.models_multitask import ModelBuilder
from models.echo_based_model_multitask import EchoBasedModel
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from util.util import compute_errors
from models import criterion 


def min_max_255(x, axis=None):
    """ normalized between 0 and 255 """
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    if min == max:
        return np.full((1, 257, 331), 0)
    result = (x-min)/(max-min)

    return result*255


if __name__ == '__main__':

    loss_criterion = criterion.LogDepthLoss()
    opt = TestOptions().parse()
    opt.device = torch.device("cuda")

    opt.mode = 'test'
    dataloader_val = CustomDatasetDataLoader()
    dataloader_val.initialize(opt)
    dataset_val = dataloader_val.load_data()
    dataset_size_val = len(dataloader_val)
    print('#validation clips = %d' % dataset_size_val)

    audio_shape = [4, 257, 331]

    for data in dataset_val: 
        audio_shape = data["audio"].shape[1:]

    builder = ModelBuilder()

    # Load the training model
    if opt.test_model_type == 'best':
        # Evaluated by the model when the training error was the smallest
        checkpoints_dir = 'trained_models/' + self.opt.dataset + '/multitask/trained_best_model'
    else:
        # Evaluated by the model at the end of the learning
        checkpoints_dir = 'trained_model_epoch_300'

    net_audiodepth = builder.build_audio(audio_shape,weights=(checkpoints_dir+ '.pth'))   
    
    # construct echo based model
    model = EchoBasedModel(net_audiodepth, opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model.to(opt.device)
    model.eval()

    losses, errs = [], []
    cnt = 0

    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):

            output = model.forward(val_data, "depth")
            depth_predicted = output['pre_depth']
            depth_gt = output['depth_gt']

            # calculate loss
            loss = loss_criterion(depth_predicted[depth_gt!=0], depth_gt[depth_gt!=0])
            losses.append(loss.item())

            # imaging estimated depth map
            for idx in range(depth_gt.shape[0]):
                errs.append(compute_errors(depth_gt[idx].cpu().numpy(), 
                                depth_predicted[idx].cpu().numpy()))

                pre_depth = (depth_predicted[idx].cpu().numpy())
                true_depth = (depth_gt[idx].cpu().numpy())
               
                pre_depth = min_max_255(pre_depth).astype(np.uint8)
                true_depth = min_max_255(true_depth).astype(np.uint8)

                pre_depth = np.reshape(pre_depth, (128,128))
                true_depth = np.reshape(true_depth, (128,128))

                pre_depth = cv2.applyColorMap(pre_depth, cv2.COLORMAP_JET)
                true_depth = cv2.applyColorMap(true_depth, cv2.COLORMAP_JET)

                dst_pre_depth = Image.fromarray(pre_depth)
                dst_true_depth = Image.fromarray(true_depth)

                # Specify where to save the estimated depth map
                dst_pre_depth.save("imaged_depth_map/estimated/scene{}.jpg".format(cnt))
                dst_true_depth.save("imaged_depth_map/ground_truth/scene{}.jpg".format(cnt))

                cnt=cnt+1
             
                 
    mean_loss = sum(losses)/len(losses)
    mean_errs = np.array(errs).mean(0)

    print('Loss: {:.3f}, RMSE: {:.3f}'.format(mean_loss, mean_errs[1])) 

    errors = {}
    errors['ABS_REL'], errors['RMSE'], errors['LOG10'] = mean_errs[0], mean_errs[1], mean_errs[5]
    errors['DELTA1'], errors['DELTA2'], errors['DELTA3'] = mean_errs[2], mean_errs[3], mean_errs[4]
    errors['MAE'] = mean_errs[6]

    print('ABS_REL:{:.3f}, LOG10:{:.3f}, MAE:{:.3f}'.format(errors['ABS_REL'], errors['LOG10'], errors['MAE']))
    print('DELTA1:{:.3f}, DELTA2:{:.3f}, DELTA3:{:.3f}'.format(errors['DELTA1'], errors['DELTA2'], errors['DELTA3']))
    print('==='*25)
