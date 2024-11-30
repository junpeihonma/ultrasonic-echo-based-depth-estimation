import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from . import networks_multitask,criterion
from torch.autograd import Variable

class EchoBasedModel(torch.nn.Module):
    def name(self):
        return 'EchoBasedModel'

    def __init__(self, nets, opt):
        super(EchoBasedModel, self).__init__()
        self.opt = opt
        self.net_audio = nets

    def forward(self, input, depth_or_spec, volatile=False):
        audio_input = input['audio']
        depth_gt = input['depth']

        if depth_or_spec == "depth":        
            pre_depth = self.net_audio(audio_input,"depth")
            output =  {
                    'pre_depth': pre_depth * self.opt.max_depth,
                    'depth_gt': depth_gt}
            return output
            
        elif depth_or_spec == "spec":
            pre_spec = self.net_audio(audio_input,"spec")
            output =  {'pre_spec': pre_spec * self.opt.max_spec}
            return output
        
