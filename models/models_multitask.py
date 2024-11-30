import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from .networks_multitask import SimpleAudioNet,  weights_init  

class ModelBuilder():
    def build_audio(self, audio_shape=[4,257,552], weights=''):
       
        # network
        net = SimpleAudioNet(8, audio_shape=audio_shape, audio_feature_length=512)
        
        net.apply(weights_init)
        
        if len(weights) > 0:
            print('Loading weights.')
            net.load_state_dict(torch.load(weights))
        return net

    

   
