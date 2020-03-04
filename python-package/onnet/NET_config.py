
import torch

'''
    parser.add_argument is better than NET_config
'''
class NET_config:
    def __init__(self,net_type, data_set, IMG_size, lr_base, batch_size,nClass,nLayer=-1):
        #seed_everything(self.seed)
        self.net_type = net_type
        self.data_set = data_set
        self.IMG_size = IMG_size
        self.lr_base = lr_base  # "random"  "zero"
        self.batch_size = batch_size
        self.nClass = nClass
        self.nLayer = nLayer
