'''
@Author: Yingshi Chen

@Date: 2020-01-16 15:08:16
@
# Description: 
'''
from .D2NNet import *
from .RGBO_CNN import *
from .OpticalFormer import *
import math
from copy import copy, deepcopy

def dump_model_params(model):
    nzParams = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            nzParams += param.nelement()
            print(f"\t{name}={param.nelement()}")
    print(f"========All parameters={nzParams}")
    return nzParams

def Net_dump(net):
    nzParams=dump_model_params(net)

#def DNet_instance(net_type,dataset,IMG_size,lr_base,batch_size,nClass,nLayer):     需要重写，只有一个config
def DNet_instance(config):
    net_type, dataset, IMG_size, lr_base, batch_size, nClass, nLayer = \
        config.net_type,config.data_set, config.IMG_size, config.lr_base, config.batch_size, config.nClass, config.nLayer
    if net_type == "BiDNet":
        lr_base = 0.01
    if dataset == "emnist":
        lr_base = 0.01

    config_base = DNET_config(batch=batch_size, lr_base=lr_base)
    if hasattr(config,'feat_extractor'):
        config_base.feat_extractor = config.feat_extractor
    env_title = f"{net_type}_{dataset}_{IMG_size}_{lr_base}_{config_base.env_title()}"
    if net_type == "MF_DNet":
        freq_list = [0.3e12, 0.35e12, 0.4e12, 0.42e12]
        env_title = env_title + f"_C{len(freq_list)}"
    if net_type == "BiDNet":
        config_base = DNET_config(batch=batch_size, lr_base=lr_base, chunk="binary")

    if net_type == "cnn":
        model = Mnist_Net(config=config_base)
        return env_title, model

    if net_type == "DNet":
        model = D2NNet(IMG_size, nClass, nLayer, config_base)
    elif net_type == "WNet":
        config_base.wavelet={"nWave":3}
        model = D2NNet(IMG_size, nClass, nLayer, config_base)
    elif net_type == "MF_DNet":
        # model = MultiDNet(IMG_size, nClass, nLayer,[0.3e12,0.35e12,0.4e12,0.42e12,0.5e12,0.6e12], DNET_config())
        model = MultiDNet(IMG_size, nClass, nLayer, [0.3e12, 0.35e12, 0.4e12, 0.42e12], config_base)
    elif net_type == "MF_WNet":
        config_base.wavelet = {"nWave": 3}
        model = MultiDNet(IMG_size, nClass, nLayer, [0.3e12, 0.35e12, 0.4e12, 0.42e12], config_base)
    elif net_type == "BiDNet":
        model = D2NNet(IMG_size, nClass, nLayer, config_base)
    elif net_type == "OptFormer":
        pass

    #model.double()

    return env_title, model

def RGBO_CNN_instance(config):
    assert config.net_type == "RGBO_CNN"
    env_title = f"{config.net_type}_{config.dnet_type}_{config.data_set}_{config.IMG_size}_{config.lr_base}_"
    assert hasattr(config,'dnet_type')

    if config.dnet_type!="":
        d_conf = deepcopy(config)
        if config.dnet_type == "stack_input":
            d_conf.net_type = "DNet"
            #d_conf.nLayer = 1
            #d_conf.feat_extractor = "layers"
        else:
            d_conf.nLayer = 10
            d_conf.net_type = "WNet"
        _,DNet = DNet_instance(d_conf)
    else:
        DNet=None
    model = RGBO_CNN(config,DNet)

    return env_title, model



