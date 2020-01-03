from .D2NNet import *
import math
import random

def DNet_instance(net_type,dataset,IMG_size,lr_base,batch_size,nClass,nLayer):
    if net_type == "BiDNet":
        lr_base = 0.01
    if dataset == "emnist":
        lr_base = 0.01

    config_base = DNET_config(batch=batch_size, lr_base=lr_base)
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
        # model = D2NNet(IMG_size, nClass,nLayer, DNET_config(chunk="logit"))
        #model = BinaryDNet(IMG_size,nClass,nLayer,1, config_base)
    model.double()

    return env_title, model



