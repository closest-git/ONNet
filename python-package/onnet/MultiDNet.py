from .D2NNet import *
import math
import random

class MultiDNet(D2NNet):
    def __init__(self, IMG_size,nCls,nInterDifrac,freq_list,config):
        super(MultiDNet, self).__init__(IMG_size,nCls,nInterDifrac,config)
        self.freq_list = freq_list
        del self.DD;     self.DD = None
        self.freq_nets=nn.ModuleList([
            nn.ModuleList([
                DiffractiveLayer(self.M, self.N, self.config, HZ=freq) for i in range(self.nDifrac)
            ]) for freq in freq_list
        ])

    def __repr__(self):
        main_str = super(MultiDNet, self).__repr__()
        main_str += f"\nfreq_list={self.freq_list}_"
        return main_str

    def forward(self, x0):
        nSamp = x0.shape[0]
        x_sum = 0
        for fNet in self.freq_nets:
            x = x0.double()
            for layD in fNet:
                x = layD(x)
            #x_sum = torch.max(x_sum,self.z_modulus(x).cuda()).values()
            x_sum += self.z_modulus(x).cuda()
        x = x_sum

        output = self.do_classify(x)
        '''
        #x = self.z_modulus(x).cuda()
                if self.config.isFC:
                    x = torch.flatten(x, 1)
                    x = self.fc1(x)
                else:
                    x = self.last_chunk(x)
        
                if self.config.chunk == "binary":
                    output = x
                else:
                    output = x
                    # output = F.log_softmax(x, dim=1)
        '''
        return output

        return output
