from .D2NNet import *
import math
import random

class GatePipe(torch.nn.Module):
    def __init__(self,M,N, nHidden,config,pooling="max"):
        super(GatePipe, self).__init__()
        self.config = config
        self.M=M
        self.N=N
        self.nHidden = nHidden
        self.pooling = pooling
        self.layers = nn.ModuleList([DiffractiveLayer(self.M, self.N, self.config, HZ=0.3e12) for j in range(self.nHidden)])
        if True:
            chunk_dim = -1 if random.choice([True, False]) else -2
            self.pool = ChunkPool(2, self.config,pooling=self.pooling,chunk_dim=chunk_dim)
        else:
            self.pt1 = (random.randint(0, self.M-1),random.randint(0,self.N-1))
            self.pt2 = (random.randint(0, self.M - 1), random.randint(0, self.N - 1))

    def __repr__(self):
        main_str = super(GatePipe, self).__repr__()
        main_str = f"GatePipe_[{len(self.layers)}]_pool[{self.pooling}]"
        return main_str

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        x1 = Z.modulus(x).cuda()
        #x1 = Z.phase(x).cuda()
        if True:
            x1 = self.pool(x1)
        else:
            x_pt1 = x1[:, 0, self.pt1[0], self.pt1[1]]
            x_pt2 = x1[:, 0, self.pt2[0], self.pt2[1]]
            x1 = torch.stack([x_pt1,x_pt2], 1)
        x2 = F.log_softmax(x1, dim=1)
        return x2

class BinaryDNet(D2NNet):
    @staticmethod
    def binary_loss(output, target, reduction='mean'):
        nGate = len(output)
        nSamp = target.shape[0]
        loss =0
        for i in range(nGate):
            target_i = target%2
            # loss = F.binary_cross_entropy(output, target, reduction=reduction)
            loss_i = F.cross_entropy(output[i], target_i, reduction=reduction)
            loss += loss_i
            target =(target-target_i)/2

        # loss = F.nll_loss(output, target, reduction=reduction)
        return loss

    def predict(self,output):
        nGate = len(output)
        pred = 0
        for i in range(nGate):
            pred_i = output[nGate-1-i].max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred = pred*2+pred_i
        #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        return pred

    def __init__(self, IMG_size,nCls,nInterDifrac,nOutDifac,config):
        super(BinaryDNet, self).__init__(IMG_size,nCls,nInterDifrac,config)
        self.nGate = (int)(math.ceil(math.log2(self.nClass)))
        self.nOutDifac = nOutDifac
        self.gates = nn.ModuleList( [GatePipe(self.M,self.N,nOutDifac,config,pooling="mean") for i in range(self.nGate)]  )
        self.config = config
        self.loss = BinaryDNet.binary_loss

    def __repr__(self):
        main_str = super(BinaryDNet, self).__repr__()
        main_str += f"_nGate={self.nGate}_Difrac=[{self.nDifrac},{self.nOutDifac}]"
        return main_str

    def legend(self):
        title = f"BinaryDNet"
        return title

    def forward(self, x):
        x = x.double()
        for layD in self.DD:
            x = layD(x)

        nSamp = x.shape[0]
        output = []
        if True:
            for gate in self.gates:
                x1 = gate(x)
                output.append(x1)
        else:
            for [diffrac,gate] in self.gates:
                x1 = diffrac(x)
                x1 = self.z_modulus(x1).cuda()
                x1 = gate(x1)
                x2 = F.log_softmax(x1, dim=1)
                output.append(x2)

        return output

