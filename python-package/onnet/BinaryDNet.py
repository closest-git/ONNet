from .D2NNet import *
import math

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

    def __init__(self, nCls):
        super(BinaryDNet, self).__init__(nCls)
        self.nGate = (int)(math.ceil(math.log2(self.nClass)))
        self.gates = nn.ModuleList([
            nn.ModuleList([DiffractiveLayer(self.M,self.N),PoolForCls(2,pooling="mean")]) for i in range(self.nGate)
        ])

        self.loss = BinaryDNet.binary_loss

    def forward(self, x):
        x = x.double()
        for layD in self.DD:
            x = layD(x)

        nSamp = x.shape[0]
        output = []
        for [diffrac,gate] in self.gates:
            x1 = diffrac(x)
            x1 = self.z_modulus(x1).cuda()
            x1 = gate(x1)
            x2 = F.log_softmax(x1, dim=1)
            output.append(x2)

        return output

