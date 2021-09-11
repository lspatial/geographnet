
import torch
import torch.nn.functional as F
import os
os.path.abspath(__file__)
import sys
sys.path.append(".")
from .geogcon  import GEOGCon

class GeoGraphPNet(torch.nn.Module):
    r""" Geographic Graph hHybrid Network for PM2.5 and PM10
       Args:
           in_channels (int or tuple): Size of each input sample. A tuple
               corresponds to the sizes of source and target dimensionalities.
           hidden_channels (int): Number of nodes for each graph convolution layer.
           out_channels (int): Size of each output sample.
           num_layers (int): Number of graph layers.
           autolayersNo (int, optional): Number of hidden layers in full deep network.
           weightedmean (bool, optional): If set to :obj:`True`, the weights will be used in graph convolution operations.
               (default: :obj:`True`)
           gcnout (int, optional): The number of the graph convolutions.
           nattlayer (int, optional): number of attention layers. (default: :obj:`None`).
    """
    def __init__(self, in_channels, hidden_channels, out_channels=2, num_layers=None, autolayersNo=None,weightedmean=True,gcnout=1,
                 nattlayer=None):
        super(GeoGraphPNet, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        if isinstance(hidden_channels,int):
            nh=hidden_channels
            hidden_channels=[nh for i in range(num_layers) ]
        if num_layers is not None and num_layers==1:
            self.convs.append(GEOGCon(in_channels, gcnout))
        else:
            self.convs.append(GEOGCon(in_channels, hidden_channels[0]))
            for i in range(1,num_layers - 1):
                self.convs.append(GEOGCon(hidden_channels[i-1], hidden_channels[i]))
            self.convs.append(GEOGCon(hidden_channels[num_layers-2], gcnout))
        self.autolayers = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        self.weightedmean=weightedmean
        self.atts = torch.nn.ModuleList()
        self.attsbn = torch.nn.ModuleList()
        self.nattlayer=nattlayer
        if autolayersNo is not None:
            if nattlayer is not None:
                for i in range(nattlayer):
                    self.atts.append(torch.nn.Linear(in_channels + gcnout, in_channels + gcnout))
                    self.attsbn.append(torch.nn.BatchNorm1d(in_channels + gcnout))
            self.autolayers.append(torch.nn.Linear(in_channels + gcnout, autolayersNo[0]))
            self.bn.append(torch.nn.BatchNorm1d(autolayersNo[0]))
            for i in range(1, len(autolayersNo)):
                self.autolayers.append(torch.nn.Linear(autolayersNo[i - 1], autolayersNo[i]))
                self.bn.append(torch.nn.BatchNorm1d(autolayersNo[i]))
            for i in range(len(autolayersNo) - 2, -1, -1):
                self.autolayers.append(torch.nn.Linear(autolayersNo[i + 1], autolayersNo[i]))
                self.bn.append(torch.nn.BatchNorm1d(autolayersNo[i]))
            self.lastLayer2 = torch.nn.Linear(autolayersNo[0], in_channels + gcnout)
            self.bn.append(torch.nn.BatchNorm1d(in_channels + gcnout))
            self.lastLayer = torch.nn.Linear(in_channels + gcnout, out_channels)
        self.autolayersNo = autolayersNo

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs, xnode):
        res = []
        for i, (edge_index, e_id, e_weight, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            if self.weightedmean:
                x = self.convs[i]((x, x_target), edge_index,e_id,e_weight)
            else:
                x = self.convs[i]((x, x_target), edge_index, e_id, None )
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        gcnx=x
        if len(self.autolayers) > 0:
            xin = torch.cat((xnode, x), 1)
            #if self.nattlayer is not None:
            for i in range(len(self.atts)):
                    prob=self.atts[i](xin)
                    prob = F.softmax(prob,dim=1)
                    xin = torch.mul(xin,prob)+xin
                    xin = self.attsbn[i](xin)
            res.append(xin)
            x = F.relu(self.autolayers[0](xin))
            x = self.bn[0](x)
            for i in range(1, len(self.autolayers)):
                if i <= len(self.autolayersNo) - 1:
                    res.append(x)
                x = F.relu(self.autolayers[i](x))
                x = self.bn[i](x)
                if i >= len(self.autolayersNo):
                    x = x + res.pop()
            x = self.lastLayer2(x)
            x = self.bn[i + 1](F.relu(x))
            x = x + res.pop()
            x = self.lastLayer(x)
            pm25 = x[:,0]
            pm10 =x[:, 1]
            pdif=F.relu(pm25-pm10)
            x=torch.cat((pm25[:,None], pm10[:,None],pdif[:,None]), 1)
        return x