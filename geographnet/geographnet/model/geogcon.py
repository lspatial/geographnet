import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from typing import Tuple, Optional, Union

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]

class GEOGCon(MessagePassing):
    r"""The graph convolution operator from the `"Geographic Graph Hybrid Network for Robust Inversion of Particular Matters"
        to be published in Remote Sensing`_ paper
    GEOGCon: Graph convolutional operator weighted by reciprocal spatial or spatiotemporal distance

    .. math::
        \mathbf{x}^{\prime}_i =  \mathbf{W}^i_l \cdot \mathrm{wmean}_{j \in
               \mathcal{N(i)}} \mathbf{x}_j + \mathbf{W}^i_r \mathbf{x}_i

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(GEOGCon, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,e_id:Tensor, edge_weight: Tensor=None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        #Weighted sum
        if  edge_weight is not None:
            nlabs=edge_index[1]
            unique_nodes, node_count = nlabs.unique(dim=0, return_counts=True)
            sum1= torch.zeros_like(unique_nodes, dtype=torch.float).scatter_add_(0, nlabs, torch.ones_like(edge_weight,
                                                                                                             dtype=torch.float))
            res = torch.zeros_like(unique_nodes, dtype=torch.float).scatter_add_(0, nlabs, edge_weight)
            summ = res[nlabs]
            normedw = edge_weight/summ
            normedw = normedw*sum1[nlabs]
            out = self.propagate(edge_index, x=x, size=size,normedw=normedw)
        else:
            out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out

    def message(self, x_j: Tensor,normedw:Tensor=None) -> Tensor:
        if normedw is not None:
            normedw = normedw.view(normedw.size(0), 1).expand(-1, x_j.size(1))
            out=x_j*normedw
            return out #x_j
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)