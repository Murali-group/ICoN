import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import GATConv

from utils.common import Device

from typing import Optional, Tuple
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor, set_diag


def weighted_softmax(
    src: Tensor,
    index: Tensor,
    edge_weights: Tensor,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    """
    MIT License

    Copyright (c) 2020 Duncan Forster

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.


    Extends the PyTorch Geometric `softmax` functionality to incorporate edge weights.

    See the PyTorch Geomtric `softmax` documentation for details on arguments.
    """

    if ptr is None:
        N = maybe_num_nodes(index, num_nodes)
        out = src - scatter(src, index, dim=0, dim_size=N, reduce="max")[index]
        out = edge_weights.unsqueeze(-1) * out.exp()  # multiply softmax by `edge_weights`
        out_sum = scatter(out, index, dim=0, dim_size=N, reduce="sum")[index]
        return out / (out_sum + 1e-16)
    else:
        raise NotImplementedError("Using `ptr` with `weighted_softmax` has not been implemented.")


class ComboWGATConv(GATConv):
    """Weighted version of the Graph Attention Network (`GATConv`).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_attn(
        self,
        x,
        edge_index,
        edge_weights: Optional[Tensor] = None,
        size=None,
    ):
        """Adapted from the PyTorch Geometric `GATConv` `forward` method. See PyTorch Geometric
        for documentation.
        """
        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in `GATConv`."
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, "Static graphs not supported in `GATConv`."
            x_l = self.lin_l(x_l).view(-1, H, C) #Nure: Dividing features into H heads.
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)
        assert x_l is not None
        assert alpha_l is not None

        # print('Before adding self loops number of edges: ',edge_index.size(1))
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, edge_weights = remove_self_loops(edge_index, edge_weights)
                edge_index, edge_weights = add_self_loops(
                    edge_index, edge_weights, num_nodes=num_nodes
                )
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
            # print('After adding self loops number of edges: ',edge_index.size(1))

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        self.edge_weights = edge_weights

        ##**********************  now compute attention ******************
        size = self.__check_input__(edge_index, size)
        coll_dict_args = {'x_j', 'alpha_j', 'alpha_i', 'index', 'ptr', 'size_i' }
        coll_dict = self.__collect__(coll_dict_args, edge_index,
                    size, {'x':(x_l, x_r), 'alpha':(alpha_l, alpha_r)})
        req_keys_to_process_alpha = {'alpha_j', 'alpha_i','index','ptr','size_i'}
        prm = {}
        for key in req_keys_to_process_alpha:
            prm[key] = coll_dict[key]

        alpha = self.process_alpha(**prm)
        self._alpha = None
        return (edge_index, alpha)

    def forward(self,
                x,
                edge_index,
                alpha, #if co_attn_weights=alpha cannot be passed like this then pass it through kwargs.
                edge_weights: Optional[Tensor] = None,
                size=None,
                return_attention_weights=None):

        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in `GATConv`."
            x_l = x_r = self.lin_l(x).view(-1, H, C)

        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, "Static graphs not supported in `GATConv`."
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
        assert x_l is not None

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha, size=size)
        self._alpha = None
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def process_alpha(
        self,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = weighted_softmax(alpha, index, self.edge_weights, ptr, size_i)
        self._alpha = alpha
        return alpha

    def message(
        self,
        x_j: Tensor,
        alpha: Tensor
    ) -> Tensor:
        self._alpha = alpha
        #TODO: should I add the following dropout?
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

class Interp(nn.Module):
    """
    MIT License

    Copyright (c) 2020 Duncan Forster

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Stochastic summation layer.

    Performs random node feature dropout and feature scaling.
    NOTE: This is not very nice, but it is differentiable. Future work
    should involve rewriting this.
    """

    def __init__(self, n_modalities: int):
        super(Interp, self).__init__()

        self.temperature = (
            1.0  # can modify this to change the relative magnitudes of network scales
        )

        self.net_scales = nn.Parameter(
            (torch.FloatTensor([1.0 for _ in range(n_modalities)]) / n_modalities).reshape((1, -1))
        )

    def forward(
        self, mask: Tensor, idxs: Tensor, evaluate: bool = False, device=None
    ) -> Tuple[Tensor, Tensor]:

        net_scales = F.softmax(self.net_scales / self.temperature, dim=-1)
        net_scales = net_scales[:, idxs]

        if evaluate:
            random_mask = torch.IntTensor(mask.shape).random_(1, 2).float()
        else:
            random_mask = torch.IntTensor(mask.shape).random_(0, 2).float()

        if device is None:
            device = Device()
        random_mask = random_mask.to(device)

        mask_sum = 1 / (1 + torch.sum(random_mask, dim=-1)) ** 20
        random_mask += mask_sum.reshape((-1, 1))
        random_mask += 1 / (torch.sum(mask, dim=-1) ** 20).reshape((-1, 1))
        random_mask = random_mask.int().float()
        random_mask = random_mask / (random_mask + 1e-10)

        mask = mask * random_mask
        mask = F.softmax(mask + ((1 - mask) * -1e10), dim=-1)

        return net_scales, mask

class NetWeights(nn.Module):
    def __init__(self, n_modalities: int, net_idx:int, init_mode:str='max_own', con=True):
        super(NetWeights, self).__init__()
        self.con=con
        if con:
            if init_mode=='max_own':
                self.net_scales = nn.Parameter((torch.FloatTensor
                                ([1.0 if i==net_idx else 0 for i in range(n_modalities)])).reshape((1, -1)))
            elif init_mode=='uniform':
                self.net_scales = nn.Parameter((torch.FloatTensor([1.0 for _ in range(n_modalities)])
                               / n_modalities).reshape((1, -1)))
        else:
            self.net_scales =torch.tensor([1.0 if i==net_idx else 0 for i in range(n_modalities)]).reshape((1, -1)).to('cuda')

    def forward(self) -> Tensor:
        if self.con:
            net_scales = F.softmax(self.net_scales, dim=-1)
            assert (not np.any(np.array(net_scales.detach().to('cpu'))< 0)),\
                print('Negative value after softmax', net_scales)
        else:
            net_scales = self.net_scales

        return net_scales