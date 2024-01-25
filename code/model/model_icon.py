import copy
import time
import cProfile
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
import torch.nn.functional as F

from torch_sparse import SparseTensor
from utils.sampler import Adj
from utils.common import Device
from typing import Dict, List, Tuple, Optional
from .layers import WGATConv, CoAttnWGATConv, ComboWGATConv,  Interp, NetWeights
from .model_utils import *

class IconGAT(nn.Module):
    def __init__(self, in_size: int, gat_shapes: Dict[str, int], alpha: float = 0.1):
        """BIONIC network encoder module.

        Args:
            in_size (int): Number of nodes in input networks.
            gat_shapes (Dict[str, int]): Graph attention layer hyperparameters.
            alpha (float, optional): LeakyReLU negative slope. Defaults to 0.1.

        Returns:
            Tensor: 2D tensor of node features. Each row is a node, each column is a feature.
        """
        super(IconGAT, self).__init__()
        self.in_size = in_size
        self.dimension: int = gat_shapes["dimension"]
        self.n_heads: int = gat_shapes["n_heads"]
        self.alpha = alpha
        self.dropout = gat_shapes["dropout"]

        self.gat = WGATConv(
            (self.dimension * self.n_heads,) * 2,
            self.dimension,
            heads=self.n_heads,
            dropout=self.dropout,
            negative_slope=self.alpha,
            add_self_loops=True,
        )

    def forward(self, data_flow, layer_no, x= None, device=None):
        _, n_id, adjs = data_flow
        if device is None:
            device = Device()

        adj = adjs[layer_no].to(device)

        if layer_no == 0: #taking only the features for the current node under
            # consideration as at the first layer whole
            #feature matrix for all nodes has been passed.
            x = x[n_id]

        #Nure: return attention weight
        x, attn = self.gat((x, x[: adj.size[1]]), adj.edge_index,
                    size=adj.size, edge_weights=adj.weights, return_attention_weights=True)
        return attn


class IconCoGAT(nn.Module):
    def __init__(self, in_size: int,  net_idx:int, n_modalities:int, gat_shapes: Dict[str, int], alpha: float = 0.1,
                net_weights =None):
        """BIONIC network encoder module.
        Args:
            in_size (int): Number of nodes in input networks.
            n_modalities (int): Number of networks.
            gat_shapes (Dict[str, int]): Graph attention layer hyperparameters.
            alpha (float, optional): LeakyReLU negative slope. Defaults to 0.1.

        Returns:
            Tensor: 2D tensor of node features. Each row is a node, each column is a feature.
        """
        super(IconCoGAT, self).__init__()
        self.in_size = in_size
        self.dimension: int = gat_shapes["dimension"]
        self.n_heads: int = gat_shapes["n_heads"]
        self.dropout = gat_shapes["dropout"]

        # self.n_layers: int = gat_shapes["n_layers"]
        self.alpha = alpha
        # self.pre_gat = nn.Linear(self.in_size, self.dimension * self.n_heads)
        self.net_idx = net_idx
        self.gat = CoAttnWGATConv(
            (self.dimension * self.n_heads,) * 2,
            self.dimension,
            heads=self.n_heads,
            dropout=self.dropout,
            negative_slope=self.alpha,
            add_self_loops=False,
        )

        self.net_weights = net_weights
        # def set_net_scale(self, n_modalities, net_idx, init_mode):
        #     #Define a learnable weight vector co_W of size m where  m=number of input networks.
        #     self.net_weights = NetWeights(n_modalities, net_idx, init_mode)

    def forward(self, data_flow, attns, edge_indices_global, layer_no, x, device=None, mmode='icon'):
        net_weights = self.net_weights()#give normalized netweights
        cogat_edge_idx_global={}
        #TODO define w_cogat_attn as module dict
        w_cogat_attn={}

        if mmode=='bionic': #just take own attention into account
            cogat_attn = attns[self.net_idx][1]
            cogat_edge_idx_global[self.net_idx] = edge_indices_global[self.net_idx]
            # Multiply cogat_attn with co_W
            w_cogat_attn[self.net_idx] = cogat_attn #put weight = 1
        else: #take attention from all inout networks
            for net_idx in attns:
                cogat_attn = attns[net_idx][1]
                cogat_edge_idx_global[net_idx] = edge_indices_global[net_idx]
                #Multiply cogat_attn with co_W
                w_cogat_attn[net_idx] = net_weights[:, net_idx]*cogat_attn

        ##***************** WITHOUT MMODE*****************
        # aggregate attention value of the same edges across networks. Compute final attns across edges.
        # and use that for following part.
        # TODO: this aggregat_edge_weights() function is a problem. Fix it.
        agg_attn_mat = aggregat_edge_weights(w_cogat_attn, cogat_edge_idx_global,
                                             self.in_size, self.n_heads, device='cuda')

        _, n_id, adjs = data_flow
        if device is None:
            device = Device()

        adj = adjs[layer_no].to(device)
        local_edge_index = edge_idx_from_global_to_local(agg_attn_mat.indices(), n_id)

        if layer_no == 0: #taking only the features for the current node under consideration as at the first layer whole
            #feature matrix for all nodes has been passed.
            x = x[n_id]
        x = self.gat((x, x[: adj.size[1]]), local_edge_index.to(device),
                     alpha=agg_attn_mat.values().to(device), size=adj.size)
        return x


class IconCombo(nn.Module):
    # def __init__(self, in_size: int,  net_idx:int, n_modalities:int, gat_shapes: Dict[str, int], alpha: float = 0.1):
    def __init__(self, in_size: int, gat_shapes: Dict[str, int], alpha: float = 0.1,
            net_weights =None, dropout=None):
        """BIONIC network encoder module.
        Args:
            in_size (int): Number of nodes in input networks.
            n_modalities (int): Number of networks.
            gat_shapes (Dict[str, int]): Graph attention layer hyperparameters.
            alpha (float, optional): LeakyReLU negative slope. Defaults to 0.1.

        Returns:
            Tensor: 2D tensor of node features. Each row is a node, each column is a feature.
        """
        super(IconCombo, self).__init__()
        self.in_size = in_size
        self.dimension: int = gat_shapes["dimension"]
        self.n_heads: int = gat_shapes["n_heads"]
        if dropout==None:
            self.dropout = gat_shapes["dropout"]
        else:
            self.dropout = dropout

        # self.n_layers: int = gat_shapes["n_layers"]
        self.alpha = alpha
        # self.pre_gat = nn.Linear(self.in_size, self.dimension * self.n_heads)
        # self.net_idx = net_idx
        self.gat = ComboWGATConv(
            (self.dimension * self.n_heads,) * 2,
            self.dimension,
            heads=self.n_heads,
            dropout=self.dropout,
            negative_slope=self.alpha,
            add_self_loops=True,
        )
        #Define a learnable weight vector co_W of size m where  m=number of input networks.
        self.net_weights = net_weights

    def compute_icon_attention(self, data_flow, layer_no, x= None, device=None):
        _, n_id, adjs = data_flow
        if device is None:
            device = Device()

        adj = adjs[layer_no].to(device)

        if layer_no == 0:  # taking only the features for the current node under
            # consideration as at the first layer whole
            # feature matrix for all nodes has been passed.
            x = x[n_id]

        # Nure: return attention weight
        attn = self.gat.compute_attn((x, x[: adj.size[1]]), adj.edge_index,
                           size=adj.size, edge_weights=adj.weights)

        return attn

    def forward(self, data_flow, attns, edge_indices_global, layer_no, x, agg_edges=None, agg_alpha=None, device=None):
        if (agg_edges==None and agg_alpha==None): #Then I have to compute network specific agg_attn_mat here
            net_weights = self.net_weights()  # give normalized netweights
            cogat_edge_idx_global = {}
            w_cogat_attn = {}

            # take attention from all inout networks
            for net_idx in attns:
                cogat_attn = attns[net_idx][1]
                cogat_edge_idx_global[net_idx] = edge_indices_global[net_idx]
                # Multiply cogat_attn with co_W
                w_cogat_attn[net_idx] = net_weights[:, net_idx] * cogat_attn

            agg_attn_mat = aggregat_edge_weights(w_cogat_attn, cogat_edge_idx_global,
                                                 self.in_size, self.n_heads, device='cuda')

            agg_alpha = agg_attn_mat.values()
            agg_edges = agg_attn_mat.indices()

        _, n_id, adjs = data_flow
        if device is None:
            device = Device()

        adj = adjs[layer_no].to(device)
        local_edge_index = edge_idx_from_global_to_local(agg_edges, n_id)

        if layer_no == 0: #taking only the features for the current node under consideration as at the first layer whole
            #feature matrix for all nodes has been passed.
            x = x[n_id]
        x = self.gat((x, x[: adj.size[1]]), local_edge_index.to(device),
                     alpha=agg_alpha.to(device), size=adj.size)


        return x, agg_attn_mat


class Icon(nn.Module):
    def __init__(
        self,
        in_size: int,
        pre_gat_type: str,
        gat_type: str,
        gat_shapes: Dict[str, int],
        emb_size: int,
        residual: False,
        n_modalities: int,
        bionic_mask: bool=True,
        alpha: float = 0.1,
        svd_dim: int = 0,
        shared_encoder: bool = False,
        n_classes: Optional[List[int]] = None,
        feats=None,
        init_mode='max_own',
        agg = 'avg',
        scale='sep',
        con=True

    ):
        """The ICoN model.

        Args:
            in_size (int): Number of nodes in input networks.
            gat_shapes (Dict[str, int]): Graph attention layer hyperparameters.
            emb_size (int): Dimension of learned node features.
            n_modalities (int): Number of input networks.
            alpha (float, optional): LeakyReLU negative slope. Defaults to 0.1.
            svd_dim (int, optional): Dimension of input node feature SVD approximation.
                Defaults to 0. No longer required and is safely ignored.
            shared_encoder (bool, optional): Whether to use the same encoder (pre-GAT
                + GAT) for all networks.
            n_classes (list of int, optional): Number of classes per supervised
                standard, if supervised standards are provided.
        """

        super(Icon, self).__init__()

        self.in_size = in_size
        self.emb_size = emb_size
        self.residual = residual
        self.alpha = alpha
        self.n_modalities = n_modalities
        self.bionic_mask = bionic_mask
        self.svd_dim = svd_dim
        self.shared_encoder = shared_encoder
        self.n_classes = n_classes

        self.pre_gat_type = pre_gat_type
        self.gat_type = gat_type
        self.scale = scale
        self.gat_shapes = gat_shapes
        self.dimension: int = self.gat_shapes["dimension"]
        self.n_heads: int = self.gat_shapes["n_heads"]
        self.n_layers = self.gat_shapes["n_layers"]
        self.dropout = self.gat_shapes["dropout"]
        self.init_mode = init_mode
        self.agg = agg
        self.con=con

        if feats is None:
            self.feats = None
        else:
            self.feats = {}
            for i in list(range(self.n_modalities)):
                if isinstance(feats[i], Data):
                    self.feats[i]=feats[i].adj_t.to_dense()
                elif isinstance(feats[i], torch.Tensor):
                    self.feats[i] = feats[i]

        self.encoders = {i:[] for i in range(self.n_modalities)}
        self.co_encoders = {}
        if bool(self.svd_dim):
            warnings.warn(
                "SVD approximation is no longer required for large networks and will be ignored."
            )
        #pre GAT
        if self.pre_gat_type == 'single':
            self.pre_gat = nn.Linear(self.in_size, self.dimension * self.n_heads)
            self.add_module(f"Pre_GAT", self.pre_gat)

        elif self.pre_gat_type == 'sep':
            self.pre_gats = []
            for i in range(self.n_modalities):
                self.pre_gats.append(nn.Linear(self.in_size, self.dimension * self.n_heads))
                self.add_module(f"Pre_GAT{i}", self.pre_gats[i])


        self.set_encoder()

        if self.agg=='avg':
            self.integration_size = self.dimension * self.n_heads
        elif self.agg=='concat':
            self.integration_size = self.dimension * self.n_heads * self.n_modalities

        self.interp = Interp(self.n_modalities)

        # Embedding
        self.emb = nn.Linear(self.integration_size, self.emb_size)

        # Supervised classification head
        if self.n_classes:
            self.cls_heads = [
                nn.Sequential(
                    nn.Linear(self.emb_size, self.emb_size),  # improves optimization
                    nn.Linear(self.emb_size, n_classes_),
                )
                for n_classes_ in self.n_classes
            ]
            for h, cls_head in enumerate(self.cls_heads):
                self.add_module(f"Classification_Head_{h}", cls_head)
        else:
            self.cls_heads = None


    def set_encoder(self):
        # Initiate the net_scales
        net_weights = {}
        for k in range(self.n_layers):  # I will always have separate net_weights at each layer
            net_weights[k] = {}
            if self.scale == 'sep':  # depending on self.scale we may have same or separate net_weights at the same layer across multiple networks.
                for i in range(self.n_modalities):
                    net_weights[k][i] = NetWeights(self.n_modalities, i, self.init_mode, self.con)
            if self.scale == 'same':
                net_weights[k][0] = NetWeights(self.n_modalities, 0,
                                               'uniform', self.con)  # initiate with same weight for each network. irrespective of self.init_mode

        if self.gat_type == 'sep_sep':
            for i in range(self.n_modalities):
                # self.encoders[i] = []
                for k in range(self.n_layers):
                    self.encoders[i].append(IconGAT(self.in_size, self.gat_shapes, self.alpha))
                    self.add_module(f"Encoder_{i}_{k}", self.encoders[i][k])
            # Co-GATS
            for i in range(self.n_modalities):
                self.co_encoders[i] = []
                for k in range(self.n_layers):
                    self.co_encoders[i].append(IconCoGAT(self.in_size, i, self.n_modalities,self.gat_shapes, self.alpha,
                        net_weights[k][i] if i in net_weights[k] else net_weights[k][0] ))
                    self.add_module(f"Co_Encoder_{i}_{k}", self.co_encoders[i][k])

        elif self.gat_type=='sep_single':#if gat_type='sep_single' then the first part 'sep'
            # means for the same network, in the same layer
            # we have different learnable W i.e., two GATConv for GAT and CO-GAT.
            # The second part 'single' means for the same network we use the same learnable W
            # across all GAT layers.
            # GATs
            for i in range(self.n_modalities):
                # self.encoders[i] = []
                self.encoders[i].append(IconGAT(self.in_size, self.gat_shapes, self.alpha))
                self.add_module(f"Encoder_{i}_0",self.encoders[i][0] )
            #Co-GATS
            for i in range(self.n_modalities):
                self.co_encoders[i] = []
                self.co_encoders[i].append(IconCoGAT(self.in_size, i, self.n_modalities, self.gat_shapes, self.alpha,
                            net_weights[0][i] if i in net_weights[0] else net_weights[0][0]))
                self.add_module(f"Co_Encoder_{i}_0", self.co_encoders[i][0])

        elif self.gat_type == 'single_sep':
            for i in range(self.n_modalities):
                # self.encoders[i] = []
                for k in range(self.n_layers):
                    self.encoders[i].append(IconCombo(self.in_size, self.gat_shapes,
                            self.alpha,net_weights[k][i] if i in net_weights[k] else net_weights[k][0]))
                    self.add_module(f"Co_Encoder_{i}_{k}", self.encoders[i][k])

        elif self.gat_type == 'single_single':
            for i in range(self.n_modalities):
                # self.encoders[i] = []
                self.encoders[i].append(IconCombo(self.in_size, self.gat_shapes, self.alpha,
                    net_weights[0][i] if i in net_weights[0] else net_weights[0][0]))
                self.add_module(f"Co_Encoder_{i}_0", self.encoders[i][0])

        #TODO Work on an encoder where across all networks for a certain layer only one w is trained.
        elif self.gat_type == 'abs_single':
            for k in range(self.n_layers):
                #initialize dropout=0 as I will use dropout even before calling the self.gat.forward() for this param setup
                icon_combo = IconCombo(self.in_size, self.gat_shapes, self.alpha, net_weights[k][0], dropout=0)
                for i in range(self.n_modalities):
                    self.encoders[i].append(icon_combo)
                    self.add_module(f"Co_Encoder_{i}_{k}", self.encoders[i][k])



    def compute_init_feat(self):
        feats = {}
        if self.pre_gat_type=='single':
            for i in range(self.n_modalities):
                if self.feats is None: #that use one-hot encoding as feature
                    feats[i] = torch.t(self.pre_gat.weight) + self.pre_gat.bias
                else: #use passed feature and reduce dimension
                    feats[i] = self.pre_gat(self.feats[i].to(Device()))
        elif self.pre_gat_type=='sep':
            for i in range(self.n_modalities):
                if self.feats is None:
                    feats[i] = torch.t(self.pre_gats[i].weight) + self.pre_gats[i].bias
                else:
                    feats[i] = self.pre_gats[i](self.feats[i].to(Device()))
        return feats

    def forward(
        self,
        data_flows: List[Tuple[int, Tensor, List[Adj]]],
        masks: Tensor,
        evaluate: bool = False,
        rand_net_idxs: Optional[np.ndarray] = None,
    ):
        """Forward pass logic.

        Args:
            data_flows (List[Tuple[int, Tensor, List[Adj]]]): Sampled bi-partite data flows.
                See PyTorch Geometric documentation for more details.
            masks (Tensor): 2D masks indicating which nodes (rows) are in which networks (columns)
            evaluate (bool, optional): Used to turn off random sampling in forward pass.
                Defaults to False.
            rand_net_idxs (np.ndarray, optional): Indices of networks if networks are being
                sampled. Defaults to None.

        Returns:
            Tensor: 2D tensor of final reconstruction to be used in loss function.
            Tensor: 2D tensor of integrated node features. Each row is a node, each column is a feature.
            List[Tensor]: Pre-integration network-specific node feature tensors. Not currently
                implemented.
            Tensor: Learned network scaling coefficients.
            Tensor or None: 2D tensor of label predictions if using supervision.
        """
        if rand_net_idxs is not None:
            idxs = rand_net_idxs
        else:
            idxs = list(range(self.n_modalities))
        net_scales, interp_masks = self.interp(masks, idxs, evaluate)
        # Define encoder logic.
        out_pre_cat_layers = []  # Final layers before concatenation, not currently used

        batch_size = data_flows[0][0]

        n_layers = self.gat_shapes["n_layers"]
        attns = {}
        x_cogats = {}
        edge_indices_global = {}

        #Initialize feature matrix with pre_gat features
        feats = self.compute_init_feat()

        agg_attn_mats = {}
        for k in range(n_layers): # at each layer

            x_cogats[k] = {}
            # for kth layer or more appropriately 'block' each block containing one gat and one co_gat
            for i, data_flow in enumerate(data_flows):#for each input network
                #for ith network
                if self.shared_encoder:
                    net_idx = 0
                else:
                    net_idx = idxs[i]
                if k == 0:
                    f = feats[net_idx]
                # for the first layer pass feat as x and later pass x_cogats computed in previous layer.
                # choose i)network (net_idx) and ii)layer (k) specific encoder if present
                # if only one encoder present across all layers then k=0
                cur_encoder_idx = k if k<len(self.encoders[net_idx]) else 0
                if (self.gat_type=='sep_sep' or self.gat_type=='sep_single'):
                    attns[net_idx] = self.encoders[net_idx][cur_encoder_idx](data_flow, k,
                        x=x_cogats[k-1][net_idx] if (k-1) in x_cogats else f)
                elif (self.gat_type=='single_sep' or self.gat_type=='single_single'or self.gat_type=='abs_single'):
                    attns[net_idx] = self.encoders[net_idx][cur_encoder_idx].compute_icon_attention(data_flow, k,
                        x=x_cogats[k - 1][net_idx] if (k - 1) in x_cogats else f)

                #get global edge index for each network at layer k
                local_edge_idx = attns[net_idx][0] #the edges for which we have attention value.
                cur_n_id = data_flow[1]
                edge_indices_global[net_idx] = edge_idx_from_local_to_global(local_edge_idx, cur_n_id)


            #TODO compute aggregate attention once
            for i, data_flow in enumerate(data_flows):
                net_idx = idxs[i]
                #TODO: check if I am taking edge index from right layer in dataflow.
                cur_encoder_idx = k if k<len(self.encoders[net_idx]) else 0
                if (self.gat_type=='sep_sep' or self.gat_type=='sep_single'):
                    x_cogats[k][net_idx] = self.co_encoders[net_idx][cur_encoder_idx]( data_flow, attns,
                        edge_indices_global, k, x_cogats[k-1][net_idx] if (k-1) in x_cogats else f)
                elif (self.gat_type=='single_sep' or self.gat_type=='single_single'):
                    x_cogats[k][net_idx], agg_attn_mats[(i,k)] = self.encoders[net_idx][cur_encoder_idx](data_flow, attns,
                        edge_indices_global, k, x_cogats[k - 1][net_idx] if (k - 1) in x_cogats else f)


            #Nure: Do embedding generation once for all networks.
            if self.gat_type=='abs_single':
                #compute aggregated attention across all networks
                icon_comb = self.encoders[net_idx][cur_encoder_idx]
                net_weights = icon_comb.net_weights()  # give normalized netweights
                cogat_edge_idx_global = {}
                w_cogat_attn = {}

                # take attention from all inout networks
                for net_idx in attns:
                    cogat_attn = attns[net_idx][1]
                    cogat_edge_idx_global[net_idx] = edge_indices_global[net_idx]
                    # Multiply cogat_attn with co_W
                    w_cogat_attn[net_idx] = net_weights[:, net_idx] * cogat_attn
                agg_attn_mat = aggregat_edge_weights(w_cogat_attn, cogat_edge_idx_global,
                                            icon_comb.in_size, icon_comb.n_heads, device='cuda')
                agg_edges = agg_attn_mat.indices()
                agg_alpha = agg_attn_mat.values()

                #dropout
                agg_alpha = F.dropout(agg_alpha, p=self.dropout, training=self.training)

                for i, data_flow in enumerate(data_flows):
                    net_idx = idxs[i]
                    cur_encoder_idx = k if k < len(self.encoders[net_idx]) else 0
                    x_cogats[k][net_idx] = self.encoders[net_idx][cur_encoder_idx](data_flow, attns,
                    edge_indices_global, k, x_cogats[k - 1][net_idx] if (k - 1) in x_cogats else f, agg_edges, agg_alpha)


        # Now take a weighted average of each node's features across all networks
        # x_all_modality = torch.zeros((batch_size, self.integration_size), device=Device())  # Tensor to store results from each modality.

        xs = []
        for net_idx in range(self.n_modalities):
            #weighted sum of features across networks
            if self.residual: #if residual=true then sum across all the hidden layers
                x = sum(x_cogats[layer_k][net_idx][:batch_size] for layer_k in x_cogats)
            else: #else take the value from last hidden layer
                x = x_cogats[n_layers-1][net_idx]

            if self.agg == 'avg':
                if self.bionic_mask:
                    x = net_scales[:, net_idx] * interp_masks[:, net_idx].reshape((-1, 1)) * x
                else:
                    x = net_scales[:, net_idx] * x

            xs.append(x)

        if self.agg=='avg':
            x_all_modality = torch.sum(torch.stack(xs), dim=0)
        elif self.agg=='concat':
            x_all_modality = torch.cat(xs, dim=1)

        #Now compute how different are the feature matrix across networks
        sum_diff, diffs, norm_diffs, sum_norm_diff = compute_feature_dist(xs)

        # Embedding
        emb = self.emb(x_all_modality)

        # Dot product (network reconstruction)
        dot = torch.mm(emb, torch.t(emb))

        # Classification (if standards are provided)
        if self.cls_heads:
            classes = [head(emb) for head in self.cls_heads]
        else:
            classes = None

        return dot, emb, None, net_scales, classes, sum_diff, diffs, norm_diffs, sum_norm_diff, agg_attn_mats


class IconParallel(Icon):
    def __init__(self, *args, **kwargs):
        """A GPU parallelized version of `Bionic`. See `Bionic` for arguments. 
        """
        super(IconParallel, self).__init__(*args, **kwargs)

        self.cuda_count = torch.cuda.device_count()

        # split network indices into `cuda_count` chunks
        self.net_idx_splits = torch.tensor_split(torch.arange(len(self.encoders)), self.cuda_count)

        # create a dictionary mapping from network idx to cuda device idx
        self.net_to_cuda_mapper = {}

        # distribute encoders across GPUs
        encoders = []
        for cuda_idx, split in enumerate(self.net_idx_splits):
            split_encoders = [self.encoders[idx].to(f"cuda:{cuda_idx}") for idx in split]
            encoders += split_encoders
            for idx in split:
                self.net_to_cuda_mapper[idx.item()] = cuda_idx
        self.encoders = encoders

        for i, enc_module in enumerate(self.encoders):
            self.add_module(f"Encoder_{i}", enc_module)

        # put remaining tensors on first GPU
        self.emb = self.emb.to("cuda:0")
        self.interp = self.interp.to("cuda:0")

        if self.cls_heads is not None:
            self.cls_heads = [head.to("cuda:0") for head in self.cls_heads]

            for h, cls_head in enumerate(self.cls_heads):
                self.add_module(f"Classification_Head_{h}", cls_head)

    def forward(
        self,
        data_flows: List[Tuple[int, Tensor, List[Adj]]],
        masks: Tensor,
        evaluate: bool = False,
        rand_net_idxs: Optional[np.ndarray] = None,
    ):
        """See `Bionic` forward methods for argument details.
        """
        if rand_net_idxs is not None:
            raise NotImplementedError("Network sampling is not used with model parallelism.")

        idxs = list(range(self.n_modalities))
        net_scales, interp_masks = self.interp(masks, idxs, evaluate, device="cuda:0")

        # Define encoder logic.
        out_pre_cat_layers = []  # Final layers before concatenation, not currently used

        batch_size = data_flows[0][0]
        x_store_modality = torch.zeros(
            (batch_size, self.integration_size), device="cuda:0"
        )  # Tensor to store results from each modality.

        # Iterate over input networks
        for i, data_flow in enumerate(data_flows):
            if self.shared_encoder:
                net_idx = 0
            else:
                net_idx = idxs[i]
            device = f"cuda:{self.net_to_cuda_mapper[net_idx]}"

            x = self.encoders[net_idx](data_flow, device).to("cuda:0")
            # x = net_scales[:, i] * interp_masks[:, i].reshape((-1, 1)) * x
            x_store_modality += x

        # Embedding
        emb = self.emb(x_store_modality)

        # Dot product (network reconstruction)
        dot = torch.mm(emb, torch.t(emb))

        # Classification (if standards are provided)
        if self.cls_heads:
            classes = [head(emb) for head in self.cls_heads]
        else:
            classes = None

        return dot, emb, out_pre_cat_layers, net_scales, classes
