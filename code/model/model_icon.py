import warnings
import torch.nn as nn
import torch.nn.functional as F

from utils.common import Device
from typing import Dict, List, Tuple, Optional
from .layers import ComboWGATConv,  Interp, NetWeights
from .model_utils import *

class IconCombo(nn.Module):
    def __init__(self, in_size: int, gat_shapes: Dict[str, int], alpha: float = 0.1,
            net_weights =None, dropout=None):

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
        alpha: float = 0.1,
        svd_dim: int = 0,
        shared_encoder: bool = False,
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
        self.svd_dim = svd_dim
        self.shared_encoder = shared_encoder
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

        if self.gat_type == 'single_sep':
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

        #An encoder where across all networks for a certain layer only one w is trained.
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
        data_flows: List[Tuple[int, torch.Tensor, List[Adj]]],
        masks: torch.Tensor,
        evaluate: bool = False,
        rand_net_idxs: Optional[np.ndarray] = None,
    ):


        if rand_net_idxs is not None:
            idxs = rand_net_idxs
        else:
            idxs = list(range(self.n_modalities))
        net_scales, _ = self.interp(masks, idxs, evaluate)

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

        return dot, emb, None, net_scales, sum_diff, diffs, norm_diffs, sum_norm_diff, agg_attn_mats


