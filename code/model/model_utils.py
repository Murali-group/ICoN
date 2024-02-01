import os
import torch
import torch_sparse
import numpy as np
from itertools import combinations
import random
from utils.sampler import Adj
from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import remove_self_loops, to_undirected

def add_random_indices(a, add_percentage,n_edges, device):
    row, col, edge_attr = a.adj_t.t().coo()
    n_nodes = a.adj_t.size(dim=0)

    #find edges that are not in current network
    all_edge_combo = set(combinations(list(range(n_nodes)), 2))
    orig_edges = set(zip(row.tolist(), col.tolist()))
    absent_edges = list(all_edge_combo.difference(orig_edges))

    #add add_percent (of original edges) false edges
    n_add = int(add_percentage * ((n_edges-n_nodes)/2))
    added_edges = random.sample(absent_edges, n_add)

    new_row = torch.tensor([a for (a, b) in added_edges] + [b for (a, b) in added_edges]).to(device)
    new_col = torch.tensor([b for (a, b) in added_edges] + [a for (a, b) in added_edges]).to(device)
    #TODO: currently giving edge_weight=1 as attribute. should I change that?
    new_attr = torch.tensor([1.0]*(new_row.size(dim=0))).to(device)

    mod_row = torch.cat((row, new_row), dim=0)
    mod_col = torch.cat((col, new_col), dim=0)
    mod_edge_index = torch.stack([mod_row, mod_col], dim=0)
    mod_edge_attr = torch.cat((edge_attr, new_attr), dim=0)

    # convert back to Data
    pyg_graph = Data(edge_index=mod_edge_index)
    pyg_graph.edge_weight = mod_edge_attr
    pyg_graph.num_nodes = n_nodes
    pyg_graph = ToSparseTensor(remove_edge_index=True)(pyg_graph)
    print('added false positives')

    return pyg_graph.to(device)



def drop_random_indices(a, drop_percentage, n_edges, device):
    row, col, edge_attr = a.adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0)
    n_nodes = a.adj_t.size(dim=0)
    dense_tensor = (torch.sparse_coo_tensor(indices=edge_index, values=edge_attr, size=
                    (n_nodes, n_nodes))).to_dense()
    upper_triangular = torch.triu(dense_tensor, diagonal=1)

    # Get the indices of non-zero elements
    edge_indices = upper_triangular.nonzero().t()
    # Extract the values at the non-zero indices
    edge_values = upper_triangular[edge_indices[0, :], edge_indices[1, :]]

    # non_zero = edge_indices.shape[1]
    up_tri_non_zero=int((n_edges-n_nodes)/2)
    # # Randomly select indices to keep
    num_elements_to_keep = int((1-drop_percentage) * up_tri_non_zero)
    indices_to_keep = torch.randperm(up_tri_non_zero)[:num_elements_to_keep].to(device)

    edge_indices = torch.index_select(edge_indices, dim=1, index=indices_to_keep)
    edge_values = torch.index_select(edge_values, dim=0, index=indices_to_keep)

    #make undirected again
    edge_indices, edge_values = to_undirected(edge_indices, edge_attr=edge_values)
    #add self loops
    union_idxs = list(range(n_nodes))
    self_loops = torch.LongTensor([union_idxs, union_idxs]).to(device)
    edge_indices = torch.cat([edge_indices, self_loops], dim=1)
    edge_values = torch.cat([edge_values, (torch.Tensor([1.0] * n_nodes)).to(device)])

    #convert back to Data
    pyg_graph = Data(edge_index=edge_indices)
    pyg_graph.edge_weight = edge_values
    pyg_graph.num_nodes = n_nodes
    pyg_graph = ToSparseTensor(remove_edge_index=True)(pyg_graph)
    print('added false negatives')
    return pyg_graph.to(device)



def compute_feature_dist(xs):
    '''
    xs is a list of features matrices containing computed features across networks.
    In this function we want to see how different are the feature matrices across networks.
    '''
    from itertools import combinations
    n_nets = len(xs)
    comb = list(combinations(list(range(n_nets)), 2))

    diffs = {}
    norm_diffs = {}
    sum_diff = 0
    sum_norm_diff = 0
    for i, j in comb:
        diff = (torch.mean((xs[i] - xs[j]) ** 2)).detach().cpu().numpy().item()

        # #nomalized diff:
        # mean_i = xs[i].mean(dim=0)
        # std_i = xs[i].std(dim=0)
        # normalized_xs_i = (xs[i] - mean_i) / std_i

        # mean_j = xs[j].mean(dim=0)
        # std_j = xs[j].std(dim=0)
        # normalized_xs_j = (xs[j] - mean_j) / std_j

        # NURE: find mean and standard deviation of a feature across all networks.
        x_ij = torch.cat((xs[i], xs[j]), dim=0)
        mean = x_ij.mean(dim=0)
        std = x_ij.std(dim=0)
        normalized_xs_i = (xs[i] - mean) / std
        normalized_xs_j = (xs[j] - mean) / std
        norm_diff = (torch.mean((normalized_xs_i - normalized_xs_j) ** 2)).\
                    detach().cpu().numpy().item()
        diffs[str((i, j))] = diff
        norm_diffs[str((i, j))] = norm_diff

        sum_diff += diff
        sum_norm_diff+=norm_diff

    return sum_diff, diffs, norm_diffs, sum_norm_diff

def batch_sampling(batch_loaders, node_ids, n_layers ):
    n_ids = torch.LongTensor(node_ids)
    data_flows = []
    for k in range(n_layers):  # for each layer
        for i in range(len(batch_loaders)):  # batchloeaders for each network
            batch_size, n_id, adj = batch_loaders[i].sample(n_ids if k==0 else data_flows[i][1])
            if k == 0:
                data_flows.append((batch_size, n_id, adj))
            else:
                data_flows[i][1] = n_id
                data_flows[i][2].append(adj[0])

        for i in range(len(data_flows)):
            data_flows[i] = list(data_flows[i])

        # find the union of n_id across networks for current layer
        n_ids = []
        for i in range(len(data_flows)):  # for each network
            n_id = data_flows[i][1].tolist()
            if i == 0:
                n_ids = n_id
            else:
                new_n_id = list(set(n_id).difference(set(n_ids)))
                n_ids = n_ids + new_n_id
        # order the n_ids so that for each net its initially sampled n_ids comes first and
        # then comes the unioned ones.

        for i in range(len(data_flows)):
            n_id = data_flows[i][1].tolist()
            new_n_id = list(set(n_ids).difference(set(n_id)))
            ordered_n_id = torch.LongTensor(n_id + new_n_id)
            data_flows[i][1] = ordered_n_id

            # update the adj for this layer accordingly
            cur_adj = data_flows[i][2][k]
            new_size = (len(ordered_n_id), cur_adj.size[1])
            data_flows[i][2][k] = Adj(cur_adj.edge_index, cur_adj.e_id, cur_adj.weights, new_size)

    # convert all data_flows[i] from list to tuple and change the orders of adjs
    for i in range(len(data_flows)):
        data_flows[i][2] = data_flows[i][2][::-1]  # change the orders of adjs
        data_flows[i] = tuple(data_flows[i])

    return data_flows

def edge_idx_from_local_to_global(edge_idx, n_id):
    '''
    This function will take a tensor with two rows containing source_node and target_node of the edges.
    Here, edge (10,121) indicates an edge between n_id[10] and n_id[121] wherever n_id contains the gobal
    indexing of a node.
    This function will convert the edge_idx such that it contains global indexing of nodes of edges.
    '''
    # start = time.time()
    s = edge_idx[0].tolist()
    t = edge_idx[1].tolist()

    node_idx_map = node_idx_local_to_global_mapping(n_id)
    s_global = [node_idx_map[x] for x in s]
    t_global = [node_idx_map[x] for x in t]
    edge_idx_global = torch.LongTensor([s_global, t_global])
    # end=time.time()
    # t1 =  end-start
    # print('local_to_global:', t1)
    # print('#edges', len(s))
    # print('#nodes', len(n_id))

    return edge_idx_global

def edge_idx_from_global_to_local(edge_idx, n_id):
    '''
    This function will take a tensor with two rows containing source_node and target_node of the edges.
    Here, edge (5120,121) contains the global index of nodes.
    This function will convert the edge_idx such that it contains local indexing of nodes. e.g., convert
    (5120, 121) to (1, 12) where n_id[1] = 5120, n_id[12] = 121
    '''
    # start = time.time()

    # if isinstance(edge_idx, Tensor):
    #     s = edge_idx[0].tolist()
    #     t = edge_idx[1].tolist()
    # else:
    #     s = edge_idx[0]
    #     t = edge_idx[1]

    s = edge_idx[0].tolist()
    t = edge_idx[1].tolist()

    node_idx_map = node_idx_global_local_mapping(n_id)
    s_local = [node_idx_map[x] for x in s]
    t_local = [node_idx_map[x] for x in t]
    edge_idx_local = torch.LongTensor([s_local, t_local])
    # end=time.time()
    # t1 =  end-start

    # print('global_to_local:', t1)
    # print('#edges', len(s))
    # print('#nodes', len(n_id))
    # a = list(tuple(zip(edge_idx.tolist()[0], edge_idx.tolist()[1])))
    # count=0
    # while True:
    #     if a[count]==(3139, 3896):
    #         print(count)
    #           break
    #     count+=1

    return edge_idx_local

def node_idx_global_local_mapping(n_id):
    return {x: i for i, x in enumerate(n_id.tolist())}

def node_idx_local_to_global_mapping(n_id):
    return {i: x for i, x in enumerate(n_id.tolist())}



def aggregat_edge_weights(w_cogat_attn, cogat_edge_idx_global, n_nodes, n_heads, device='cpu'):
    '''
    Inputs:
        - w_cogat_attn (dict): Contains weighted co-attention from each network.
          size = e*h: where e = number of edges, h=number of heads.
        - cogat_edge_idx_global (dict): Contains global edge index from each network.
        - n_nodes: total number of nodes across all networks

    Output:
    agg_attn: aggregated attention across networks for same edge.
    global_edges: A list of edges without any repeat edge.

    Function: Same edge can appear in multiple networks and have different
    weighted co-attn values. We want to sum the attention for a certain
    edge across all networks in this funcction.
    '''

    agg_attn_mat = torch.empty(0)
    for net_idx in w_cogat_attn:
        data = w_cogat_attn[net_idx]
        indices = cogat_edge_idx_global[net_idx]
        if agg_attn_mat.numel() == 0:
            agg_attn_mat = torch.sparse_coo_tensor(indices.to(device),
                            data.to(device), [n_nodes, n_nodes, n_heads])
        else:
            agg_attn_mat += torch.sparse_coo_tensor(indices.to(device),
                            data.to(device), [n_nodes, n_nodes, n_heads])
    ## Does coalesce() work as intended? It does.
    return agg_attn_mat.coalesce()


def degree_norm_adj(adj_matrix):
    '''
    Take an adjacency matrix as input and degree nomalize it.
    A => D^(-1/2) A D^(-1/2)
    '''

    # Calculate the degree matrix
    degree = torch_sparse.sum(adj_matrix, dim=1).to(torch.float32)

    # Calculate the inverse square root of the degree matrix
    degree_sqrt_inv = 1.0 / torch.sqrt(degree)

    # Create diagonal matrices for D^(-1/2)
    degree_sqrt_inv_diag = torch.diag(degree_sqrt_inv)

    # Convert the SparseTensor to a dense Tensor
    dense_adj_matrix = adj_matrix.to_dense()

    # Degree normalize the adjacency matrix
    normalized_adj_matrix = degree_sqrt_inv_diag @ dense_adj_matrix @ degree_sqrt_inv_diag


    #Min-max scalinge
    # dense_adj_matrix = adj_matrix.to_dense()
    # min, _ = dense_adj_matrix.min(dim=0)
    # max, _ = dense_adj_matrix.max(dim=0)
    # normalized_adj_matrix = (dense_adj_matrix-min)/(max-min)

    return normalized_adj_matrix


def process_agg_mat(agg_single_head):
    '''Convert agg_mat into a tow column matrix. Where for each row i, the first column
    contains self attention and the second column contains attention to all others.'''
    diagonal_elements = np.diag(agg_single_head)
    row_sums = np.sum(agg_single_head, axis=0) - diagonal_elements
    result_matrix = np.column_stack((diagonal_elements, row_sums))


    #now slice result matrix along rows and concat the slices along columns.
    return result_matrix
