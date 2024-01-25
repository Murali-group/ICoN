'''
Create union of the input newtorks such that the new network contain all the nodes and edges from the input networks.
If one edge appears in multiple netwroks then take the highest weight. (Ref: BIONIC (Methods: Obtaining integrated results))
'''

import sys
import json
from pathlib import Path
import pandas as pd
import os
from itertools import combinations
import random


def get_net_name(filename):
    return filename.split('/')[-1].replace('.txt','')


def init_process(net_file_names, delimiter=' '):
    # read net_files
    net_dict = {get_net_name(file_name): pd.read_csv(file_name, delimiter=delimiter, header=None) for file_name in
               net_file_names}
    # Add weights of 1.0 if weights are missing
    for net_name in net_dict:
        net_df = net_dict[net_name]
        if net_df.shape[1] < 3:
            net_df[2] = pd.Series([1.0] * len(net_df))
    return net_dict


def make_undirected(net_df):
    df = pd.DataFrame({0: net_df[1], 1: net_df[0], 2: net_df[2]})
    df = pd.concat([net_df, df]).drop_duplicates()
    return df

def net_union_wrapper(net_dict, out_dir):
    #make the networks undirected
    union_df = pd.concat(list(net_dict.values()))
    #make it undirected
    union_df = make_undirected(union_df)

    #take the maximum weight if an edge is present in multiple networks
    union_df = union_df.groupby([0, 1], as_index=False)[2].max()

    #save union
    out_file  = out_dir + '/' + '_'.join(list(net_dict.keys()))+'.txt'
    union_df.to_csv(out_file, sep=' ', header=None, index=False)
    return union_df

def add_noise_to_networks(net_dict, noise_list, out_dir):
    noise_added_net_dict={}
    count=0
    for net_name in net_dict:
        net_df = net_dict[net_name]
        noise = noise_list[count]

        #add noise by adding and dropping noise% of non-loop edges
        #remove self-loops
        net_df = net_df[net_df[0]!=net_df[1]]

        nodes = list(set(net_df[0]).union(set(net_df[1])))
        orig_edges = set(zip(net_df[0], net_df[1]))
        n_edges = len(net_df)
        n_drop_add_edges = int(n_edges*noise)

        # add edges
        # find edges that are not in current network
        all_edge_combo = set(combinations(nodes, 2))
        absent_edges = list(all_edge_combo.difference(orig_edges))
        added_edges = random.sample(absent_edges, n_drop_add_edges)
        added_net_df = pd.DataFrame({0: [a for (a,b) in added_edges],
                                     1: [b for (a,b) in added_edges],
                                     2: [1.0]*len(added_edges)})


        #drop edges
        drop_saved_df = net_df.sample(frac=(1-noise), random_state=1)

        #final noise added net_df
        noise_added_net_dict[net_name+'_noise-'+str(noise).replace('.','-')] = pd.concat([drop_saved_df, added_net_df], axis=0)



        #save to file
        out_file = out_dir+ net_name + '_noise-'+str(noise).replace('.','-')+'.txt'
        noise_added_net_dict[net_name+'_noise-'+str(noise).replace('.','-')].to_csv(out_file, sep=' ', header=None, index=False)
        count += 1
    return noise_added_net_dict




def main(config_path):
    with config_path.open() as f:
        config_dict = json.load(f)
    out_dir = os.path.dirname(config_dict['net_names'][0])+'/'
    net_dict = init_process(config_dict['net_names'])

    #get union of network files
    net_union_wrapper(net_dict, out_dir)

    #TODO: uncomment for noisy network
    #get noisy network files
    p=0.3 #add frac p noise to each network
    noise_list =[p]*len(net_dict.keys())
    noise_added_net_dict = add_noise_to_networks(net_dict, noise_list, out_dir)

    #get union of noisy network
    net_union_wrapper(noise_added_net_dict, out_dir)



if __name__=='__main__':
    if len(sys.argv) == 1: #no argument passed so use a default config file.
        config_path = Path('/home/grads/tasnina/Projects/ICON/code/config/icon_best_human_ppi.json')
    else:
        config_path = Path(sys.argv[1])
    print("config_path: ", config_path)

    main(config_path)

