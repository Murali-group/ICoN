
import sys
import json
from pathlib import Path
import pandas as pd
import os
from itertools import combinations
import random


def main():
    pombe_biogrid_file = "/home/grads/tasnina/Projects/ICON/datasets/pombe/networks/all_networks.txt"
    nets_df = pd.read_csv(pombe_biogrid_file, sep='\t')[['Systematic Name Interactor A',
                                                   'Systematic Name Interactor B','Publication Source']]
    print(nets_df.columns)

    #now filter out the S. pombe interactions from three networks mentioned in BIONIC
    keep_PMIDs = {'Ryan-2012-gi': '22681890', 'Martin-2017-coex': '28041796', 'VO-2016-ppi': '26771498' }
    for net_name in keep_PMIDs:
        pmid=keep_PMIDs[net_name]
        spec_net_df = nets_df[nets_df['Publication Source']=='PUBMED:'+pmid][['Systematic Name Interactor A',
                                                   'Systematic Name Interactor B']]

        #save to file
        spec_net_file = '/home/grads/tasnina/Projects/ICON/inputs/'+net_name+'.txt'
        spec_net_df.to_csv(spec_net_file, sep=" ", index=False, header=False)

main()