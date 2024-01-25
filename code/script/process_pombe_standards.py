import sys
import json
from pathlib import Path
import pandas as pd
import os
from itertools import combinations
import random


def process_coannotation(go_df, out_dir):

    all_genes = list(set(go_df['systematic_id']))

    go_2_gene_dict = {go: [] for go in list(set(go_df['acc']))}
    for index, row in go_df.iterrows():
        go_2_gene_dict[row['acc']].append(row['systematic_id'])

    pos_pairs = []
    for go in go_2_gene_dict:
        genes= go_2_gene_dict[go]
        #find all pairwise combinations of genes
        pos_pairs += combinations(genes, 2)
    all_possible_pairs = combinations(all_genes, 2)
    neg_pairs = list(set(all_possible_pairs).difference(set(pos_pairs)))

    srcs = [a for (a,b) in pos_pairs] + [a for (a,b) in neg_pairs]
    dsts = [b for (a,b) in pos_pairs] + [b for (a,b) in neg_pairs]
    scores= [1.0]*len(pos_pairs) + [0]*len(neg_pairs)

    df = pd.DataFrame({'src': srcs, 'dst': dsts, 'score': scores })

    coannotation_file = out_dir + 'coannotation-prediction/pombe-GO-coannotation.csv'
    df.to_csv(coannotation_file, sep=',', index=False, header=False)





def process_function_prediction(go_df, out_dir):
    #prepare and save the function-prediction standard
    #Filter GO terms to only include classes with 20 or more members

    count_df = go_df.groupby('acc').size().reset_index(name='counts')
    res_acc= list(count_df[count_df['counts']>20]['acc'])
    filt_go_df=go_df[go_df['acc'].isin(res_acc)]

    gene_2_go_dict = {gene: [] for gene in list(set(filt_go_df['systematic_id']))}
    for index, row in filt_go_df.iterrows():
        gene_2_go_dict[row['systematic_id']].append(row['acc'])

    function_pred_file = out_dir + 'function-prediction/pombe-GO-labels.json'
    with open(function_pred_file, 'w') as json_file:
        json.dump(gene_2_go_dict, json_file)



def process_module_detection(go_df, out_dir):

    #Filter GO terms to only include classes with 2 or more members
    count_df = go_df.groupby('acc').size().reset_index(name='counts')
    res_acc= list(count_df[count_df['counts']>2]['acc'])
    filt_go_df=go_df[go_df['acc'].isin(res_acc)]

    go_2_gene_dict = {go: [] for go in list(set(filt_go_df['acc']))}
    for index, row in filt_go_df.iterrows():
        go_2_gene_dict[row['acc']].append(row['systematic_id'])

    # prepare and save the module_detection standard
    module_detection_file = out_dir + 'module-detection/pombe-GO-modules.json'
    with open(module_detection_file, 'w') as json_file:
        json.dump(go_2_gene_dict, json_file)


def main():
    # process pombe GO dataset for 3 tasks
    out_dir = "/home/grads/tasnina/Projects/BIONIC-evals/bioniceval/standards/"

    pombe_GO_file ="/home/grads/tasnina/Projects/ICON/datasets/pombe/standards/Complex_annotation.tsv"
    go_df = pd.read_csv(pombe_GO_file, sep='\t')[['acc','systematic_id','evidence_code']]

    #As per BIONIC's recommendation, filter out terms with IEA evidence code
    go_df = go_df[go_df['evidence_code']!='IEA']

    #As per BIONIC's recommendation, we should filter out terms with >30 proteins annotated to it. But doing that
    #gives very small samples.
    count_df = go_df.groupby('acc').size().reset_index(name='counts')
    res_acc= list(count_df[(count_df['counts']>1)]['acc'])
    go_df=go_df[go_df['acc'].isin(res_acc)]

    # ********************* Coannotation ********************
    process_coannotation(go_df, out_dir)

    #********************* Function prediction  *************
    process_function_prediction(go_df, out_dir)

    #********************* Module detection  *************
    process_module_detection(go_df, out_dir)


main()

