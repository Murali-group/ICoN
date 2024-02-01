import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import softmax
import sys

net_name_mapping = {'Costanzo-2016.txt': 'GI Network', 'Hu-2007.txt':'COEX Network', 'Krogan-2006.txt': 'PPI Network'}

def parse_netscale_file(net_scale_file, noise):
    with open(net_scale_file,'r') as file:
        file_content = file.read()

    dataframes=pd.DataFrame()

    # Initialize dataset_name variable
    dataset_name = None

    #first line contains dataset names in order
    lines = [line for line in file_content.split('\n') if line.strip()]

    ordered_dataset_name = lines[0].split(' ')
    # Iterate through lines in the file
    for line in lines[1:]:  # Skip the first line
        # Check if the line contains the dataset name
        if '.txt' in line:
            dataset_name = line.strip()
            # Create a dataframe with three columns
            df = pd.DataFrame(columns=['cur_dataset', 'noise']+ ordered_dataset_name)
        else:
            # Split the line to get layer information
            layer_info = line.split('\t')
            layer_name = layer_info[0]
            layer_values = eval(layer_info[1])
            # convert all netweights into probabilities using softmax
            layer_values = list(softmax(layer_values))
            df.loc[layer_name] = [dataset_name, noise] +  layer_values

            # If the last line for the current dataset is reached, store the dataframe
            if layer_name == 'layer_2':
                dataframes= pd.concat([dataframes,df], axis=0)

    dataframes.reset_index(inplace=True)
    dataframes.rename(columns={'index':'layer'}, inplace=True)
    return dataframes, ordered_dataset_name


def main(input_dir):
    # input_dir = '/home/grads/tasnina/Submission_Projects/BIONIC-evals/bioniceval/datasets/yeast/ICON/Costanzo-2016-Hu-2007-Krogan-2006/'
    input_dir = input_dir + '/yeast/ICON/Costanzo-2016-Hu-2007-Krogan-2006/'
    output_dir = input_dir + '/attention_weights/'

    noises = [[0, 0],[0.1, 0.1], [0.3, 0.3], [0.5, 0.5]]
    parsed_weights_dict= {key:pd.DataFrame() for key in str(noises)}
    for noise in noises:
        noise_str = '_'.join(str(i).replace('.','-') for i in noise)
        net_scale_file = input_dir + \
            f'gat-68-10-3-0-4_co-attn-True_emb-512_e-3000_lr-0-0005_nbr-2_noise-{noise_str}/run_0_net_scales.txt'
        parsed_weights_dict[str(noise)], ordered_dataset_name=parse_netscale_file(net_scale_file, str(noise))
    print('hello')



    #heatmap for just 0th layer
    for layer in [0,1,2]:
        plt.clf()
        # n_rows=2
        # n_cols = int(len(noises)/n_rows)
        self_attn_per_network = {net_name:[] for net_name in ordered_dataset_name}
        attn_from_others_per_network = {net_name:[] for net_name in ordered_dataset_name}

        noises_percent = []
        for i in range(len(noises)):
            noise=noises[i]
            n = int(noise[0]) * 100
            noise_percent = f'{n}%'
            noises_percent.append(noise_percent)

            df = parsed_weights_dict[str(noise)]
            df = df[df['layer']==f'layer_{layer}'][['cur_dataset']+ordered_dataset_name]
            df_reordered = df.set_index('cur_dataset').reindex(ordered_dataset_name)

            #take the the diagonal values only
            df_vals = df_reordered.values
            self_attn_values = df_vals.diagonal()
            others_attn = np.sum(df_vals, axis=0)-self_attn_values

            for i in range(len(ordered_dataset_name)):
                self_attn_per_network[ordered_dataset_name[i]].append(self_attn_values[i])
                attn_from_others_per_network[ordered_dataset_name[i]].append(others_attn[i])


        ## Plot self-attention
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        for net_name in ordered_dataset_name:
            y = self_attn_per_network[net_name]
            plt.plot(noises_percent, y, label=net_name_mapping[net_name], marker='o')

        plt.xlabel('Noise')
        plt.ylabel('Self-attention')
        plt.title('')
        plt.legend()
        plt.tight_layout()
        # plt.show()


        # filename = output_dir+f'yeast_layer_{layer}_lineplot'+'.pdf'
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        # plt.savefig(filename)


        ##Plot attention from others
        # plt.clf()
        plt.subplot(1, 2, 2)
        for net_name in ordered_dataset_name:
            y = attn_from_others_per_network[net_name]
            plt.plot(noises_percent, y, label=net_name_mapping[net_name], marker='o')

        plt.xlabel('Noise')
        plt.ylabel('Total attention from other networks')
        plt.title('')
        plt.legend()
        plt.tight_layout()
        # plt.show()

        filename = output_dir + f'yeast_layer_{layer}_lineplot' + '.pdf'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)


if __name__=='__main__':
    input_dir = sys.argv[1]
    main(input_dir)