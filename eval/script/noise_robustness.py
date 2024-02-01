import pandas as pd
import seaborn as sns
from script.paper_plots import plot_bars


#********************************** PLOT Utils******************************************

PALETTE = sns.cubehelix_palette(start=0.3, hue=1,
            gamma=0.4,dark=0.1, light=0.9,
            rot=-0.6,reverse=True, n_colors=3)
MODELS = ['ICON', 'BIONIC', 'Union']
standard = 'IntAct'


def sort_according_to_model_names(df, models=MODELS):
    # sort model_df according to model names
    df['Dataset_name'] = pd.Categorical(df['Dataset_name'], categories=models, ordered=True)
    df = df.sort_values(by='Dataset_name')
    return df

def compute_relative_score_drop(diff_noise_df, metric ='Module Match Score (AMI)'):
    #compute mean of all the runs and sample modules for specific noise-Dataset_name pair
    agg_df = diff_noise_df.groupby(['noise', 'Dataset_name']).agg({metric: 'mean'}).reset_index()

    #now compute relative drop along added level of noise by considering AMI@noise 0 to be highest or 1, for a certain Dataset_name.
    model_highest = {x:0 for x in MODELS}
    noises=[]
    models=[]
    amis=[]
    for index, row in agg_df.iterrows():
        model_name=  row['Dataset_name']

        noises.append(row['noise'])
        models.append(model_name)
        if row['noise']=='0%':
            model_highest[model_name] = row[metric]
            amis.append(1)
        else:
            amis.append(row[metric]/model_highest[model_name])
    relative_df = pd.DataFrame({'noise':noises, 'Dataset_name':models, 'Relative AMI': amis})
    return relative_df

def main():
    input_dir = '/home/grads/tasnina/Submission_Projects/BIONIC-evals/bioniceval/results/'
    out_dir = input_dir+'noisyinput/module_detection/'
    ##************ module detection files
    noise_file=input_dir+ 'noisyinput_icon_bionic_union_module_detection.tsv'
    noise_df = pd.read_csv(noise_file, sep='\t')[['Standard','Dataset','Module Match Score (AMI)']]

    #remove runx
    # noise_df['Dataset_name'] = noise_df['Dataset'].astype(str).apply(lambda x: x.split('__')[0])

    noise_df['noise'] = noise_df['Dataset'].astype(str).apply(lambda x: x.split('_')[-1])
    noise_df['Dataset_name'] = noise_df['Dataset'].astype(str).apply(lambda x: x.split('_')[0])

    #Keep chosen models
    noise_df = noise_df[noise_df['Dataset_name'].isin(MODELS)]
    #Keep chosen standard
    noise_df = noise_df[noise_df['Standard']==standard]

    #remove any suffix from any model name (e.g., convert ICON_l3_nomask to ICON) in the Dataset_name
    # noise_df['Dataset_name'] = noise_df['Dataset_name'].astype(str).apply(lambda x:x.split('_')[0])


    #plot
    # plot ICON with other models
    noise_df = sort_according_to_model_names(noise_df,models=MODELS)
    plot_bars(noise_df, "noise", "Module Match Score (AMI)", "Dataset_name",
              3, "Module Detection",
              out_dir + standard+'_comapare_model_robustness_exact_module_detection.png',
              palette=PALETTE)


    #now plot relative drop of AMI across different noisy input networks
    relative_noise_df = compute_relative_score_drop(noise_df, metric ='Module Match Score (AMI)')
    relative_noise_df = sort_according_to_model_names(relative_noise_df,models=MODELS)
    plot_bars(relative_noise_df, "noise", "Relative AMI", "Dataset_name",
              3, "Module Detection",
              out_dir + standard + '_comapare_model_robustness_relative_module_detection.png',
              palette=PALETTE)

    print('done')

if __name__=='__main__':
    main()