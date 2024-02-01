import pandas as pd
import seaborn as sns
from script.paper_plots import plot_bars



#********************************** PLOT Utils******************************************
def get_palette(n_models):
    palette = sns.cubehelix_palette(start=0.3, hue=1,
                gamma=0.4,dark=0.1, light=0.9,
                rot=-0, n_colors=n_models)
    return palette
standards = ['IntAct','GO','KEGG']


def sort_according_to_model_names(df, models):
    # sort model_df according to model names
    df['Dataset_name'] = pd.Categorical(df['Dataset_name'], categories=models, ordered=True)
    df = df.sort_values(by='Dataset_name')
    return df

def wrapper_plot_ablation(ablation_df, metric, task,model_name_mapping, out_dir):
    #filter
    ablation_df = ablation_df[ablation_df['Dataset'].isin(list(model_name_mapping.keys()))].copy()

    ablation_df['Dataset_name'] =ablation_df['Dataset'].astype(str).\
                        apply(lambda x: model_name_mapping[x] if x in model_name_mapping else x)

    #    #*************** PLOT for with or without attn
    #Keep chosen models
    models = list(model_name_mapping.values())
    #plot
    ablation_filt_df = sort_according_to_model_names(ablation_df, models=models)
    plot_bars(ablation_filt_df, "Standard", metric, "Dataset_name",
              3, task,
              out_dir +f'ablation_study_coattn_{task}.png',
              palette=get_palette(2), remove_prefix=False, ord = standards)



def main():
    input_dir = '/home/grads/tasnina/Submission_Projects/BIONIC-evals/bioniceval/results/'
    model_name_mapping = {'without noise__run0':'without noise', 'with noise__run0':'with noise' }

    ##************ Ablation study on module detection files******************
    task = 'Module Detection'
    metric = 'Module Match Score (AMI)'
    module_file = input_dir + 'ablation_nonoise_module_detection.tsv'
    module_df = pd.read_csv(module_file, sep='\t')[['Standard', 'Dataset', metric]]
    wrapper_plot_ablation(module_df, metric, task, model_name_mapping, input_dir)



    #ablation study for function pred
    task = 'Function Prediction'
    metric = 'Accuracy'
    func_file = input_dir + 'ablation_nonoise_function_prediction.tsv'
    func_df = pd.read_csv(func_file, sep='\t')[['Standard', 'Dataset', metric]]
    wrapper_plot_ablation(func_df, metric, task,model_name_mapping, input_dir)




    #ablation study for coannotation pred
    task = 'Co-annotation Prediction'
    metric = 'Average Precision'
    coann_file = input_dir + 'ablation_nonoise_coannotation.tsv'
    coann_df = pd.read_csv(coann_file, sep='\t')[['Standard', 'Dataset', metric]]
    wrapper_plot_ablation(coann_df, metric, task, model_name_mapping, input_dir)





    print('done')

if __name__=='__main__':
    main()