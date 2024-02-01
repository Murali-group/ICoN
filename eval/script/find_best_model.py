import os

import pandas as pd
def main():
    results_dir = '/home/grads/tasnina/Projects/BIONIC-evals/bioniceval/results/'
    pombe_coann_file = results_dir + 'icon_pombe_hyperparam_coannotation.tsv'
    pombe_func_file = results_dir + 'icon_pombe_hyperparam_function_prediction.tsv'
    pombe_module_file = results_dir + 'icon_pombe_hyperparam_module_detection.tsv'

    coann_df =  pd.read_csv(pombe_coann_file, sep='\t')
    func_df =  pd.read_csv(pombe_func_file, sep='\t')
    module_df =  pd.read_csv(pombe_module_file, sep='\t')

    #Take average across all runs of a certain model
    coann_df['Model'] = coann_df['Dataset'].astype(str).apply(lambda x:x.split('__')[0])
    func_df['Model'] = func_df['Dataset'].astype(str).apply(lambda x:x.split('__')[0])
    module_df['Model'] = module_df['Dataset'].astype(str).apply(lambda x:x.split('__')[0])

    coann_avg = coann_df.groupby('Model').agg({'Average Precision': 'mean'}).round(2)
    func_avg = func_df.groupby('Model').agg({'Accuracy': 'mean'}).round(2)
    module_avg = module_df.groupby('Model').agg({'Module Match Score (AMI)': 'mean'}).round(2)

    #rank the models
    coann_avg['rank'] = coann_avg['Average Precision'].rank(method='min', ascending=False)
    func_avg['rank'] = func_avg['Accuracy'].rank(method='min', ascending=False)
    module_avg['rank'] = module_avg['Module Match Score (AMI)'].rank(method='min', ascending=False)

    #sort the dfs from 3 different tasks according to model name so that we can compute the
    #average raning of each model across all three tasks.
    coann_avg = coann_avg.sort_index()
    func_avg = func_avg.sort_index()
    module_avg = module_avg.sort_index()

    #now take average rank across all three tasks

    df = pd.concat([coann_avg[['rank']], func_avg[['rank']], module_avg[['rank']]], axis=1)
    df['avg_rank'] = df.mean(axis=1)
    df = df.sort_values(by='avg_rank')

    rank_out_file = results_dir + 'pombe_model_ranks/'+ 'models.tsv'
    os.makedirs(os.path.dirname(rank_out_file), exist_ok=True)
    df.to_csv(rank_out_file , sep='\t')
    print('done ranking')


if __name__=='__main__':
    main()