from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#********************************** PLOT Utils******************************************
PALETTE = sns.cubehelix_palette(start=0.3, hue=1,
            gamma=0.4,dark=0.1, light=0.9,
            rot=-0.6,reverse=True, n_colors=6)


EDGE_WIDTH = 0.5
EDGE_COLOUR = "#666666"
CAPSIZE = 0.05



def plot_bars(df: pd.DataFrame, x: str, y: str, hue: str, legend_col:int, title: str,
              out_path: Path, palette = PALETTE, remove_prefix = True, ord=['IntAct','GO','KEGG']):
    #n compared models or networks
    n_compared = len(list(df['Dataset_name'].unique()))
    plt.clf()
    plt.figure(figsize=(4, 6))

    #change the Dataset_name to keep only the first part e.g., BERTWalk_given --> BERTWalk
    if remove_prefix:
        df['Dataset_name'] = df['Dataset_name'].astype(str).apply(lambda x: x.split('_')[0])

    ax = sns.barplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        linewidth=EDGE_WIDTH,
        edgecolor=EDGE_COLOUR,
        capsize=CAPSIZE,
        ci='sd',#this plots an error bar extending one standard deviation up and below the mean value.
        order = ord
        )
    sns.despine(offset={"left": 10})
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    #customize legend
    legend= ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=legend_col)
    for i in range(n_compared):
        legend.get_texts()[i].set_fontsize('9')  # Adjust the font size of the legend labels
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(out_path.replace('.png','.pdf'))


def wrapper_plot_coann(coannotation_file, models, networks, order, out_dir):
    coann_df = pd.read_csv(coannotation_file, sep='\t')

    #seperate models and networks
    models_df, nets_df = sep_nets_models(coann_df, models, networks)

    # plot ICON with other models
    plot_bars(models_df, "Standard", "Average Precision", "Dataset_name",
              2, "Co-annotation",
              out_dir + 'comapare_models_coannotation.png', ord=order)

    #plot ICON with individual networks
    plot_bars(nets_df, "Standard", "Average Precision", "Dataset_name",
              2, "Co-annotation",
              out_dir+'compare_nets_coannotation.png',ord=order)


def wrapper_plot_module_detection(module_file, models, networks,order, out_dir):

    module_df = pd.read_csv(module_file, sep='\t')


    # seperate models and networks
    models_df, nets_df = sep_nets_models(module_df, models, networks)


    # plot ICON with other models
    plot_bars(models_df, "Standard", "Module Match Score (AMI)", "Dataset_name",
              2, "Module Detection",
              out_dir + 'comapare_models_module_detection.png', ord=order)

    # plot ICON with individual networks
    plot_bars(nets_df, "Standard", "Module Match Score (AMI)", "Dataset_name",
              2, "Module Detection",
              out_dir + 'compare_nets_module_detection.png', ord=order)

def wrapper_plot_function_pred(func_file, models, networks,order, out_dir):
    func_df = pd.read_csv(func_file, sep='\t')

    #seperate models and networks
    models_df, nets_df = sep_nets_models(func_df, models, networks)

    # plot ICON with other models
    plot_bars(models_df, "Standard", "Accuracy", "Dataset_name",
              2, "Function Prediction",
              out_dir + 'comapare_models_function_prediction_acc.png', ord=order)
    plot_bars(models_df, "Standard", "Micro F1", "Dataset_name",
              2, "Function Prediction",
              out_dir + 'comapare_models_function_prediction_microF1.png', ord=order)
    plot_bars(models_df, "Standard", "Macro F1", "Dataset_name",
              2, "Function Prediction",
              out_dir + 'comapare_models_function_prediction_macroF1.png', ord=order)

    #plot ICON with individual networks
    plot_bars(nets_df, "Standard", "Accuracy", "Dataset_name",
              2, "Function Prediction",
              out_dir+'compare_nets_function_prediction_acc.png', ord=order)
    plot_bars(nets_df, "Standard", "Micro F1", "Dataset_name",
              2, "Function Prediction",
              out_dir + 'comapare_nets_function_prediction_microF1.png', ord=order)
    plot_bars(nets_df, "Standard", "Macro F1", "Dataset_name",
              2, "Function Prediction",
              out_dir + 'comapare_nets_function_prediction_macroF1.png', ord=order)

#****************************** Data processing utils*******************************
def sep_nets_models(df, models, networks):
    '''
    Given a file with downstream performance values, this function will create two dfs one having
    ICON with other networks and another having ICON with other models.
    '''
    #separate rows with icon
    df['Dataset_name'] = df['Dataset'].astype(str).apply(lambda x: x.split('__')[0])

    #separate the models
    models_df = df[df['Dataset_name'].apply(lambda x: any(model in x for model in models))].copy()
    #sort model_df according to model names
    models_df['Dataset_name'] = pd.Categorical(models_df['Dataset_name'], categories=models, ordered=True)
    models_df = models_df.sort_values(by='Dataset_name')


    #separate networks
    nets_df = df[df['Dataset_name'].apply(lambda x: (any(network in x for network in networks)))].copy()
    #sort model_df according to model names
    nets_df['Dataset_name'] = pd.Categorical(nets_df['Dataset_name'], categories=networks, ordered=True)
    nets_df = nets_df.sort_values(by='Dataset_name')


    return models_df, nets_df


def plot_for_yeast(input_dir):
    # ********************* For yeast ********************
    order = ['IntAct', 'GO', 'KEGG']
    models = ['ICoN', 'BERTWalk', 'BIONIC', 'deepNF', 'Mashup', 'Union']
    networks = ['ICoN', 'PPI Network', 'COEX Network', 'GI Network']

    # ************************************ COANNOTATION PLOTS **************************************
    coannotation_file = input_dir + 'yeast_coannotation.tsv'
    wrapper_plot_coann(coannotation_file, models, networks, order, input_dir)

    # *********************************** MODULE DETECTION *****************************************
    module_file = input_dir + 'yeast_module_detection.tsv'

    wrapper_plot_module_detection(module_file, models, networks, order, input_dir)

    # ************************************ Function prediction PLOTS **************************************
    func_file = input_dir + 'yeast_function_prediction.tsv'
    wrapper_plot_function_pred(func_file, models, networks, order, input_dir)

def plot_for_human(input_dir):
    # ********************* For human ********************
    models = ['ICoN', 'BERTWalk', 'BIONIC', 'deepNF', 'Mashup', 'Union']
    networks = ['ICoN', 'Rolland-14', 'Hein-15', 'Huttlin-15', 'Huttlin-17']
    order = None
    # ************************************ COANNOTATION PLOTS **************************************
    coannotation_file = input_dir + 'human_coannotation_coannotation.tsv'
    wrapper_plot_coann(coannotation_file, models, networks, order, input_dir)


def main():
    input_dir = '/home/grads/tasnina/Submission_Projects/BIONIC-evals/bioniceval/results/'

    plot_for_yeast(input_dir)
    plot_for_human(input_dir)

if __name__=='__main__':
    main()