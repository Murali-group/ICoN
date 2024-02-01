import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind


def compute_pvalues(coann_file, best_model, second_best_model, standards, metric='Average Precision', alternative='greater'):
    df = pd.read_csv(coann_file, sep='\t')
    df['Model'] = df['Dataset'].astype(str).apply(lambda x: x.split('__r')[0])

    p_val_Mannu = {}
    p_val_ttest = {}

    for standard in standards:
        df_standard = df[df['Standard']==standard]
        best_metric = list(df_standard[df_standard['Model']==best_model][metric])
        second_best_metric = list(df_standard[df_standard['Model']==second_best_model][metric])
        p_val_Mannu[standard] = mannwhitneyu(best_metric, second_best_metric, alternative=alternative).pvalue
        p_val_ttest[standard] = ttest_ind(best_metric, second_best_metric, alternative=alternative).pvalue

    return p_val_Mannu, p_val_ttest

def wrapper_significance_test_yeast(icon_model_name, compared_model, standards, alternative, coann_file, module_file, func_file, out_file):

    print(f'\n\n{icon_model_name} vs. {compared_model}')
    out_file.write(f'\n\n\n{icon_model_name} vs. {compared_model}\n')

    if not (compared_model=='BERTWalk_given'):#bertwalk_given has only one run. so no point in significance test for that.
        coann_pval_mannu,coann_pval_ttest  = compute_pvalues(coann_file, icon_model_name, compared_model,standards,
                                            metric='Average Precision', alternative=alternative)
        print('Coannotation \n' + str(coann_pval_mannu)+'\n' + str(coann_pval_ttest)+ '\n')
        out_file.write('Coannotation \nMannu: ' + str(coann_pval_mannu)+'\nttest: ' + str(coann_pval_ttest)+ '\n')
    else:
        out_file.write('Coannotation \nMannu: NA'+'\nttest: NA'+ '\n')

    module_pval_mannu, module_pval_ttest = compute_pvalues(module_file, icon_model_name,compared_model,standards,
                                        metric='Module Match Score (AMI)', alternative=alternative)
    print('Module detection \n' + str(module_pval_mannu)+'\n' + str(module_pval_ttest)+'\n')
    out_file.write('\nModule detection \nMannu: ' + str(module_pval_mannu)+'\nttest: ' + str(module_pval_ttest)+'\n')

    func_pval_mannu, func_pval_ttest = compute_pvalues(func_file, icon_model_name,compared_model,standards,
                                        metric='Accuracy', alternative=alternative)
    print('Function prediction  \n' + str(func_pval_mannu)+'\n' + str(func_pval_ttest) + '\n')
    out_file.write('\nFunction prediction  \nMannu: ' + str(func_pval_mannu)+'\nttest: ' + str(func_pval_ttest) + '\n')


def wrapper_significance_test_human(icon_model_name, compared_model, standards, alternative, coann_file, module_file, func_file, out_file):

    print(f'\n\n{icon_model_name} vs. {compared_model}')
    out_file.write(f'\n\n\n{icon_model_name} vs. {compared_model}\n')

    if not (compared_model=='BERTWalk_given'):#bertwalk_given has only one run. so no point in significance test for that.
        coann_pval_mannu,coann_pval_ttest  = compute_pvalues(coann_file, icon_model_name, compared_model,standards,
                                            metric='Average Precision', alternative=alternative)
        print('Coannotation \n' + str(coann_pval_mannu)+'\n' + str(coann_pval_ttest)+ '\n')
        out_file.write('Coannotation \nMannu: ' + str(coann_pval_mannu)+'\nttest: ' + str(coann_pval_ttest)+ '\n')
    else:
        out_file.write('Coannotation \nMannu: NA'+'\nttest: NA'+ '\n')


def main():

    #************************************ YEAST ****************************************
    standards = ['IntAct','GO','KEGG']
    results_dir_yeast = '/home/grads/tasnina/Projects/BIONIC-evals/bioniceval/final_results/yeast/'
    # ********************************** ICoN_l3_nomask_noise0.7 **************************************************
    icon_model_name= 'ICON_l3_nomask_n0.7'
    coann_file = results_dir_yeast + 'coannotation/coannotation_coannotation.tsv'
    module_file = results_dir_yeast + 'module_detection/combo_module_detection.tsv'
    func_file = results_dir_yeast + 'function_prediction/function_prediction_function_prediction.tsv'

    alternative = 'greater'
    out_file = open(results_dir_yeast + f'{alternative}_significance_test.txt', "w")

    # #************************* vs. BIONIC ***********************************
    compared_model = 'BIONIC'
    wrapper_significance_test_yeast(icon_model_name, compared_model, standards, alternative, coann_file, module_file, func_file, out_file)

    #*********************** vs. BERTWalk ****************************************
    compared_model = 'BERTWalk_given'
    wrapper_significance_test_yeast(icon_model_name, compared_model, standards, alternative, coann_file, module_file, func_file, out_file)

    out_file.close()



    # ************************************ Human ****************************************
    standards=['CORUM']
    results_dir_human = '/home/grads/tasnina/Projects/BIONIC-evals/bioniceval/final_results/human/'
    # ********************************** ICoN_l3_nomask_noise0.7 **************************************************
    icon_model_name = 'ICON_l3_nomask_n0.7'
    coann_file = results_dir_human + 'coannotation/human_coannotation_coannotation.tsv'

    alternative = 'greater'
    out_file = open(results_dir_human + f'{alternative}_significance_test.txt', "w")

    # #************************* vs. BIONIC ***********************************
    compared_model = 'BIONIC'
    wrapper_significance_test_human(icon_model_name, compared_model, standards,
                                    alternative, coann_file, module_file, func_file,
                                    out_file)

    out_file.close()

    # # #********************************** ICON V2 **************************************************
    # print('\n\n ICONv2 vs BIONIC')
    #
    # # coann_file = results_dir + 'ICON_v2_Pos_Noisy/sep_scale/fig2a_iconv2_posnoisy_coannotation_coannotation.tsv'
    # # coann_pval_mannu,coann_pval_ttest  = compute_pvalues(coann_file, 'ICON_V2_noise0.7', 'BIONIC', metric='Average Precision', alternative='two-sided')
    # # print('Coannotation  \n' + str(coann_pval_mannu)+'\n' + str(coann_pval_ttest)+ '\n')
    # #
    # module_file = results_dir + 'ICON_ICONV2_BERTWalk_BIONIC/fig2a_icon_berwalk_noisy_module_module_detection.tsv'
    # module_pval_mannu, module_pval_ttest = compute_pvalues(module_file, 'ICONv2_posnoise0.7', 'BIONIC',
    #                                                        metric='Module Match Score (AMI)')
    # print('Module detection \n' + str(module_pval_mannu) + '\n' + str(module_pval_ttest) + '\n')
    #
    # # func_file = results_dir + 'ICON_ICONV2_BERTWalk_BIONIC/fig2a_icon_berwalk_noisy_function_function_prediction.tsv'
    # # func_pval_mannu, func_pval_ttest = compute_pvalues(func_file, 'ICONv2_posnoise0.7', 'BIONIC', metric='Accuracy', alternative='two-sided')
    # # print('Function prediction  \n' + str(func_pval_mannu)+'\n' + str(func_pval_ttest) + '\n')
    # #
    # #
    # #
    # # # #************************* vs. BERTWalk ****************************************
    # print('\n\n ICONv2 vs BERTWalk')
    #
    # module_file = results_dir + 'ICON_ICONV2_BERTWalk_BIONIC/fig2a_icon_berwalk_noisy_module_module_detection.tsv'
    # module_pval_mannu, module_pval_ttest = compute_pvalues(module_file, 'ICONv2_posnoise0.7', 'BERTWalk',
    #                                                        metric='Module Match Score (AMI)', alternative='less')
    # print('Module detection \n' + str(module_pval_mannu) + '\n' + str(module_pval_ttest) + '\n')
    #
    # # func_file = results_dir + 'ICON_ICONV2_BERTWalk_BIONIC/fig2a_icon_berwalk_noisy_function_function_prediction.tsv'
    # # func_pval_mannu, func_pval_ttest = compute_pvalues(func_file, 'ICONv2_posnoise0.7', 'BERTWalk', metric='Accuracy',
    # #                                                    alternative='two-sided')
    # # print('Function prediction  \n' + str(func_pval_mannu) + '\n' + str(func_pval_ttest) + '\n')
    #

if __name__=='__main__':
    main()