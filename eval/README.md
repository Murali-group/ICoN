# Evaluate ICoN
To evaluate ICoN in terms of downstream tasks: i. gene module detection, ii.gene coannotation prediction, and  iii. gene function prediction we utilized [BIONIC-evals](https://github.com/duncster94/BIONIC-evals).

We have provided the [datasets](https://github.com/Murali-group/ICoN/tree/main/eval/datasets/) (embeddings generated from 1 run of models), [standards](https://github.com/Murali-group/ICoN/tree/main/eval/standards), [config](https://github.com/Murali-group/ICoN/tree/main/eval/config/), and [script](https://github.com/Murali-group/ICoN/tree/main/eval/script) used in creating figures in the manuscript of ICoN. To reproduce these figures (for one run) follow the instructions below:
1. First install BIONIC-evals following the instructions given in [BIONIC-evals](https://github.com/duncster94/BIONIC-evals)
2. Place our provided <script> folder inside <BIONIC-evals/bioniceval>. 
3. Now replace the following folders in <BIONIC-evals/bioniceval> with our provided folders [here](https://github.com/Murali-group/ICoN/tree/main/eval):
   i. datasets
   ii. config
   iii. standards
   
   **Note**: We have provided some files in .zip format. Please extract them before proceeding.

## i. Comparative analysis between ICoN and other network integration models (and input networks):
1. Run BIONIC-evals with <config/single_runs/yeast.json>
2. Then run:
   ```
   python paper_plots.py <bionic_eval_results_folder>
   ```
## ii. Ablation study of ICoN:
### Co-attention
1. Run BIONIC-evals with <config/single_runs/ablation_nocoattn.json>
2. Then run:
   ```
   python ablation_study_coattn.py <bionic_eval_results_folder>
   ```
### Noise induction module
1. Run BIONIC-evals with <config/single_runs/ablation_nonoise.json>
2. Then run:
   ```
   python ablation_study_noise.py <bionic_eval_results_folder>
   ```

## iii. Co-attention coeffcient:
Run:
   ```
   python co_attention_weights-lineplot.py <bionic_eval_datasets_folder>
   ```

## iv. Robustness to noise:
1. Run BIONIC-evals with <config/single_runs/noisyinput_icon_bionic_union.json>
2. Then run:
   ```
   python noise_robustness.py <bionic_eval_results_folder>
   ```
