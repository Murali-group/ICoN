import json
import os
import pathlib
import time
import math
import warnings
from pathlib import Path
from typing import Union, List, Optional
import re
import typer
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.multiprocessing
import matplotlib.pyplot as plt

from utils.config_parser import ConfigParser
from utils.plotter import plot_losses, save_losses
from utils.preprocessor import Preprocessor
from utils.sampler import StatefulSampler, NeighborSamplerWithWeights
from utils.common import extend_path, cyan, magenta, Device
#TODO uncomment later


# from model.model_icon_with_bionic_mode import Icon, IconParallel

from model.loss import masked_scaled_mse, classification_loss
from model.model_utils import  *


class Trainer:
    def __init__(self, config: Union[Path, dict]):
        """Defines the relevant training and forward pass logic for BIONIC.

        A model is trained by calling `train()` and the resulting gene embeddings are
        obtained by calling `forward()`.

        Args:
            config (Union[Path, dict]): Path to config file or dictionary containing config
                parameters.
        """

        typer.secho("Using CUDA", fg=typer.colors.GREEN) if Device() == "cuda" else typer.secho(
            "Using CPU", fg=typer.colors.RED
        )

        self.params = self._parse_config(
            config
        )  # parse configuration and load into `params` namespace
        self.writer = (
            self._init_tensorboard()
        )  # create `SummaryWriter` for tensorboard visualization
        self.wandb_writer = self._init_wandb()

        (
            self.index,
            self.masks,
            self.weights,
            self.adj,
            self.labels,
            self.label_masks,
            self.class_names,
        ) = self._preprocess_inputs()

        #if noise>0, then add some % of noise to the input self.adj. It can be random
        #dropout of edges (false negative) or random addition of edges (false positive)
        #TODO start with false negative i.e., incomplete network
        add_noise = self.params.noise[0]
        drop_noise = self.params.noise[1]

        # if add_noise+drop_noise>0 : #if either add_noise or drop_noise >0 then call noise adding function.
        self.adj_noisy = []
        for i in range(len(self.adj)):
            a = self.adj[i]
            n_edges = a.num_edges
            row, col, _ = a.adj_t.t().coo()
            orig_edges = set(zip(row.tolist(), col.tolist()))

            #randomly drop self.params.noise% of edges from the adjacency matrix.
            a_drop = drop_random_indices(a, drop_noise, n_edges, Device()) if drop_noise>0 else a

            #randomly add self.params.noise% of edges.
            a_add = add_random_indices(a_drop, add_noise, n_edges, Device()) if add_noise>0 else a_drop
            self.adj_noisy.append(a_add)

        #NURE: sanity check: find how much edges are common between noisy and non-noisy networks
        for i in range(len(self.adj)):
            a = self.adj[i]
            row, col, _ = a.adj_t.t().coo()
            orig_edges = set(zip(row.tolist(), col.tolist()))

            a_noisy = self.adj_noisy[i]
            row_ns, col_ns, _ = a_noisy.adj_t.t().coo()
            noisy_edges = set(zip(row_ns.tolist(), col_ns.tolist()))

            print( 'frac common edges: ',(len(orig_edges.intersection(noisy_edges))-5232)/(len(orig_edges)-5232))


        self.train_loaders = self._make_train_loaders()
        self.inference_loaders = self._make_inference_loaders()
        self.model, self.optimizer = self._init_model()

    def _parse_config(self, config):
        cp = ConfigParser(config)
        return cp.parse()

    def _init_tensorboard(self):
        if self.params.tensorboard and (
            self.params.tensorboard["training"] or self.params.tensorboard["embedding"]
        ):
            from torch.utils.tensorboard import SummaryWriter

            typer.secho("Using TensorBoard logging", fg=typer.colors.GREEN)
            log_dir = (
                None
                if "log_dir" not in self.params.tensorboard
                else self.params.tensorboard["log_dir"]
            )
            comment = (
                ""
                if "comment" not in self.params.tensorboard
                else self.params.tensorboard["comment"]
            )
            return SummaryWriter(log_dir=log_dir, comment=comment, flush_secs=10,)
        return None

    def _init_wandb(self):
        if self.params.wandb and (self.params.wandb["training"] or self.params.wandb["validation"]):
            import wandb
            wandb.login()
            wandb_writer = wandb.init(config=self.params, project=self.params.wandb['project'])
            return wandb_writer
        return None

    def _preprocess_inputs(self):
        preprocessor = Preprocessor(
            self.params.net_names,
            label_names=self.params.label_names,
            delimiter=self.params.delimiter,
            svd_dim=self.params.svd_dim,
        )
        return preprocessor.process()

    def _make_train_loaders(self):
        return [
            NeighborSamplerWithWeights(
                ad,
                # sizes=[self.params.neighbor_sample_size] * self.params.gat_shapes["n_layers"],
                #One layer batching at a time.
                sizes=[self.params.neighbor_sample_size],
                batch_size=self.params.batch_size,
                sampler=StatefulSampler(torch.arange(len(self.index))),
                shuffle=False,
            )
            for ad in self.adj_noisy
        ]

    def _make_inference_loaders(self):
        return [
            NeighborSamplerWithWeights(
                ad,
                # sizes=[-1] * self.params.gat_shapes["n_layers"],  # all neighbors
                # One layer batching at a time.
                sizes=[-1], # all neighbors
                batch_size=1,
                sampler=StatefulSampler(torch.arange(len(self.index))),
                shuffle=False,
            )
            for ad in self.adj
        ]

    def _init_model(self):
        if self.labels:
            n_classes = [label.shape[1] for label in self.labels]
        else:
            n_classes = None

        if self.params.model =="ICON":
            from model.model_icon import Icon
            Model = Icon



        #If norm_adj == True in config, nomalize adjacency matrix before using as feature.
        self.feat = []

        model = Model(
            len(self.index),
            self.params.pre_gat_type,
            self.params.gat_type,
            self.params.gat_shapes,
            self.params.embedding_size,
            self.params.residual,
            len(self.adj),
            self.params.bmask,
            svd_dim=self.params.svd_dim,
            shared_encoder=self.params.shared_encoder,
            n_classes=n_classes,
            feats = self.feat if self.params.feat_type=='adj' else None,
            init_mode=self.params.init_mode,
            agg = self.params.agg,
            scale=self.params.scale,
            con=self.params.con
        )

        model.apply(self._init_model_weights)
        # Load pretrained model
        if self.params.pretrained_model_path:
            typer.echo("Loading pretrained model...")
            missing_keys, unexpected_keys = model.load_state_dict(
                torch.load(self.params.pretrained_model_path), strict=False)
            if missing_keys:
                warnings.warn(
                    "The following parameters were missing from the provided pretrained model:"
                    f"{missing_keys}")
            if unexpected_keys:
                warnings.warn(
                    "The following unexpected parameters were provided in the pretrained model:"
                    f"{unexpected_keys}")

        # Push model to device
        if not self.params.model_parallel:
            model.to(Device())

        optimizer = optim.Adam(model.parameters(), lr=self.params.learning_rate, weight_decay=self.params.gat_shapes["decay"])

        return model, optimizer

    def _init_model_weights(self, model):
        if hasattr(model, "weight"):
            if self.params.initialization == "kaiming":
                torch.nn.init.kaiming_uniform_(model.weight, a=0.1)
            elif self.params.initialization == "xavier":
                torch.nn.init.xavier_uniform_(model.weight)
            else:
                raise ValueError(
                    f"The initialization scheme {self.params.initialization} \
                    provided is not supported"
                )

    def train(self,verbosity: Optional[int] = 1):
        """Trains ICON model.
        Args:
            verbosity (int): 0 to supress printing (except for progress bar), 1 for regular printing.
        """
        # Track losses per epoch.
        train_loss = []
        best_loss = None
        best_state = None
        # model, log = 'all', log_freq = 1
        if self.wandb_writer and  self.params.wandb["training"]:
            self.wandb_writer.watch(self.model, log = self.params.wandb['log'],
                                    log_freq=self.params.wandb['log_freq'] )

        # Train model.
        for epoch in range(self.params.epochs):
            time_start = time.time()

            # Track average loss across batches
            if self.labels is not None:
                epoch_losses = np.zeros(len(self.adj) + len(self.labels))
            else:
                epoch_losses = np.zeros(len(self.adj))

            if bool(self.params.sample_size):
                rand_net_idxs = np.random.permutation(len(self.adj))
                idx_split = np.array_split(
                    rand_net_idxs, math.floor(len(self.adj) / self.params.sample_size)
                )
                for rand_idxs in idx_split:
                    _, losses, sum_diff, diffs, norm_diffs, sum_norm_diff = self._train_step(rand_idxs)
                    for idx, loss in zip(rand_idxs, losses):
                        epoch_losses[idx] += loss

                    # Add classification losses if applicable
                    for idx, loss in enumerate(losses[len(rand_idxs) :]):
                        epoch_losses[len(rand_idxs) + idx] = loss

            else:
                t1=time.time()
                _, losses, sum_diff, diffs, norm_diffs, sum_norm_diff, agg_attn_mats = self._train_step()
                # print('timestep: ',time.time()-t1 )

                epoch_losses = [
                    ep_loss + b_loss.item() / (len(self.index) / self.params.batch_size)
                    for ep_loss, b_loss in zip(epoch_losses, losses)
                ]

            if verbosity:
                progress_string = self._create_progress_string(epoch, epoch_losses, time_start)
                typer.echo(progress_string)

            #NURE: plot heatmap for aggregate attention at each 250th epoch
            # if (epoch%25)==0:
            #     plot_agg_att_mat(agg_attn_mats, epoch, self.params.noise[0])

            # Add loss data to tensorboard visualization
            if self.writer and self.params.tensorboard["training"]:
                recon_loss_dct = {}
                cls_loss_dct = {}
                project_name = self.writer.log_dir.split('/')[-1]
                for i, loss in enumerate(epoch_losses):
                    if self.labels is not None and i >= len(self.adj):
                        name = self.params.label_names[i - len(self.adj)].stem
                        cls_loss_dct[name] = loss
                    else:
                        name = self.params.net_names[i].stem
                        recon_loss_dct[name] = loss

                recon_loss_dct["Total"] = sum(recon_loss_dct.values())
                cls_loss_dct["Total"] = sum(cls_loss_dct.values())

                self.writer.add_scalars(f"{project_name}/Reconstruction Loss", recon_loss_dct, epoch)
                if cls_loss_dct:
                    self.writer.add_scalars(f"{project_name}/Classification Loss", cls_loss_dct, epoch)

            if self.wandb_writer and self.params.wandb["training"]:
                recon_loss_dct = {}
                cls_loss_dct = {}
                for i, loss in enumerate(epoch_losses):
                    if self.labels is not None and i >= len(self.adj):
                        name = self.params.label_names[i - len(self.adj)].stem
                        cls_loss_dct[name] = loss
                    else:
                        name = self.params.net_names[i].stem
                        recon_loss_dct[name] = loss

                recon_loss_dct["Total"] = sum(recon_loss_dct.values())
                cls_loss_dct["Total"] = sum(cls_loss_dct.values())

                self.wandb_writer.log({'Reconstruction Loss': recon_loss_dct}, step=epoch)
                self.wandb_writer.log({'Total Feature Distance': sum_diff}, step=epoch)
                self.wandb_writer.log({'Total Normalized Feature Distance': sum_norm_diff}, step=epoch)
                self.wandb_writer.log({'Pairwise Feature Distance': diffs}, step=epoch)
                self.wandb_writer.log({'Normalized Pairwise Feature Distance': norm_diffs}, step=epoch)


                if cls_loss_dct:
                    self.wandb_writer.log({'Classification Loss': cls_loss_dct}, step=epoch)



            train_loss.append(epoch_losses)

            # Store best parameter set
            if not best_loss or sum(epoch_losses) < best_loss:
                best_loss = sum(epoch_losses)
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_loss": best_loss,
                }
                best_state = state

        if self.writer:
            self.writer.close()
        if self.wandb_writer:
            self.wandb_writer.unwatch()
            self.wandb_writer.finish()

        self.train_loss, self.best_state = train_loss, best_state

    def _train_step(self, rand_net_idx=None):
        """Defines training behaviour."""
        # Get random integers for batch.
        rand_int = StatefulSampler.step(len(self.index))
        int_splits = torch.split(rand_int, self.params.batch_size)

        # Initialize loaders to current batch.
        if bool(self.params.sample_size):
            batch_loaders = [self.train_loaders[i] for i in rand_net_idx]

            # Subset `masks` tensor.
            mask_splits = torch.split(self.masks[:, rand_net_idx][rand_int], self.params.batch_size)

        else:
            batch_loaders = self.train_loaders
            mask_splits = torch.split(self.masks[rand_int], self.params.batch_size)

        # List of losses.
        if self.labels is not None:
            losses = [0.0 for _ in range(len(batch_loaders) + len(self.labels))]
        else:
            losses = [0.0 for _ in range(len(batch_loaders))]

        for batch_masks, node_ids in zip(mask_splits, int_splits):
            #NURE: incorporate batches to accommodate independent GAT layers
            data_flows = batch_sampling(batch_loaders, node_ids, self.params.gat_shapes["n_layers"] +
                                        self.params.gat_shapes["free_layers"] )

            t2=time.time()
            # print('batching: ', t2-t1)
            # print('Done batching')
            #************************
            self.optimizer.zero_grad()
            # Subset supervised labels and masks if provided
            if self.labels is not None:
                batch_labels = [labels[node_ids, :] for labels in self.labels]
                batch_labels_masks = [label_masks[node_ids] for label_masks in self.label_masks]

            if bool(self.params.sample_size):
                output, _, _, _, label_preds, sum_diff, diffs, norm_diffs, sum_norm_diff = self.model(
                    data_flows, batch_masks, rand_net_idxs=rand_net_idx,)
                recon_losses = [
                    masked_scaled_mse(
                        output,
                        self.adj[i],
                        self.weights[i],
                        node_ids,
                        batch_masks[:, j],
                        self.params.lambda_,
                        device="cuda:0" if self.params.model_parallel else None,
                    )
                    for j, i in enumerate(rand_net_idx)
                ]
            else: #ICON runs this
                output, _, net_spec_outputs, _, label_preds, sum_diff, diffs, norm_diffs, sum_norm_diff, agg_attn_mats = self.model(data_flows, batch_masks)
                recon_losses = [
                    masked_scaled_mse(
                        output,
                        self.adj[i],
                        self.weights[i],
                        node_ids,
                        batch_masks[:, i],
                        self.params.lambda_,
                        device="cuda:0" if self.params.model_parallel else None,
                        loss_type = self.params.loss_type,
                        bmask = self.params.bmask,
                        spec_recons = net_spec_outputs[i] if net_spec_outputs is not None else None
                    )
                    for i in range(len(self.adj))
                ]

            if label_preds is not None:
                cls_losses = [
                    classification_loss(pred, label, label_mask, self.params.lambda_)
                    for pred, label, label_mask in zip(
                        label_preds, batch_labels, batch_labels_masks
                    )
                ]
                curr_losses = recon_losses + cls_losses
                losses = [loss + curr_loss for loss, curr_loss in zip(losses, curr_losses)]
                loss_sum = sum(curr_losses)
            else:
                losses = [loss + curr_loss for loss, curr_loss in zip(losses, recon_losses)]
                loss_sum = sum(recon_losses)

            loss_sum.backward()
            self.optimizer.step()

        return output, losses, sum_diff, diffs, norm_diffs, sum_norm_diff, agg_attn_mats

    def _create_progress_string(
        self, epoch: int, epoch_losses: List[float], time_start: float
    ) -> str:
        """Creates a training progress string to display."""
        sep = magenta("|")

        progress_string = (
            f"{cyan('Epoch')}: {epoch + 1} {sep} "
            f"{cyan('Loss Total')}: {sum(epoch_losses):.6f} {sep} "
        )
        if len(self.adj) <= 10:
            for i, loss in enumerate(epoch_losses):
                if self.labels is not None and i >= len(self.adj):
                    progress_string += (
                        f"{cyan(f'ClsLoss {i + 1 - len(self.adj)}')}: {loss:.6f} {sep} "
                    )
                else:
                    progress_string += f"{cyan(f'Loss {i + 1}')}: {loss:.6f} {sep} "
        progress_string += f"{cyan('Time (s)')}: {time.time() - time_start:.4f}"
        return progress_string

    def create_param_str(self):

        '''
        This function return a string which can be used to save files with the
        parameter names the model had run on.
        While using the return string: consider out_name as outer dir and
        input net_names as inner directory,
        and the rest of the parameters as file name prefix.
        '''

        #network names
        net_names_str=''
        for net_name in self.params.net_names:
            net_names_str+= re.split(r'[./]', str(net_name))[-2]+'-'
        net_names_str = net_names_str[:-1] #remove the last '-'
        #GAT related paramateres
        gat_str = 'gat'+\
                '-'+str(self.params.gat_shapes['dimension'])+\
                '-'+str(self.params.gat_shapes['n_heads'])+\
                '-' + str(self.params.gat_shapes['n_layers'])+ \
                '-'+str(self.params.gat_shapes['dropout'])+\
                '-'+str(self.params.gat_shapes['decay'])+ \
                '-' + str(self.params.gat_shapes['free_layers'])+\
                '-'+self.params.gat_type +\
                '-'+self.params.scale
        gat_str =  gat_str + (f'-{self.params.con}' if not (self.params.con) else '')

        # Nure: Whenever a new parameter is considered, add it here.
        # Rest of the parameters
        other_tracked_parameters = {'emb': str(self.params.embedding_size),
                'e': str(self.params.epochs),
                'lr': str(self.params.learning_rate).replace('.','-'),
                'nbr': str(self.params.neighbor_sample_size),
                'pgat': self.params.pre_gat_type, 'res':str(self.params.residual),
                'ls':self.params.loss_type,
                'ft': self.params.feat_type+('-0' if not self.params.bmask else '')
                                    +('-norm' if self.params.norm_adj else ''),
                'init': self.params.init_mode,
                'agg': str(self.params.agg),
                'noise': str(self.params.noise)}

        other_param_str = ''
        for key in other_tracked_parameters:
            other_param_str+= '_'+key+'-'+other_tracked_parameters[key]

        #For each combination of input networks, create a folder.
        file_prefix = str(self.params.out_name)+ '/'+ net_names_str+'/' +\
                      gat_str + other_param_str

        return  file_prefix

    def forward(self, run_no, verbosity: int = 1):
        """Runs the forward pass on the trained BIONIC model.
        Args:

            verbosity (int): 0 to supress printing (except for progress bar), 1 for regular printing.
        """

        # FILE SAVE
        out_dir = self.create_param_str()
        file_prefix = out_dir+'/run_'+str(run_no)
        os.makedirs(os.path.dirname(file_prefix), exist_ok=True)
        file_prefix = Path(file_prefix)

        # Begin inference
        if self.labels is None:
            # if model is in train mode than load from just computed self.best_state else
            # load from a corresponding saved model.
            if self.params.mode == 'train':
                self.model.load_state_dict(self.best_state["state_dict"])
                if verbosity:
                    typer.echo(
                        (
                            f"""Loaded best model from epoch {magenta(f"{self.best_state['epoch']}")} """
                            f"""with loss {magenta(f"{self.best_state['best_loss']:.6f}")}"""
                        )
                    )
                #save best loss
                loss_file = extend_path(file_prefix, "_best_loss.txt")
                with open(loss_file, 'w') as f:
                    f.write(f"{self.best_state['best_loss']:.6f}")
                    f.close()

                #Write the computed network weights
                net_scale_file = extend_path(file_prefix, "_net_scales.txt")
                with open(net_scale_file, 'w') as f:
                    net_names = [str(net_name).split('/')[-1] for net_name in self.params.net_names]
                    ordered_net_str =' '.join(net_names) + '\n'
                    all_nets_str = ''
                    for net_idx in range(len(self.adj)):
                        cur_net_str = net_names[net_idx]+'\n'
                        all_layers_str = ''
                        for k in range(self.params.gat_shapes['n_layers']):
                            co_encoder_name = "Co_Encoder_"+f"{net_idx}_{k}"
                            layer_str = 'layer_' + str(k) + '\t'
                            if hasattr(self.model, co_encoder_name):
                                co_encoder = getattr(self.model, co_encoder_name)
                                net_scale = co_encoder.net_weights.net_scales
                                net_scale_str = str(net_scale.to('cpu').detach().numpy().flatten().tolist()) + '\n'
                                all_layers_str = all_layers_str + layer_str + net_scale_str
                        all_nets_str = all_nets_str + cur_net_str + all_layers_str

                    f.write(ordered_net_str + all_nets_str)
                    f.close()

            else:
                model_path = extend_path(file_prefix, "_model.pt")
                if model_path.exists():
                    torch.load(model_path)
                else:
                    print('Error: No corresponding model found! at: ',str(model_path) )
             # Recover model with lowest reconstruction loss if no classification objective


        self.model.eval()
        StatefulSampler.step(len(self.index), random=False)
        emb_list = []
        prediction_lists = [[] for _ in self.labels] if self.labels is not None else None

        #**************** ICoN Version *******************
        count=0
        for mask in self.masks:
            data_flows = batch_sampling(self.inference_loaders,[count],
                        self.params.gat_shapes["n_layers"]+ self.params.gat_shapes["free_layers"])
            mask = mask.reshape((1, -1))
            _, emb, _, learned_scales, label_preds, sum_diff, diff, _, sum_norm_diff,_ = self.model(data_flows, mask, evaluate=True)
            emb_list.append(emb.detach().cpu().numpy())

            if label_preds is not None:
                for i, pred in enumerate(label_preds):
                    prediction_lists[i].append(torch.sigmoid(pred).detach().cpu().numpy())
            count+=1

        emb = np.concatenate(emb_list)
        emb_df = pd.DataFrame(emb, index=self.index)



        emb_df.to_csv(extend_path(file_prefix, "_features.tsv"), sep="\t")

        # Free memory (necessary for sequential runs)
        if Device() == "cuda":
            torch.cuda.empty_cache()

        if self.params.mode=='train':
            # Output loss plot
            if self.params.plot_loss:
                if verbosity:
                    typer.echo("Plotting loss...")
                plot_losses(
                    self.train_loss,
                    self.params.net_names,
                    extend_path(file_prefix, "_loss.png"),
                    self.params.label_names,
                )

            # Save losses per epoch
            if self.params.save_loss_data:
                if verbosity:
                    typer.echo("Saving loss data...")
                save_losses(
                    self.train_loss,
                    self.params.net_names,
                    extend_path(file_prefix, "_loss.tsv"),
                    self.params.label_names,
                )

            # Save model
            if self.params.save_model:
                if verbosity:
                    typer.echo("Saving model...")
                torch.save(self.model.state_dict(), extend_path(file_prefix, "_model.pt"))


        # Save internal learned network scales
        if self.params.save_network_scales:
            if verbosity:
                typer.echo("Saving network scales...")
            learned_scales = pd.DataFrame(
                learned_scales.detach().cpu().numpy(), columns=self.params.net_names
            ).T
            learned_scales.to_csv(
                extend_path(file_prefix, "_network_weights.tsv"), header=False, sep="\t"
            )

        # Save label predictions
        if self.params.save_label_predictions:

            if verbosity:
                typer.echo("Saving predicted labels...")
            if self.params.label_names is None:
                warnings.warn(
                    "The `label_names` parameter was not provided so there are "
                    "no predicted labels to save."
                )
            else:
                for i, (pred, class_names) in enumerate(zip(prediction_lists, self.class_names)):
                    pred = np.concatenate(pred)
                    pred = pd.DataFrame(pred, index=self.index, columns=class_names)
                    pred.to_csv(
                        extend_path(file_prefix, f"_label_set_{i+1}_predictions.tsv"),
                        sep="\t",
                    )

        typer.echo(magenta("Complete!"))
