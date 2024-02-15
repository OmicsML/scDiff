"""
Wild mixture of:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8

Thank you!
"""
import warnings
from contextlib import contextmanager
from functools import partial

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops.layers.torch import Rearrange
from scipy.sparse import csr_matrix
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from scdiff.modules.diffusion_model import Decoder, Embedder, Encoder
from scdiff.evaluate import (
    denoising_eval,
    evaluate_annotation,
    perturbation_eval,
    calculate_batch_r_squared,
)
from scdiff.modules.ema import LitEma
from scdiff.modules.layers.attention import BasicTransformerBlock
from scdiff.modules.layers.basic import FeedForward
from scdiff.modules.layers.scmodel import EmbeddingDict
from scdiff.utils.diffusion import MaskedEncoderConditioner, timestep_embedding
from scdiff.utils.diffusion import make_beta_schedule
from scdiff.utils.misc import as_1d_vec, exists, count_params, instantiate_from_config
from scdiff.utils.misc import default
from scdiff.utils.modules import create_activation, create_norm
from scdiff.utils.modules import extract_into_tensor, init_weights, mean_flat, noise_like


RESCALE_FACTOR = np.log(1e4)


class DiffusionModel(nn.Module):
    def __init__(self, pretrained_gene_list, input_gene_list=None, dropout=0., cell_mask_ratio=0.75, mask_context=True,
                 encoder_type='stackffn', embed_dim=1024, depth=4, dim_head=64, num_heads=4,
                 feat_mask_ratio=0., decoder_embed_dim=512, decoder_embed_type='linear', decoder_num_heads=4,
                 decoder_dim_head=64, cond_dim=None, cond_tokens=1, cond_type='crossattn', cond_strategy='full_mix',
                 cond_emb_type='linear', cond_num_dict=None, cond_mask_ratio=0.5, cond_cat_input=False,
                 post_cond_num_dict=None, post_cond_layers=2, post_cond_norm='layernorm',
                 post_cond_mask_ratio=0.0, norm_layer='layernorm', mlp_time_embed=False, no_time_embed=False,
                 activation='gelu', mask_strategy='random', mask_mode='v1', mask_dec_cond=False,
                 mask_dec_cond_ratio=False, mask_dec_cond_se=False, mask_dec_cond_semlp=False,
                 mask_dec_cond_concat=False, mask_value=0, pad_value=0, decoder_mask=None, text_emb=None,
                 text_emb_file=None, freeze_text_emb=True, text_proj_type='linear', text_proj_act=None,
                 stackfnn_glu_flag=False, text_proj_hidden_dim=512, text_proj_num_layers=2, text_proj_norm=None,
                 cond_emb_norm=None, num_perts=None, gears_flag=False, gears_hidden_size=64,
                 gears_mode="single", gears_mlp_layers=2, gears_norm=None, num_go_gnn_layers=1):
        super().__init__()
        self.depth = depth

        # --------------------------------------------------------------------------
        # MAE masking options
        self.cell_mask_ratio = cell_mask_ratio
        self.feat_mask_ratio = feat_mask_ratio
        self.mask_context = mask_context
        self.mask_mode = mask_mode
        self.mask_strategy = mask_strategy
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.decoder_mask = decoder_mask
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        activation = create_activation(activation)
        # self.in_dim = len(input_gene_list) if input_gene_list is not None else len(pretrained_gene_list)
        self.in_dim = len(pretrained_gene_list) if pretrained_gene_list is not None else len(input_gene_list)
        self.pretrained_gene_list = pretrained_gene_list
        self.input_gene_list = input_gene_list
        pretrained_gene_index = dict(zip(self.pretrained_gene_list, list(range(len(self.pretrained_gene_list)))))
        self.input_gene_idx = torch.tensor([
            pretrained_gene_index[o] for o in self.input_gene_list
            if o in pretrained_gene_index
        ]).long() if self.input_gene_list is not None else None

        assert embed_dim == decoder_embed_dim  # XXX: this seems to be required for MAE (see forward dec)?
        full_embed_dim = embed_dim * cond_tokens
        self.post_encoder_layer = Rearrange('b (n d) -> b n d', n=cond_tokens, d=embed_dim)

        self.embedder = Embedder(pretrained_gene_list, full_embed_dim, 'layernorm', dropout=dropout)

        self.encoder_type = encoder_type
        if encoder_type == 'attn':
            self.blocks = nn.ModuleList([
                BasicTransformerBlock(full_embed_dim, num_heads, dim_head, self_attn=True, cross_attn=False,
                                      dropout=dropout, qkv_bias=True, final_act=activation)
                for _ in range(depth)])
        elif encoder_type in ('mlp', 'mlpparallel'):
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(full_embed_dim, full_embed_dim),
                    activation,
                    create_norm(norm_layer, full_embed_dim),
                ) for _ in range(depth)])
        elif encoder_type in ('stackffn', 'ffnparallel'):
            self.blocks = nn.ModuleList([
                # FeedForward(full_embed_dim, mult=4, glu=False, dropout=dropout)
                nn.Sequential(
                    FeedForward(full_embed_dim, mult=4, glu=False, dropout=dropout),
                    create_norm(norm_layer, full_embed_dim),
                ) for _ in range(depth)])
        elif encoder_type == 'none':
            self.blocks = None
        else:
            raise ValueError(f'Unknown encoder type {encoder_type}')
        # self.encoder_proj = nn.Linear(full_embed_dim, latent_dim)
        # self.norm = create_norm(norm_layer, full_embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.subset_output = True
        self.decoder_embed_dim = decoder_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(decoder_embed_dim, 4 * decoder_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * decoder_embed_dim, decoder_embed_dim),
        ) if mlp_time_embed else nn.Identity()
        self.no_time_embed = no_time_embed

        self.cond_type = cond_type
        assert cond_strategy in ("full_mix", "pre_mix")
        self.cond_strategy = cond_strategy
        self.cond_emb_type = cond_emb_type
        self.cond_tokens = cond_tokens
        self.cond_cat_input = cond_cat_input
        if cond_dim is not None or cond_num_dict is not None:
            if cond_emb_type == 'linear':
                assert cond_dim is not None
                self.cond_embed = nn.Sequential(
                    nn.Linear(cond_dim, decoder_embed_dim * cond_tokens),
                    Rearrange('b (n d) -> b n d', n=cond_tokens, d=decoder_embed_dim),
                )
            elif cond_emb_type == 'embedding':
                assert cond_num_dict is not None
                self.cond_embed = EmbeddingDict(cond_num_dict, decoder_embed_dim, depth,
                                                cond_tokens, mask_ratio=cond_mask_ratio,
                                                text_emb=text_emb, text_emb_file=text_emb_file,
                                                norm_layer=cond_emb_norm,
                                                freeze_text_emb=freeze_text_emb,
                                                text_proj_type=text_proj_type,
                                                text_proj_num_layers=text_proj_num_layers,
                                                stackfnn_glu_flag=stackfnn_glu_flag,
                                                text_proj_hidden_dim=text_proj_hidden_dim,
                                                text_proj_act=text_proj_act,
                                                text_proj_norm=text_proj_norm,
                                                # text_proj_dropout=dropout, G_go=G_go,
                                                # G_go_weight=G_go_weight, num_perts=num_perts,
                                                text_proj_dropout=dropout, gears_flag=gears_flag, num_perts=num_perts,
                                                gears_hidden_size=gears_hidden_size, gears_mode=gears_mode,
                                                gears_mlp_layers=gears_mlp_layers, gears_norm=gears_norm,
                                                num_go_gnn_layers=num_go_gnn_layers)
            elif cond_emb_type == 'none':
                self.cond_embed = None
            else:
                raise ValueError(f"Unknwon condition embedder type {cond_emb_type}")
        else:
            self.cond_embed = None

        self.encoder = Encoder(depth, decoder_embed_dim, decoder_num_heads, decoder_dim_head,
                               dropout=dropout, cond_type=cond_type, cond_cat_input=cond_cat_input)

        # self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim))
        self.decoder_embed_type = decoder_embed_type
        assert decoder_embed_type in ['linear', 'embedder', 'encoder']
        if decoder_embed_type == 'linear':
            self.decoder_embed = nn.Linear(self.in_dim, decoder_embed_dim)
        elif decoder_embed_type == 'embedder':
            self.decoder_embed = Embedder(pretrained_gene_list, decoder_embed_dim, 'layernorm', dropout=dropout)
        elif decoder_embed_type == 'encoder':
            self.decoder_embed = self.embedder

        self.mask_decoder_conditioner = MaskedEncoderConditioner(
            decoder_embed_dim, mult=4, use_ratio=mask_dec_cond_ratio, use_se=mask_dec_cond_se,
            use_semlp=mask_dec_cond_semlp, concat=mask_dec_cond_concat, disable=not mask_dec_cond)

        self.decoder_norm = create_norm(norm_layer, decoder_embed_dim)
        self.decoder = Decoder(decoder_embed_dim, self.in_dim, dropout, post_cond_norm,
                               post_cond_layers, post_cond_num_dict, act=activation,
                               cond_emb_dim=decoder_embed_dim, cond_mask_ratio=post_cond_mask_ratio)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize linear and normalization layers
        self.apply(init_weights)

    # TODO: move to DDPM and get mask from there (masking is indepdent on forward)?
    def random_masking(self, x):
        # mask: 0 keep, 1 drop
        cell_mask_ratio = self.cell_mask_ratio
        feat_mask_ratio = self.feat_mask_ratio
        N, D = x.shape  # batch, dim

        if self.mask_mode == "v1":
            x_masked = x.clone()

            # apply cell masking
            len_keep = int(N * (1 - cell_mask_ratio))
            perm = np.random.permutation(N)
            idx_keep = perm[:len_keep]

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, D], device=x.device)
            mask[idx_keep] = 0

            # apply feature masking on the remaining part
            if feat_mask_ratio > 0:
                if self.mask_strategy == 'random':
                    feat_mask = mask[idx_keep]
                    feat_mask[torch.rand(len_keep, D) <= feat_mask_ratio] = 1
                    mask[idx_keep] = feat_mask
                elif self.mask_strategy == 'none_pad':
                    for i in idx_keep:
                        row = x_masked[i]
                        non_padding_idx = torch.nonzero(row - self.pad_value)[0]
                        n_mask = int(len(non_padding_idx) * feat_mask_ratio)
                        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
                        mask[i][mask_idx] = 1
                else:
                    raise NotImplementedError(f'Unsupported mask strategy: {self.mask_strategy}')

            x_masked[mask.bool()] = self.mask_value
        elif self.mask_mode == "v2":
            if feat_mask_ratio != 0:
                warnings.warn(
                    "v2 mask disregards feat_mask_ratio, which is currently "
                    f"set to {feat_mask_ratio!r}.",
                    UserWarning,
                    stacklevel=2,
                )
            mask_ratios = torch.rand(N, 1, device=x.device)
            mask_ratios[torch.rand(N) < self.cell_mask_ratio] = 1
            mask = torch.rand_like(x) < mask_ratios

            x_masked = torch.zeros_like(x).masked_scatter(~mask, x)

        return x_masked, mask

    def forward_encoder(self, x, pe_input=None, input_gene_list=None, input_gene_idx=None):
        # embed input
        input_gene_list = default(input_gene_list, self.input_gene_list)
        input_gene_idx = default(input_gene_idx, self.input_gene_idx)
        x, gene_idx = self.embedder(x, pe_input, input_gene_list, input_gene_idx)

        if self.blocks is None:
            hist = [None] * self.depth
        elif self.encoder_type in ("mlpparallel", "ffnparallel"):
            hist = [self.post_encoder_layer(blk(x)) for blk in self.blocks]
        else:
            hist = []
            for blk in self.blocks:  # apply context encoder blocks
                x = blk(x)
                hist.append(self.post_encoder_layer(x))

        return hist, gene_idx

    def forward_decoder(self, x, context_list, timesteps=None, pe_input=None, conditions=None,
                        input_gene_list=None, input_gene_idx=None, aug_graph=None,
                        return_latent=False, mask=None):
        # embed tokens
        if self.decoder_embed_type == 'linear':
            x = self.decoder_embed(x)
        else:
            input_gene_list = default(input_gene_list, self.input_gene_list)
            input_gene_idx = default(input_gene_idx, self.input_gene_idx)
            x, _ = self.decoder_embed(x, pe_input, input_gene_list, input_gene_idx)

        # apply masked conditioner
        x = self.mask_decoder_conditioner(x, mask)

        # calculate time embedding
        if timesteps is not None and not self.no_time_embed:
            timesteps = timesteps.repeat(x.shape[0]) if len(timesteps) == 1 else timesteps
            time_embed = self.time_embed(timestep_embedding(timesteps, self.decoder_embed_dim))
            x = x + time_embed
            # x = torch.cat([x, time_embed], dim=0)

        # calculate cell condition embedding
        cond_emb_list = None if self.cond_embed is None else self.cond_embed(conditions, aug_graph=aug_graph)
        if not isinstance(cond_emb_list, list):
            cond_emb_list = [cond_emb_list] * self.depth

        x = self.encoder(x, context_list, cond_emb_list)

        # apply post conditioner layers
        x = self.decoder_norm(x)
        return x if return_latent else self.decoder(x, conditions)

    def forward_loss(self, target, pred, mask=None):
        if mask is None:
            mask = torch.ones(target.shape, device=target.device)
        loss = (pred - target) ** 2
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed tokens
        return loss

    def get_latent(self, x_orig, x, timesteps=None, pe_input=None, conditions=None,
                   input_gene_list=None, text_embeddings=None, aug_graph=None, mask=None):
        # embed input
        context_list, _ = self.forward_encoder(x_orig, pe_input, input_gene_list)
        latent = self.forward_decoder(x, context_list, timesteps, pe_input, conditions, input_gene_list,
                                      text_embeddings, aug_graph=aug_graph, return_latent=True, mask=mask)
        return latent

    def forward(self, x_orig, x, timesteps=None, pe_input=None, conditions=None, input_gene_list=None,
                input_gene_idx=None, target_gene_list=None, aug_graph=None, mask=True):
        # masking: length -> length * mask_ratio
        if isinstance(mask, torch.Tensor):
            x_orig_masked = x_orig * ~mask.bool()

        elif isinstance(mask, bool):
            if mask:
                x_orig_masked, mask = self.random_masking(x_orig)
                if self.decoder_mask is not None:
                    if self.decoder_mask == 'enc':
                        x[mask.bool()] = self.mask_value
                    elif self.decoder_mask == 'inv_enc':
                        x[~mask.bool()] = self.mask_value
                        # mask = torch.ones_like(x_orig)
                    elif self.decoder_mask == 'dec':
                        _, dec_mask, _, _ = self.random_masking(x)
                        x[dec_mask.bool()] = self.mask_value
                        mask = (mask.bool() | dec_mask.bool()).float()
                    else:
                        raise NotImplementedError(f"Unsuppoted decoder mask choice: {self.decoder_mask}")
            else:
                x_orig_masked = x_orig
                mask = torch.zeros_like(x_orig, dtype=bool)
        elif isinstance(mask, str):
            if mask == "all":
                x_orig_masked = x_orig * 0  # XXX: assumes mask value is 0
                mask = torch.ones_like(x_orig, dtype=bool)
            elif mask == "showcontext":
                x_orig_masked = x_orig
                mask = torch.ones_like(x_orig, dtype=bool)
            else:
                raise ValueError(f"Unknwon mask type {mask!r}")
        else:
            raise TypeError(f"Unknwon mask specification type {type(mask)}")

        if self.mask_context:
            warnings.warn(
                "After v6.0, mask_context should only be set in the DDPM level, instead of the diffusion model.",
                DeprecationWarning,
                stacklevel=2,
            )
            x = x * mask.bool()

        context_list, gene_idx = self.forward_encoder(x_orig_masked, pe_input, input_gene_list, input_gene_idx)
        pred = self.forward_decoder(x, context_list, timesteps, pe_input, conditions, input_gene_list,
                                    input_gene_idx, aug_graph=aug_graph, mask=mask)

        if target_gene_list is not None:
            gene_to_idx = dict(zip(self.pretrained_gene_list, list(range(len(self.pretrained_gene_list)))))
            target_gene_idx = torch.tensor([gene_to_idx[o] for o in target_gene_list if o in gene_to_idx]).long()
            target_gene_idx = target_gene_idx.to(x.device)
            ignored_gene_idx = [x for x in range(len(gene_idx)) if gene_idx[x] not in target_gene_idx]
            mask[:, ignored_gene_idx] = 0

        if self.subset_output:
            pred = pred[:, gene_idx]

        return pred, mask


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

    def forward(self, x_orig, x, t, pe_input=None, conditions=None, input_gene_list=None, target_gene_list=None,
                text_embeddings=None, aug_graph=None, mask=True):
        out = self.diffusion_model(x_orig, x, t, pe_input, conditions, input_gene_list, target_gene_list,
                                   text_embeddings, aug_graph, mask)
        return out

class ScDiff(pl.LightningModule):
    def __init__(self,
                 model_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 loss_strategy="recon_masked",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 monitor_mode="min",
                 use_ema=True,
                 input_key="input",
                 pe_input_key="coord",
                 raw_input_key="raw_input",
                 cond_key="cond",
                 input_gene_list_key="input_gene_list",
                 target_gene_list_key="target_gene_list",
                 target_key='target',
                 pert_target_key='pert_target',
                 target_gene_idx_key='target_gene_idx',
                 cond_mapping_key='cond_mapping_dict',
                 cond_names_key='cond_names',
                 top_de_key='top_de_dict',
                 denoise_mask_key="mask",
                 denoise_target_key="masked_target",
                 text_embeddings_key='text_emb',
                 aug_graph_key='aug_graph',
                 extras_key='extras',
                 cond_names=None,
                 log_every_t=100,
                 in_dim=None,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 learn_logvar=False,
                 logvar_init=0.,
                 mask_context=True,
                 mask_noised_context=True,
                 recon_flag=True,
                 recon_sample=True,
                 denoise_flag=False,
                 denoise_t_sample=1000,
                 denoise_rescale=None,
                 rescale_flag=False,
                 fold_flag=False,
                 classify_flag=True,
                 pert_flag=False,
                 t_sample=500,
                 t_sample_train_mode="indep",  # ['indep', 'tied']
                 t_cond_mapping=None,
                 classifier_config=None,
                 test_target_sum=1e3,
                 in_dropout=0.0,
                 cond_to_ignore: list = None,
                 balance_loss=False,
                 path_to_save_fig='./results/hpoly_ddpm_seed10.png',
                 eval_vlb_flag=False,
                 r_squared_flag=True,
                 **kwargs
        ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.mask_context = mask_context
        self.mask_noised_context = mask_noised_context
        self.recon_flag = recon_flag
        self.recon_sample = recon_sample
        self.denoise_flag = denoise_flag
        self.denoise_t_sample = denoise_t_sample
        self.denoise_rescale = denoise_rescale
        self.rescale_flag = rescale_flag
        self.fold_flag = fold_flag
        self.pert_flag = pert_flag
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.input_key = input_key
        self.pe_input_key = pe_input_key
        self.cond_key = cond_key
        self.raw_input_key = raw_input_key
        self.input_gene_list_key = input_gene_list_key
        self.target_gene_list_key = target_gene_list_key
        self.target_key = target_key
        self.pert_target_key = pert_target_key
        self.target_gene_idx_key = target_gene_idx_key
        self.cond_mapping_key = cond_mapping_key
        self.cond_names_key = cond_names_key
        self.top_de_key = top_de_key
        self.denoise_mask_key = denoise_mask_key
        self.denoise_target_key = denoise_target_key
        self.text_embeddings_key = text_embeddings_key
        self.aug_graph_key = aug_graph_key
        self.extras_key = extras_key
        self.in_dim = in_dim
        self.t_sample = t_sample
        self.t_sample_train_mode = t_sample_train_mode
        self.test_target_sum = test_target_sum
        self.cond_to_ignore = cond_to_ignore
        self.balance_loss = balance_loss
        self.t_cond_mapping = t_cond_mapping
        self.path_to_save_fig = path_to_save_fig
        self.eval_vlb_flag = eval_vlb_flag
        self.r_squared_flag = r_squared_flag
        self.model = DiffusionWrapper(model_config)
        count_params(self.model, verbose=True)

        self.in_dropout = nn.Dropout(in_dropout)

        self.classify_flag = classify_flag
        if self.classify_flag:
            assert classifier_config is not None, "Classify flag set but classifier config not passed"
            self.classifier = instantiate_from_config(classifier_config, model=self)

            if cond_names is None:
                cond_names = ["batch", "cell_type"]
                warnings.warn(
                    f"Condition names cond_names not set, using default of {cond_names!r}. "
                    "This is needed for specifying metric names for classification. "
                    "Please specify to suppress the warning.",
                    UserWarning,
                    stacklevel=2,
                )
            self.cond_names = cond_names

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
            self.monitor_mode = monitor_mode
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        self.register_buffer("unique_conditions", None)

        self.loss_type = loss_type
        assert loss_strategy in ("recon_masked", "recon_full"), f"Unknwon {loss_strategy=}"
        self.loss_strategy = loss_strategy

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        self.val_step_outputs = []
        self.test_step_outputs = []

        self.save_hyperparameters()

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_start, x, t, clip_denoised: bool, pe_input=None, conditions=None, input_gene_list=None,
                        target_gene_list=None, text_embeddings=None, aug_graph=None, mask=None):
        if self.cond_to_ignore is not None:
            assert len(self.cond_to_ignore) <= conditions.shape[1]
            assert all([0 <= x < conditions.shape[1] for x in self.cond_to_ignore])
            conditions[:, self.cond_to_ignore] = 0
        model_out, _ = self.model(x_start, x, t, pe_input=pe_input, conditions=conditions,
                                  input_gene_list=input_gene_list, target_gene_list=target_gene_list,
                                  aug_graph=aug_graph, mask=mask)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            # x_recon.clamp_(-1., 1.)
            x_recon.clamp_(0)

        if target_gene_list is not None:  # only valid when all(t == 0)
            return x_recon, self.posterior_variance[0], self.posterior_log_variance_clipped[0]
        else:
            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_start, x, t, clip_denoised=True, repeat_noise=False, pe_input=None, conditions=None,
                 input_gene_list=None, target_gene_list=None, text_embeddings=None, aug_graph=None,
                 calculate_vlb=False, mask=None):
        b, *_, device = *x.shape, x.device
        # no noise when t == 0
        nonzero_mask = torch.full((x.shape[0], 1), (1 - (t == 0).float()).item()).to(x)
        if all(nonzero_mask == 0) and target_gene_list is not None:
            model_mean, _, _ = self.p_mean_variance(x_start=x_start, x=x, t=t, clip_denoised=clip_denoised,
                                                    pe_input=pe_input, conditions=conditions,
                                                    input_gene_list=input_gene_list, target_gene_list=target_gene_list,
                                                    text_embeddings=text_embeddings, aug_graph=aug_graph, mask=mask)
            return model_mean, 0
        else:
            model_mean, _, model_log_variance = self.p_mean_variance(x_start=x_start, x=x, t=t, mask=mask,
                                                                     clip_denoised=clip_denoised, pe_input=pe_input,
                                                                     conditions=conditions, aug_graph=aug_graph,
                                                                     input_gene_list=input_gene_list)
            noise = noise_like(x.shape, device, repeat_noise)
            if calculate_vlb:
                q_mean, _, q_log_variance = self.q_posterior(x_start, x, t)
                vlb = self.normal_kl(q_mean, q_log_variance, model_mean, model_log_variance).cpu()
                vlb = mean_flat(vlb)
            else:
                vlb = 0
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, vlb

    @torch.no_grad()
    def p_sample_loop(self, x_start, shape, t_start, pe_input=None, conditions=None, input_gene_list=None,
                      target_gene_list=None, text_embeddings=None, aug_graph=None, return_intermediates=False,
                      return_vlb=False, mask=None, inpaint_flag=False):
        assert t_start <= self.num_timesteps
        device = self.betas.device
        noise = torch.randn(shape, device=device)
        if t_start == self.num_timesteps:
            # x = noise  # NOTE: this is incorrect for sampling w diff num of query and ctxt cells
            if isinstance(conditions, torch.Tensor):
                N = conditions.shape[0]
            elif isinstance(conditions, dict):
                N = conditions[list(conditions)[0]].shape[0]
            x = torch.randn(N, x_start.shape[1], device=device)
            t_start -= 1
        else:
            x = self.q_sample(x_start=x_start, t=t_start, noise=noise)

        intermediates = [x]
        # torch.full((b,), i, device=device, dtype=torch.long)
        vlb_list = []
        for i in tqdm(reversed(range(0, t_start + 1)), desc='Sampling t', total=int(t_start + 1)):
            x, vlb = self.p_sample(x_start, x, torch.tensor([i], device=device, dtype=torch.long),
                                   clip_denoised=self.clip_denoised, pe_input=pe_input, conditions=conditions,
                                   input_gene_list=input_gene_list, target_gene_list=target_gene_list,
                                   text_embeddings=text_embeddings, aug_graph=aug_graph, calculate_vlb=return_vlb,
                                   mask=mask)

            if inpaint_flag and i > 0 and mask is not None:
                # Reset masked entries (inpainting setting)
                x[~mask] = self.q_sample(x_start=x_start, t=torch.tensor([i], device=device, dtype=torch.long),
                                         noise=torch.randn_like(x))[~mask]

            vlb_list.append(vlb)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(x)
        if return_vlb:
            return x, torch.stack(vlb_list, dim=1)
        if return_intermediates:
            return x, intermediates
        return x

    @torch.no_grad()
    def sample(self, x_start, t_start, pe_input=None, conditions=None, input_gene_list=None,
               target_gene_list=None, text_embeddings=None, aug_graph=None, return_intermediates=False,
               return_vlb=False, mask=None, inpaint_flag=False):
        # mask: 0 for context, 1 for denoise
        in_dim = self.in_dim
        shape = (x_start.shape[0], in_dim) if in_dim is not None else x_start.shape
        return self.p_sample_loop(x_start, shape, t_start, pe_input=pe_input, conditions=conditions,
                                  input_gene_list=input_gene_list, target_gene_list=target_gene_list,
                                  text_embeddings=text_embeddings, aug_graph=aug_graph, mask=mask,
                                  return_vlb=return_vlb, return_intermediates=return_intermediates,
                                  inpaint_flag=inpaint_flag)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        out = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
               extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        if self.fold_flag:
            out.abs_()
        return out

    def get_loss(self, pred, target, mask=None, mean=True):
        size = mask.numel()
        if mask is not None and self.loss_strategy == "recon_masked":
            pred = pred * mask
            target = target * mask
            size = mask.sum()

        if self.loss_type == 'l1':
            loss = (target - pred).abs()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        if mean:
            loss = loss.sum() / size

        return loss

    def prepare_noised_input(self, x, t, noise=None, mask=None, x_inpaint=None):
        noise = default(noise, lambda: torch.randn_like(x))
        x_inp = self.in_dropout(x)

        # 0 for ctxt, 1 for input
        if isinstance(mask, bool) and not mask:
            mask = torch.zeros_like(x)  # use all for context
        elif not isinstance(mask, torch.Tensor):
            _, mask = self.model.diffusion_model.random_masking(x)

        x_inp_ctxt = x_inp * ~mask

        x_inp_noised = x_inp.clone()
        if x_inpaint is not None:  # XXX: only effective when mask_noised_context=False
            x_inp_noised[~mask] = x_inpaint[~mask]
        elif self.mask_context:
            x_inp_noised *= mask

        if isinstance(t, int):
            t = torch.tensor([t], device=x.device)

        x_inp_noised = self.q_sample(x_start=x_inp_noised, t=t, noise=noise)

        if self.mask_noised_context:
            x_inp_noised = x_inp_noised * mask

        return x_inp_ctxt, x_inp_noised, mask

    def p_losses(self, x_start, t, noise=None, pe_input=None, conditions=None, input_gene_list=None,
                 target_gene_list=None, text_embeddings=None, aug_graph=None, target=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_inp_ctxt, x_inp_noised, mask = self.prepare_noised_input(x_start, t, noise)

        model_out, _ = self.model(x_inp_ctxt, x_inp_noised, t=t, pe_input=pe_input, conditions=conditions,
                                  input_gene_list=input_gene_list, target_gene_list=target_gene_list,
                                  text_embeddings=text_embeddings, aug_graph=aug_graph, mask=mask)

        loss_dict = {}
        if target is not None:
            pass
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mask, mean=False)
        if not self.balance_loss:
            loss = loss.mean(dim=1)
        else:
            nonzero = target != 0
            loss_nonzero = (loss * nonzero).sum(1) / nonzero.sum(1)
            loss_zero = (loss * ~nonzero).sum(1) / (~nonzero).sum(1)
            loss = (loss_nonzero + loss_zero) / 2

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        if self.t_sample_train_mode == "indep":
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        else:
            t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        if k in batch.keys():
            x = batch[k]
            if isinstance(x, torch.Tensor):
                x = x.to(memory_format=torch.contiguous_format).float()
            if k == self.input_key and self.rescale_flag:
                x /= RESCALE_FACTOR
        else:
            x = None
        return x

    def scnormalize(self, x, target_sum=1e4, eps=1e-8):
        x = x * target_sum / (x.sum(1, keepdim=True) + eps)
        x = torch.log1p(x)
        return x

    def invervse_scnormalize(self, x, library_size=1e4, eps=1e-8):
        x = torch.exp(x) - 1
        x = x * library_size / (x.sum(1, keepdim=True) + eps)
        return x

    def maybe_record_conditions(self, batch):
        """Gather conditions information over the full dataset in the first
        training epoch.

        """
        conditions = self.get_input(batch, self.cond_key)
        if (self.current_epoch == 0) and (conditions is not None):
            self.cond_names = list(conditions)
            conditions_tensor = torch.cat([as_1d_vec(conditions[k]) for k in self.cond_names], dim=1)
            # FIX: option to skip (load from pre-trained weights)
            if self.unique_conditions is not None and conditions_tensor.shape[1] != self.unique_conditions.shape[1]:
                self.unique_conditions = conditions_tensor.unique(dim=0)
            else:
                self.unique_conditions = (
                    conditions_tensor.unique(dim=0)
                    if self.unique_conditions is None
                    else torch.cat((self.unique_conditions, conditions_tensor)).unique(dim=0)
                )

    def shared_step(self, batch):
        x = self.get_input(batch, self.input_key)
        pe_input = self.get_input(batch, self.pe_input_key)
        conditions = self.get_input(batch, self.cond_key)
        input_gene_list = self.get_input(batch, self.input_gene_list_key)
        target_gene_list = self.get_input(batch, self.target_gene_list_key)
        text_embeddings = self.get_input(batch, self.text_embeddings_key)
        aug_graph = self.get_input(batch, self.aug_graph_key)
        target = self.get_input(batch, self.target_key)
        loss, loss_dict = self(x, pe_input=pe_input, conditions=conditions, input_gene_list=input_gene_list,
                               target_gene_list=target_gene_list, text_embeddings=text_embeddings,
                               aug_graph=aug_graph, target=target)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        # if self.classify_flag:
        self.maybe_record_conditions(batch)

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return

        target = torch.cat([i['target'] for i in self.val_step_outputs])
        recon = torch.cat([i['recon'] for i in self.val_step_outputs])
        x = torch.cat([i['x'] for i in self.val_step_outputs])
        true_conds = torch.cat([i['true_conds'] for i in self.val_step_outputs])
        de_gene_idx_dict = self.val_step_outputs[0]['de_gene_idx_dict']
        ndde20_idx_dict = self.val_step_outputs[0]['ndde20_idx_dict']
        gene_names = self.val_step_outputs[0]['gene_names']

        scores = perturbation_eval(target, recon, x, gene_names=gene_names, true_conds=true_conds,
                                   de_gene_idx_dict=de_gene_idx_dict, ndde20_idx_dict=ndde20_idx_dict)
        self.log_dict({f"val/{i}": j for i, j in scores.items()},
                      prog_bar=False, logger=True, on_step=False, on_epoch=True)

        self.val_step_outputs.clear()

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        extras = self.get_input(batch, self.extras_key) or {}
        if (
            (de_gene_idx_dict := extras.get("rank_genes_groups_cov_all_idx_dict")) is None
            or (ndde20_idx_dict := extras.get("top_non_dropout_de_20")) is None
        ):
            return

        x = self.get_input(batch, self.input_key)
        target = self.get_input(batch, self.pert_target_key)

        gene_names = self.get_input(batch, "gene_names")
        assert gene_names is not None

        pe_input = self.get_input(batch, self.pe_input_key)
        conditions = self.get_input(batch, self.cond_key)
        input_gene_list = self.get_input(batch, self.input_gene_list_key)
        text_embeddings = self.get_input(batch, self.text_embeddings_key)
        aug_graph = self.get_input(batch, self.aug_graph_key)
        target_gene_list = self.get_input(batch, self.target_gene_list_key)
        t_sample = torch.tensor([self.t_sample]).to(x).long()

        t_sample = torch.tensor([self.t_sample], dtype=int, device=x.device)
        recon = self.sample(x, t_sample, pe_input, conditions, input_gene_list, target_gene_list,
                            text_embeddings, aug_graph=aug_graph, mask="showcontext")

        self.val_step_outputs.append({
            "target": target,
            "recon": recon,
            "x": x,
            "true_conds": conditions['pert'],
            "de_gene_idx_dict": de_gene_idx_dict,
            "ndde20_idx_dict": ndde20_idx_dict,
            "gene_names": gene_names,
        })

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x = self.get_input(batch, self.input_key)
        pe_input = self.get_input(batch, self.pe_input_key)
        conditions = self.get_input(batch, self.cond_key)
        input_gene_list = self.get_input(batch, self.input_gene_list_key)
        text_embeddings = self.get_input(batch, self.text_embeddings_key)
        aug_graph = self.get_input(batch, self.aug_graph_key)
        target_gene_list = self.get_input(batch, self.target_gene_list_key)
        target_gene_idx = self.get_input(batch, self.target_gene_idx_key)
        denoise_mask = self.get_input(batch, self.denoise_mask_key)
        t_sample = torch.tensor([self.t_sample]).to(x).long()

        extras = self.get_input(batch, self.extras_key)
        de_gene_idx_dict = None if extras is None else extras.get("rank_genes_groups_cov_all_idx_dict")
        ndde20_idx_dict = None if extras is None else extras.get("top_non_dropout_de_20")

        null_conditions = {i: torch.zeros_like(j) for i, j in conditions.items()}

        if self.recon_flag:
            target = self.get_input(batch, self.pert_target_key)
            if self.pert_flag:
                gene_names = self.get_input(batch, 'gene_names')
                recon = self.sample(x, t_sample, pe_input, conditions, input_gene_list, target_gene_list,
                                    text_embeddings, aug_graph=aug_graph, mask="showcontext")
            elif self.recon_sample:
                recon, vlb = self.sample(x, t_sample, pe_input, conditions, input_gene_list, target_gene_list,
                                         text_embeddings, aug_graph=aug_graph, return_vlb=True, mask="all")
                if self.eval_vlb_flag:
                    vlb = vlb.sum(dim=1) + self.calculat_prior_kl(x).cpu()

            else:
                noise = torch.randn_like(x)
                if t_sample == self.num_timesteps:
                    x_noised = noise
                    t_sample -= 1
                else:
                    x_noised = self.q_sample(x, t_sample, noise=noise)
                recon, _ = self.model(torch.zeros_like(x), x_noised, t_sample, pe_input, conditions,
                                      input_gene_list, target_gene_list, text_embeddings,
                                      aug_graph=aug_graph, mask=False)
                recon.clamp_(0)

        raw_x = self.get_input(batch, self.raw_input_key)
        # if raw_x is not None and self.recon_flag:
        #     recon = self.invervse_scnormalize(recon, library_size=raw_x.sum(1, keepdim=True))

        pred_conds = target_conds = None
        if self.classify_flag:
            pred_conds, target_conds = self.classifier(x, conditions)

        if self.denoise_flag:
            denoise_t_sample = torch.tensor([self.denoise_t_sample], dtype=torch.long, device=x.device)
            denoise_mask = denoise_mask.bool()
            denoise_recon = self.sample(x, denoise_t_sample, pe_input, conditions, input_gene_list,
                                        target_gene_list, text_embeddings, aug_graph=aug_graph,
                                        mask=denoise_mask, inpaint_flag=True).cpu()
            denoise_target = self.get_input(batch, self.denoise_target_key).cpu()
            denoise_mask = denoise_mask.cpu()

            if self.rescale_flag:
                denoise_target /= RESCALE_FACTOR

            if self.denoise_rescale is not None:
                if self.denoise_rescale == "from_normed":
                    scale = RESCALE_FACTOR
                elif self.denoise_rescale == "to_normed":
                    scale = 1 / RESCALE_FACTOR
                else:
                    raise ValueError(f"Unknown denoising evaluation rescale option {self.denoise_rescale!r}")
                denoise_recon *= scale
                denoise_target *= scale

        else:
            denoise_mask = denoise_recon = denoise_target = None

        out = {
            'x': x.cpu(),
            'raw_x': raw_x.cpu() if raw_x is not None else None,
            'recon': recon.cpu() if self.recon_flag else None,
            'target': target.cpu() if self.pert_flag else None,
            'pred_conds': pred_conds,
            'target_conds': target_conds,
            'denoise_mask': denoise_mask,
            'denoise_recon': denoise_recon,
            'denoise_target': denoise_target,
            'gene_names': gene_names if self.pert_flag else None,
            'conditions': {k: conditions[k].cpu() for k in sorted(conditions)},
            'vlb': vlb.cpu() if self.eval_vlb_flag and self.recon_flag else None,
            'de_gene_idx_dict': de_gene_idx_dict,
            'ndde20_idx_dict': ndde20_idx_dict,
        }
        self.test_step_outputs.append(out)

        return out

    @torch.no_grad()
    def on_test_epoch_end(self):
        assert self.recon_flag or self.classify_flag
        outputs = self.test_step_outputs
        x = torch.cat([outdict['x'].cpu() for outdict in outputs])
        recon = torch.cat([outdict['recon'].cpu() for outdict in outputs]) if self.recon_flag else None
        conditions = {k: torch.cat([outdict['conditions'][k].cpu() for outdict in outputs]).numpy() for k in outputs[0]['conditions'].keys()}

        metrics_dict = {}
        if self.recon_flag:
            if self.pert_flag:
                # target = outputs[0]['target'].cpu()  # scGEN no collate, but effectively the same if mean over batches
                target = torch.cat([outdict['target'].cpu() for outdict in outputs])
                gene_names = outputs[0]['gene_names']  # TODO: add gene_names_key?
                de_gene_idx_dict = outputs[0]['de_gene_idx_dict']
                ndde20_idx_dict = outputs[0]['ndde20_idx_dict']

                if de_gene_idx_dict is not None:  # GEARS eval
                    kwargs = dict(true_conds=torch.from_numpy(conditions['pert']),
                                  de_gene_idx_dict=de_gene_idx_dict, ndde20_idx_dict=ndde20_idx_dict)
                else:
                    kwargs = {}

                scores = perturbation_eval(target, recon, x, gene_names=gene_names,
                                           path_to_save=self.path_to_save_fig, **kwargs)

                metrics_dict.update(scores)
            else:
                if outputs[0]['raw_x'] is not None:
                    raw_x = torch.cat([outdict['raw_x'].cpu() for outdict in outputs])
                    recon_inv = self.invervse_scnormalize(recon, library_size=raw_x.sum(1, keepdim=True))
                    poisson_nll = nn.PoissonNLLLoss(log_input=False, full=True)(recon_inv, raw_x).item()
                    rmse = np.sqrt(F.mse_loss(recon_inv, raw_x).item())
                    rmse_normed = np.sqrt(F.mse_loss(recon, x).item())
                else:
                    poisson_nll = nn.PoissonNLLLoss(log_input=False, full=True)(recon, x).item()
                    rmse = np.sqrt(F.mse_loss(recon, x).item())
                    x_normed = self.scnormalize(x, target_sum=self.test_target_sum)
                    recon_normed = self.scnormalize(recon, target_sum=self.test_target_sum)
                    rmse_normed = np.sqrt(F.mse_loss(recon_normed, x_normed).item())

                metrics_dict['poisson_nll'] = poisson_nll
                metrics_dict['rmse'] = rmse
                metrics_dict['rmse_normed'] = rmse_normed
                if self.eval_vlb_flag:
                    vlb = torch.cat([outdict['vlb'].cpu() for outdict in outputs])
                    metrics_dict['vlb'] = vlb.mean().item()
                if self.r_squared_flag:
                    r_squared_list = calculate_batch_r_squared(recon, x, conditions)
                    # metrics_dict['recon_R^2_list'] = r_squared_list
                    metrics_dict['recon_R^2'] = sum(r_squared_list) / len(r_squared_list)

        if self.denoise_flag:
            denoise_mask = torch.cat([outdict['denoise_mask'].cpu() for outdict in outputs])
            denoise_recon = torch.cat([outdict['denoise_recon'].cpu() for outdict in outputs])
            denoise_target = torch.cat([outdict['denoise_target'].cpu() for outdict in outputs])
            metrics_dict.update(denoising_eval(denoise_target, denoise_recon, denoise_mask))

        if self.classify_flag:
            for cond_name in outputs[0]['pred_conds']:
                pred_conds = torch.cat([outdict['pred_conds'][cond_name] for outdict in outputs]).numpy()
                true_conds = torch.cat([outdict['target_conds'][cond_name] for outdict in outputs]).numpy()
                metrics_dict.update(evaluate_annotation(true_conds, pred_conds, cond_name))

        self.log_dict(metrics_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('t_sample', self.t_sample, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.test_step_outputs.clear()

    # Alias for compatibility with early versions of Lightning
    # test_epoch_end = on_test_epoch_end

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        # Length: number of context cells
        x = self.get_input(batch, self.input_key)
        # Length: number of context cells + number of generated cells
        pe_input = None  # not supported yet
        conditions = self.get_input(batch, self.cond_key)
        mask = torch.ones_like(x)  # NOTE: we use all mask to turn off context 

        if False:  # XXX: fast dev test (do not use for production!)
            t = torch.LongTensor([1]).to(x.device)
            x_gen = self.sample(x, t, pe_input, conditions[x.shape[0]], mask=mask)
        else:
            t = torch.LongTensor([self.num_timesteps]).to(x.device)
            x_gen = self.sample(x, t, pe_input, conditions, mask=mask)

        return {
            "x_gen": x_gen.cpu(),
            "query_conditions": {k: v.cpu() for k, v in conditions.items()},
            "context_cell_ids": batch["context_cell_ids"],
        }

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def calculate_vlb(self, x_start, pe_input=None, conditions=None, input_gene_list=None, aug_graph=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]
        full_mask = torch.ones_like(x_start, dtype=bool)

        vlb = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            p_mean, _, p_log_variance = self.p_mean_variance(x_start, x_t, t, self.clip_denoised, pe_input,
                                                             conditions, input_gene_list, aug_graph=aug_graph,
                                                             mask=full_mask)
            q_mean, _, q_log_variance = self.q_posterior(x_start, x_t, t)
            vlb.append(self.normal_kl(q_mean, q_log_variance, p_mean, p_log_variance))
        vlb = torch.stack(vlb, dim=1)
        prior_kl = self.calculat_prior_kl(x_start)
        total_vlb = vlb.sum(dim=1) + prior_kl
        return total_vlb

    def calculat_prior_kl(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = self.normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.

        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"

        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for th.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]

        return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )
