import argparse
import datetime
import glob
import os
import sys
import time
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, IterableDataset

from scdiff.collate import CollateDictionary
from scdiff.data.base import Txt2ImgIterableBaseDataset
from scdiff.patches import pl_trainer_add_argparse_args, pl_trainer_from_argparse_args
from scdiff.utils.data import combine_predictions
from scdiff.utils.misc import instantiate_from_config

KEYS_TO_IGNORE = ['input_gene_list', 'target_gene_list', 'target_gene_idx', 'cond_names',
                  'cond_mapping_dict', 'top_de_dict', 'target', 'pert_target', 'gene_names',
                  'aug_graph', 'extras']


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?",
                        help="resume from logdir or checkpoint in logdir")
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml",
                        help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten "
                        "or added with command-line options of the form `--key value`.", default=[])
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=True, nargs="?", help="train")
    parser.add_argument("--no_test", type=str2bool, const=True, default=False, nargs="?", help="disable test")
    parser.add_argument("--predict", type=str2bool, const=True, default=False, nargs="?", help="enable prediction")
    parser.add_argument("--pred_save_path", type=str, default="", help="path to save the generated predictions")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False,
                        help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=10, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True,
                        help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--wandb", type=str2bool, const=True, default=True, nargs="?", help="enable wandb logger")
    parser.add_argument("--wandb_offline", type=str2bool, const=True, default=False, nargs="?", help="offline wandb")
    parser.add_argument("--wandb_project", type=str, default="scDiff", help="wandb project name")
    parser.add_argument("-g", "--group", type=str, default="runs", help="group name of wandb experiments")
    parser.add_argument("-sub", "--subset", default=False, action="store_true",
                        help="whether to subset origin output or build new prediction layer")
    parser.add_argument("-w", "--weights_only", type=str2bool, const=True, default=False, nargs="?",
                        help="only load weights or resume training")
    parser.add_argument("--ckpt_every_n_epochs", type=int, default=50)
    parser.add_argument("-cg", "--custom_clip_gradient", type=float, default=None)
    parser.add_argument("-om", "--overwrite_model", type=str2bool, const=True, default=False, nargs="?",
                        help="overwrite model configs with pretraining config")
    parser.add_argument("-abc", "--allow_both_ckpts", type=str2bool, const=True, default=False, nargs="?",
                        help="allow both pretrained checkpoint and finetuned checkpoint")
    parser.add_argument("-lc", "--load_conditions", type=str2bool, const=True, default=True, nargs="?",
                        help="whether to laod model.unique_conditions")
    parser.add_argument("--load_best", type=str2bool, const=True, default=False, nargs="?",
                        help="Load best checkpoint from best k checkpoints")
    parser.add_argument("--profile_train", type=str2bool, const=True, default=False, nargs="?")
    parser.add_argument("--profile_test", type=str2bool, const=True, default=False, nargs="?")
    parser.add_argument("--profile_pred", type=str2bool, const=True, default=False, nargs="?")
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = pl_trainer_add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def _setup(self, stage=None):
        # XXX: We call _setup once to set up the data. Lightning calls setup
        # everytime it enters train / test / prediction. There's no need to
        # redo the preprocessing (which could be expensive).
        self.datasets = {}
        for data_name, data_cfg in self.dataset_configs.items():
            print(f"Initializing {data_name} with config:\n{OmegaConf.to_yaml(data_cfg)}")
            data_obj = instantiate_from_config(data_cfg)
            self.datasets[data_name] = WrappedDataset(data_obj) if self.wrap else data_obj

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          collate_fn=CollateDictionary(KEYS_TO_IGNORE), num_workers=self.num_workers,
                          shuffle=False if is_iterable_dataset else True, worker_init_fn=init_fn,
                          persistent_workers=True, pin_memory=True)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle,
                          persistent_workers=True,
                          pin_memory=True,
                          collate_fn=CollateDictionary(KEYS_TO_IGNORE))

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['test'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size, persistent_workers=True,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, pin_memory=True,
                          collate_fn=CollateDictionary(KEYS_TO_IGNORE))

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        if isinstance(self.datasets['predict'], IterableDataset):
            return DataLoader(self.datasets["predict"], batch_size=None, worker_init_fn=init_fn)
        else:
            return DataLoader(self.datasets["predict"], batch_size=self.batch_size, persistent_workers=True,
                              num_workers=self.num_workers, worker_init_fn=init_fn, pin_memory=True)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class PatchedModelCheckpoint(ModelCheckpoint):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        is_save_epoch = self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        is_final_epoch = trainer.current_epoch + 1 == trainer.max_epochs
        # Save last epoch once every_n_epochs instead of every epoch
        if is_save_epoch or is_final_epoch:
            # print(f"\n[*] Try to save ckpt at {trainer.current_epoch}\n")
            super().on_train_epoch_end(trainer, pl_module)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class GradNormCallback(Callback):
    def __init__(self, gradient_clip_val=None):
        super().__init__()
        self.gradient_clip_val = gradient_clip_val

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        log_opts = dict(prog_bar=False, logger=True, on_step=False, on_epoch=True)

        if self.gradient_clip_val is not None:
            clip_grad_norm_(pl_module.parameters(), self.gradient_clip_val)

        gns = {
            f"gradnorm/{name}": params.grad.data.norm(2).item()
            for name, params in pl_module.named_parameters() if params.grad is not None
        }

        if gns:
            pl_module.log_dict(gns, **log_opts)
            gn = torch.tensor(list(gns.values())).norm(2).item()
            pl_module.log("gradnorm", gn, **log_opts)


def maybe_hack_get_best_ckpt_path(ckpt_path, load_best):
    """Load best (highest val score) model for testing."""
    if not load_best:
        return None

    if ckpt_path is None:
        # Try to get from trainer's model checkpoint call back
        for cbk in trainer.callbacks:
            if (ckpt_path := getattr(cbk, "last_model_path", None)) is not None:
                break
        else:
            warnings.warn(
                "Unable to retrieve last checkpoint path, falling back to using default checkpoint",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    cbks = torch.load(ckpt_path).get("callbacks")
    if cbks is None:
        warnings.warn(
            f"Unable to retrieve callbacks info (contains checkpoints info) from {ckpt_path}, "
            "falling back to using default checkpoint.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    for cbk in cbks.values():
        if "best_k_models" in cbk:
            ckpt_info = cbk["best_k_models"]
            print(f"Best k model info:\n{ckpt_info}")
            best_ckpt = sorted(ckpt_info, key=ckpt_info.get)[-1]
            print(f"Best ckpt selected: {best_ckpt}")
            return best_ckpt

    raise ValueError(f"Unable to obtain best k model info from {ckpt_path}")


@contextmanager
def maybe_profile_ctxt(enable: bool = False, msg: Optional[str] = None):
    try:
        if not enable:
            yield
        else:
            from cProfile import Profile
            from pstats import SortKey, Stats

            if msg is None:
                msg = " Profiling begins "
            else:
                msg = f" Profiling begins ({msg}) "

            print(f"\n{msg:=^100}")
            with Profile() as profile:
                yield profile
    finally:
        if enable:
            Stats(profile).sort_stats(SortKey.CUMULATIVE).print_stats()
            exit()


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = pl_trainer_add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        ckpt_path = ckpt
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            # name = "_" + opt.name
            name = opt.name + "_"
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            # name = "_" + cfg_name
            name = cfg_name + "_"
        else:
            name = ""
        # nowname = now + name
        if opt.postfix:
            # nowname += "_" + opt.postfix
            name += opt.postfix + "_"
        nowname = name + now
        logdir = os.path.join(opt.logdir, nowname)
        ckpt_path = None

    os.makedirs(logdir, exist_ok=True)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    # prfdir = os.path.join(logdir, "profile")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs)
        pretrained_ckpt_path = config.get("pretrained_ckpt_path") or cli.get("pretrained_ckpt_path")
        if not opt.allow_both_ckpts:
            assert pretrained_ckpt_path is None or ckpt_path is None
        if pretrained_ckpt_path is not None and ckpt_path is None:
            ckpt_path = pretrained_ckpt_path

            if os.path.isfile(ckpt_path):
                if opt.overwrite_model:
                    paths = ckpt_path.split("/")
                    config_path = os.path.join('/'.join(paths[:-2]), "configs")
                    pretrained_configs = os.listdir(config_path)
                    pretrained_configs = OmegaConf.merge(*[
                        OmegaConf.load(os.path.join(config_path, cfg)) for cfg in pretrained_configs
                    ])
                    config.model = pretrained_configs.get("model", OmegaConf.create())
            else:
                assert os.path.isdir(ckpt_path), ckpt_path
                config_path = os.path.join(ckpt_path, "configs")
                ckpt_path = os.path.join(ckpt_path, "checkpoints", "last.ckpt")
                if opt.overwrite_model:
                    pretrained_configs = os.listdir(config_path)
                    pretrained_configs = OmegaConf.merge(*[
                        OmegaConf.load(os.path.join(config_path, cfg)) for cfg in pretrained_configs
                    ])
                    config.model = pretrained_configs.get("model", OmegaConf.create())
        config = OmegaConf.merge(config, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = "gpu"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if "devices" not in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["devices"]
            print(f"Running on GPUs {gpuinfo}")
            if len(trainer_config["devices"]) > 1:
                trainer_config["strategy"] = "ddp"
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # data
        if not opt.train:
            assert ('test' in config.data.params.keys()
                    or 'predict' in config.data.params.keys())
            config.data.params.pop('train', None)
            config.data.params.pop('validation', None)
        if opt.no_test:
            config.data.params.pop('test', None)
        data = instantiate_from_config(config.data)
        data._setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k].adata)}")

        # model
        dataset_keys = list(data.datasets.keys())
        if (
            'pretrained_gene_list' not in config.model.params.model_config.params.keys()
            and getattr(data.datasets[dataset_keys[0]], "pretrained_gene_list", None) is None
        ):
            config.model.params.model_config.params['pretrained_gene_list'] = (
                data.datasets[dataset_keys[0]].adata.var.index.to_list())
            new_genes = []
        else:
            if config.model.params.model_config.params.get('pretrained_gene_list') is not None:
                pretrained_gene_list = config.model.params.model_config.params['pretrained_gene_list']
            else:
                assert hasattr(data.datasets[dataset_keys[0]], 'pretrained_gene_list')
                pretrained_gene_list = data.datasets[dataset_keys[0]].pretrained_gene_list.tolist()
            input_gene_list = data.datasets[dataset_keys[0]].adata.var.index.to_list()
            new_genes = list(set(input_gene_list) - set(pretrained_gene_list))
            gene_list = pretrained_gene_list + new_genes
            config.model.params.model_config.params['pretrained_gene_list'] = gene_list
            config.model.params.model_config.params['input_gene_list'] = input_gene_list

        if config.model.params.model_config.params.cond_emb_type == 'embedding':
            config.model.params.model_config.params['cond_num_dict'] = data.datasets[dataset_keys[0]].cond_num_dict
            config.model.params.model_config.params['post_cond_num_dict'] = (
                data.datasets[dataset_keys[0]].post_cond_num_dict)

        if (
            hasattr(data.datasets[dataset_keys[0]], 'G_go') and
            hasattr(data.datasets[dataset_keys[0]], 'G_go_weight') and
            hasattr(data.datasets[dataset_keys[0]], 'num_perts')
        ):
            config.model.params.model_config.params['num_perts'] = data.datasets[dataset_keys[0]].num_perts
            config.model.params.model_config.params['gears_flag'] = True

        model = instantiate_from_config(config.model)
        config.model.params.model_config.params.pop('pretrained_gene_list', None)
        config.model.params.model_config.params.pop('input_gene_list', None)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "project": opt.wandb_project,
                    "entity": "danceteam",
                    "group": opt.group,
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.wandb_offline,
                    "id": nowname,
                }
            },
            "csv": {
                "target": "pytorch_lightning.loggers.CSVLogger",
                "params": {
                    "name": "csv",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["wandb"] if opt.wandb else default_logger_cfgs["csv"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "main.PatchedModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
                "save_on_train_epoch_end": True,
                "every_n_epochs": opt.ckpt_every_n_epochs,
                # "every_n_train_steps": 200,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["mode"] = model.monitor_mode
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
            "gradnorm_callback": {
                "target": "main.GradNormCallback",
                "params": {
                    "gradient_clip_val": opt.custom_clip_gradient,
                },
            }
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        # trainer_kwargs["profiler"] = SimpleProfiler(dirpath=prfdir, filename=opt.name + opt.postfix)
        trainer = pl_trainer_from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir

        # configure learning rate and weight decay
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.devices)  # strip(",").split(',')
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(f"Setting learning rate to {model.learning_rate:.2e} = "
                  f"{accumulate_grad_batches} (accumulate_grad_batches) "
                  f"* {ngpu} (num_gpus) * {bs} (batchsize) * {base_lr:.2e} (base_lr)")
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")
        model.weight_decay = config.model.get("weight_decay", 0)

        # allow checkpointing via USR1

        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # load weights and run
        def check_weights(model, ckpt_path, load_conditions=True):
            print(f"Ckecking model states from the checkpoint path at {ckpt_path}")
            manual_load_list = ["unique_conditions"] if load_conditions else []
            model_dict = model.state_dict()
            ckpt_state_dict = torch.load(ckpt_path, map_location=model.device)["state_dict"]
            loadable_state_dict = {
                k: v
                for k, v in ckpt_state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            ignored_keys = [x for x in list(ckpt_state_dict) if x not in list(loadable_state_dict) + manual_load_list]
            not_pretrained_keys = [x for x in list(model_dict) if x not in list(loadable_state_dict)]
            if len(new_genes) > 0:  # only support OmicsEmbeddingLayer for both encoder_embed and decoder_embed
                encoder_embed_key = 'model.diffusion_model.encoder_embed.feat_enc.emb'
                decoder_embed_key = 'model.diffusion_model.decoder_embed.feat_enc.emb'
                pretrained_embed = {
                    encoder_embed_key: ckpt_state_dict[encoder_embed_key],
                    decoder_embed_key: ckpt_state_dict[decoder_embed_key],
                }
            else:
                pretrained_embed = None
            if len(ignored_keys) > 0:
                print(f"Ignored keys: {ignored_keys}")
            if len(not_pretrained_keys) > 0:
                print(f"Not pretrained keys: {not_pretrained_keys}")
            for key in manual_load_list:
                if key in ckpt_state_dict:
                    setattr(model, key, ckpt_state_dict[key])
            return loadable_state_dict, ignored_keys, not_pretrained_keys, pretrained_embed

        def load_weights(model, loadable_state_dict, pretrained_embed=None):
            print(f"Restoring states from the checkpoint path at {ckpt_path}")
            model_dict = model.state_dict()
            model_dict.update(loadable_state_dict)
            model.load_state_dict(model_dict)
            if len(new_genes) > 0 and pretrained_embed is not None:
                # only support OmicsEmbeddingLayer for both encoder_embed and decoder_embed
                with torch.no_grad():
                    for k in pretrained_embed:
                        import operator
                        operator.attrgetter(k)(model).data[:len(pretrained_gene_list)] = pretrained_embed[k]
                        # exec(f"model.{k}.data[:len(pretrained_gene_list)] = pretrained_embed[k]")
            return model

        with maybe_profile_ctxt(enable=opt.profile_train, msg="Training"):
            if opt.train:
                if ckpt_path is not None:  # TODO: force to retrain conditioner embeddings
                    loadable_state_dict, ignored_keys, not_pretrained_keys, pretrained_embed = check_weights(
                        model, ckpt_path, opt.load_conditions)
                    if len(ignored_keys) > 0 or len(not_pretrained_keys) > 0 or opt.weights_only:
                        model = load_weights(model, loadable_state_dict, pretrained_embed)
                        model.unique_conditions = None  # HACK: force reinitialize conditions
                        trainer.fit(model, data)
                    else:
                        trainer.fit(model, data, ckpt_path=ckpt_path)
                else:
                    trainer.fit(model, data)
            elif ckpt_path is not None:
                loadable_state_dict, _, _, pretrained_embed = check_weights(model, ckpt_path)
                model = load_weights(model, loadable_state_dict, pretrained_embed)

        with maybe_profile_ctxt(enable=opt.profile_test, msg="Testing"):
            if not opt.no_test and not trainer.interrupted:
                ckpt_path = maybe_hack_get_best_ckpt_path(ckpt_path, opt.load_best)
                trainer.test(model, data, ckpt_path)

        with maybe_profile_ctxt(enable=opt.profile_pred, msg="Prediction"):
            if opt.predict and not trainer.interrupted:
                assert opt.pred_save_path, "Specify --pred_save_path for saving predictions"
                predictions = trainer.predict(model, data)
                combine_predictions(data, predictions, opt.pred_save_path)

    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import ipdb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
