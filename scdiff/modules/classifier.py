"""Zero-shot classification from diffusion model.

Modified from
https://github.com/diffusion-classifier/diffusion-classifier/blob/a5c4eb8f4d5d68cf85067eb0847255da3b5dcf6e/eval_prob_adaptive.py

@misc{li2023diffusion,
      title={Your Diffusion Model is Secretly a Zero-Shot Classifier},
      author={Alexander C. Li and Mihir Prabhudesai and Shivam Duggal and Ellis Brown and Deepak Pathak},
      year={2023},
      eprint={2303.16203},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

"""
import inspect
import itertools
import warnings
from functools import partial
from typing import Any, Callable, Literal, List, Dict, Optional, Union, get_args

import numpy as np
import torch
import tqdm
from torch.nn import functional as F

from scdiff.utils.misc import check_str_option, default, list_exclude


CLF_LOSS_TYPE = Literal["l1", "l2", "huber", "poisson_kl"]
CLF_QUERY_MODE = Literal["all", "seen", "batch_all", "batch_seen", "specified"]
TS_SAMPLER_TYPE = Literal["IterativeUniform"]


class IterativeUniformTimeStepSampler:
    """Sample time steps in uniform grids with increasing granularity."""

    def __init__(
        self,
        max_time_steps: int,
        max_num_steps: int,
        num_repeats: int = 1,
        random_state: Optional[int] = 42,
    ):
        self.max_time_steps = max_time_steps
        self.max_num_steps = max_num_steps
        self.num_repeats = num_repeats
        self.rng = np.random.default_rng(random_state)

        time_steps = self._get_time_steps(max_time_steps, max_num_steps)
        if len(time_steps) > max_num_steps:
            # print(time_steps)
            time_steps = self._sample_sorted(time_steps, max_num_steps, self.rng)
            # print(time_steps)
        self.time_steps = time_steps

        self.sampled_set = set()

    @staticmethod
    def _get_time_steps(max_time_steps: int, num_steps: int) -> List[int]:
        interval = max_time_steps // num_steps
        start = interval // 2
        return list(range(start, max_time_steps, interval))

    @staticmethod
    def _sample_sorted(
        time_steps: List[int],
        max_num_steps: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        rng = rng or np.random.default_rng()
        return sorted(rng.choice(time_steps, size=max_num_steps, replace=False))

    def __call__(
        self,
        num_steps: int,
        register_steps: bool = True,
        shuffle: bool = True,
    ) -> List[int]:
        """Sample time steps.

        Args:
            num_steps: Number of uniform steps to take.
            register_steps: If set to True, then only return steps that are not
                drawn from previous draws and register the returned steps as
                drawn.
            shuffle: If set to True and the number of uniform steps exceed the
                specified num_steps, then uniformly randomly sample steps.

        """
        idxs = self._get_time_steps(self.max_num_steps, num_steps)

        if shuffle and len(idxs) > num_steps:
            idxs = self._sample_sorted(idxs, num_steps, self.rng)

        if register_steps:
            idxs = sorted(set(idxs) - self.sampled_set)
            self.sampled_set.update(idxs)

        time_steps = [self.time_steps[i] for i in idxs]
        time_steps = np.repeat(time_steps, self.num_repeats).tolist()

        return time_steps


class DiffusionClassifier:
    """Diffusion classifier object.

    From a high level, the classifier works as follow

        1. Use all unique conditions as candidate.
        2. Evaluate reconstruction errors of cells with different candidate
           conditions.
        3. Select the top k conditions for each cell that resulted in the
           lowest errors and use these as the new candidate conditions.
           Note that k gradually decrease from round to round, and eventually
           drops to 1 in the final round to select the best matching condition
           for each cell.
        4. Repeat 2 and 3 until the last round as reached.

    Args:
        n_samples_list: List of samples to draw from in each round.
        to_kee_list: List of top conditions to keep for next round of
            evaluation.
        n_trials: Number of trials per sampled time points.
        loss: Type of loss to evaluate the error of the predictions given a
            particular condition.
        query_mode: What conditions to query for. "all" uses all possible
            combinations of the conditions, "seen" uses only the combinations
            seen during training, and "specified" uses the combinations passed
            to the :attr:`conditions` argument. "batch_all" and "batch_seen"
            are analogous to "all" and "seen" but only select from the batch.
        model: Model object to bind to. See
            :class:`scdiff.models.diffusion.ddpm.DDPM` for an example.

    Notes:
        There are several requirements for the binding model to work with the
        :class:`DiffusionClassifier` object.

            1. There should be a :attr:`diffusion_model` (or
               :attr:`model.diffusion_model`) attribute, which holds the
               denoising diffusion model.
            2. There should be a :attr:`num_timesteps` attribute about the
               maximum time step size of the diffusion model.
            3. There should be a :meth:`q_sample` method that performs forward
               sampling of the original data, i.e., adding noise.

    """

    def __init__(
        self,
        n_samples_list: List[int],
        to_keep_list: List[int],
        n_trials: int = 1,
        loss: CLF_LOSS_TYPE = "l2",
        query_mode: CLF_QUERY_MODE = "all",
        inference_mask: Union[bool, str] = "all",
        time_step_sampler: TS_SAMPLER_TYPE = "IterativeUniform",
        model: Optional[Any] = None,
        conds_to_fix: Optional[Union[str, List[str]]] = None,
        conds_to_null: Optional[Union[str, List[str]]] = None,
    ):
        assert len(n_samples_list) == len(to_keep_list)
        assert to_keep_list[-1] == 1, "Last trial must only select one best matching condition."
        self.n_samples_list = n_samples_list
        self.to_keep_list = to_keep_list
        self.n_trials = n_trials
        self.loss = check_str_option("loss", loss, CLF_LOSS_TYPE)
        self.query_mode = check_str_option("query_mode", query_mode, CLF_QUERY_MODE)
        convert = lambda x: [x] if isinstance(x, str) else x
        self.conds_to_fix = set(default(convert(conds_to_fix), []))
        self.conds_to_null = set(default(convert(conds_to_null), []))
        self.inference_mask = inference_mask
        self._model = model

    @property
    def get_time_step_sampler(self):
        return self._time_step_sampler_cls

    @get_time_step_sampler.setter
    def get_time_step_sampler(self, val):
        if val == "IterativeUniform":
            self._time_step_sampler_cls = IterativeUniformTimeStepSampler
        elif val not in (opts := get_args(TS_SAMPLER_TYPE)):
            raise ValueError(f"Unknown time step sampler {val!r}, available options are {opts}")
        else:
            raise NotImplementedError(f"[DEVERROR] please implement {val}")

    @property
    def model(self):
        return self._model

    def __call__(
        self,
        x: torch.Tensor,
        x_conds: Dict[str, torch.Tensor],
        model: Optional[Any] = None,
        specified_conds: Optional[torch.Tensor] = None,
    ):
        """Predict conditions of each cell.

        Args:
            x_orig: Original expression values to be used for generating noised
                expression input to the diffusion model.
            x_conds: Conditions of all cells (i.e., rows of x_orig). These
                are the label to be predicted against.
            model: Model object to bind to. See
                :class:`scdiff.models.diffusion.ddpm.DDPM` for an example.
                Only needed if model was not binded during initialization.
            specified_conds: Specify the conditions to query for in "specified"
                query mode.

        Returns:
            Predicted conditions for each cell.

        """
        # Hook up with model
        if (model is None) and (self.model is None):
            raise ValueError("Model object not stored during init, please pass during call.")
        if (model is not None) and (self.model is not None):
            warnings.warn(
                f"Model object already specified during init: {self.model} but "
                f"also passed during call: {model}. Using the passed model "
                "in this call. Please remove duplicated model specification "
                "to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
        model = default(model, self.model)
        diffusion_model, timesteps = self.get_assets_from_model(model)
        conditions = self._get_conditions(model, x_conds, specified_conds)
        assert conditions, "Failed to extract query conditions"

        # Set constants
        max_n_samples = max(self.n_samples_list)
        num_cells = len(x)
        num_conditions = len(conditions[list(conditions)[0]])
        num_t_splits = len(self.n_samples_list)
        n_samples_list = self.n_samples_list
        to_keep_list = self.to_keep_list

        if num_conditions < (max_to_keep := max(to_keep_list)):
            warnings.warn(
                f"Maximum conditions to keep ({max_to_keep}) exceeds the total "
                f"number of conditions available ({num_conditions}). Implicitly "
                f"setting max number of conditions to keep to {num_conditions}.",
                UserWarning,
                stacklevel=2,
            )
            to_keep_list = [min(num_conditions, i) for i in to_keep_list]

        # TODO: support other samplers
        time_step_sampler = IterativeUniformTimeStepSampler(
            timesteps,
            max_n_samples,
            self.n_trials,
        )

        eval_error_func = partial(
            self.eval_error,
            diffusion_model=diffusion_model,
            q_sample=model.q_sample,
            x_orig=x,
            x_conds=x_conds,
            query_conds=conditions,
            loss=self.loss,
            inference_mask=self.inference_mask,
        )

        full_error_tensor = torch.full((num_t_splits, num_cells, num_conditions), torch.nan)
        best_idxs = torch.arange(num_conditions).repeat(num_cells, 1)
        for i, (n_samples, n_to_keep) in enumerate(zip(n_samples_list, to_keep_list)):
            error_tensor = torch.zeros_like(best_idxs, dtype=torch.float)
            curr_t_to_eval = time_step_sampler(n_samples)

            # Perform one round of evaluation
            for j in tqdm.trange(
                best_idxs.shape[1],
                leave=False,
                desc=f"Round {i + 1} / {len(n_samples_list)}",
            ):
                error_tensor[:, j] = eval_error_func(ts=curr_t_to_eval, query_cond_idx=best_idxs[:, j])

            # Aggregate evaluation results across rounds
            if i == 0:
                full_error_tensor[i] = error_tensor
                prev_size = len(curr_t_to_eval)
            else:
                curr_size = len(curr_t_to_eval)
                full_error_tensor[i].scatter_(dim=1, index=best_idxs, src=error_tensor)
                # Aggregate errors with previous runs
                full_error_tensor[i] = (
                    (full_error_tensor[i] * curr_size + full_error_tensor[i-1] * prev_size)
                    / (curr_size + prev_size)
                )
                prev_size += curr_size

            # Find best conditions for each cell
            new_best_vals, new_best_idxs = error_tensor.topk(n_to_keep, dim=1, largest=False)
            assert not torch.isnan(new_best_vals).any(), "Found nans in selected entries."
            # Convert idx back to original condition idx
            best_idxs = best_idxs.gather(dim=1, index=new_best_idxs)

        pred_conds, target_conds = {}, {}
        for i, j in conditions.items():
            if len(x_conds[i].unique()) == 1:
                warnings.warn(
                    "Current batch only has one type of {i}, try increasing batch size?",
                    RuntimeWarning,
                    stacklevel=2,
                )
            pred_conds[i] = j[best_idxs.flatten()].cpu()
            target_conds[i] = x_conds[i].cpu()

        return pred_conds, target_conds

    def _get_conditions(
        self,
        model: torch.nn.Module,
        x_conds: Dict[str, torch.Tensor],
        specified_conds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # First prepare the unique conditions, either from the observations
        # from the batch, or from the full training dataset (recorded by the
        # unique_conditions attribute in the model object)
        self.all_cond_names = model.cond_names
        self.cond_names = list_exclude(self.all_cond_names, self.conds_to_fix | self.conds_to_null)
        if not self.query_mode.startswith("batch_"):
            unique_conditions = model.unique_conditions

        old_version_ckpt_flag = getattr(model, "unique_conditions") is None
        if self.query_mode.startswith("batch_") or old_version_ckpt_flag:
            if old_version_ckpt_flag:
                warnings.warn(
                    "The model ckpt being used is from an older version that "
                    "do not contain the 'unique_conditions'. Implicitly "
                    f"switching the query_mode from {self.query_mode} to "
                    f"batch_{self.query_mode}",
                    UserWarning,
                    stacklevel=2,
                )
            conditions_tensor = torch.cat([x_conds[k].unsqueeze(1) for k in self.all_cond_names], dim=1)
            unique_conditions = conditions_tensor.unique(dim=0)

        # Extract conditions to query
        valid_cond_idx = [i for i, j in enumerate(self.all_cond_names) if j in self.cond_names]
        unique_conditions = unique_conditions[:, valid_cond_idx].unique(dim=0)

        # Remove NULL conditions from query candidates
        unique_conditions = unique_conditions[~(unique_conditions == 0).any(1)]

        # The only difference is how we prepare the unique_conditions, which
        # we have handeled above
        query_mode = self.query_mode.replace("batch_", "")

        if query_mode == "all":
            individual_unique_conditions = [i.unique().tolist() for i in unique_conditions.T]
            return torch.tensor(
                list(itertools.product(*individual_unique_conditions)),
                device=unique_conditions.device,
            )
        elif query_mode == "seen":
            out = unique_conditions
        elif query_mode == "specified":
            if specified_conds is None:
                raise ValueError("query_mode set to 'specified' but did not passed specified conditions")
            elif not isinstance(specified_conds, torch.Tensor):
                raise TypeError(f"Please pass specified contions as tensor, got {type(specified_conds)}")
            elif specified_conds.shape[1] != unique_conditions:
                raise ValueError(
                    f"Inconsistent condition type number. Got {specified_conds.shape[1]} "
                    f"conditions in the specified conditions, but model only recorded "
                    f"{unique_conditions.shape[1]} conditions.",
                )
            # FIX: specified conds might not match cond_names
            out = specified_conds
        else:
            raise NotImplementedError(query_mode)

        return {i: j for i, j in zip(self.cond_names, out.T)}

    @staticmethod
    def get_assets_from_model(model):
        if hasattr(model, "diffusion_model"):
            diffusion_model = model.diffusion_model
        else:
            diffusion_model = model.model.diffusion_model

        if hasattr(model, "num_timesteps"):
            timesteps = model.num_timesteps
        else:
            timesteps = model.timesteps

        return diffusion_model, timesteps

    @torch.inference_mode()
    def eval_error(
        self,
        *,
        diffusion_model: torch.nn.Module,
        q_sample: Callable,
        x_orig: torch.Tensor,
        x_conds: Dict[str, torch.Tensor],
        query_conds: Dict[str, torch.Tensor],
        query_cond_idx: torch.Tensor,
        ts: List[int],
        loss: CLF_LOSS_TYPE = 'l2',
        inference_mask: Union[bool, str] = "all",
    ) -> torch.Tensor:
        device = x_orig.device
        pred_errors = torch.zeros(len(x_orig), device=device)
        x_empty = torch.zeros_like(x_orig)  # use decoder only (no context encoder)

        conditions = {}
        for i, j in x_conds.items():
            if i in query_conds:
                conditions[i] = query_conds[i][query_cond_idx]  # query
            elif i in self.conds_to_fix:
                conditions[i] = j  # fixed from input
            elif i in self.conds_to_null:
                conditions[i] = torch.zeros_like(j)  # fixed as null
            else:
                raise ValueError(f"Unknown conditions found in x_conds: {i}")

        for t in tqdm.tqdm(ts, leave=False, desc="Estimating errors"):
            t_input = torch.tensor([t], device=device)
            x_noised = q_sample(x_orig, t_input)
            if "mask_all" in inspect.signature(diffusion_model.forward).parameters:
                raise NotImplementedError("Not tested yet")
                pred, mask = diffusion_model(x_empty, x_noised, timesteps=t_input,
                                             pe_input=None, conditions=conditions,
                                             mask=False, mask_all=inference_mask)
            else:
                pred, mask = diffusion_model(x_empty, x_noised, timesteps=t_input,
                                             pe_input=None, conditions=conditions,
                                             mask=inference_mask)

            # UPDATE: full recon eval instead to align with training obj
            # # Only evaluate performance on masked entries
            # pred = pred * mask
            # x_orig = x_orig * mask

            if loss == 'l2':
                error = F.mse_loss(x_orig, pred, reduction='none').mean(1)
            elif loss == 'l1':
                error = F.l1_loss(x_orig, pred, reduction='none').mean(1)
            elif loss == 'huber':
                error = F.huber_loss(x_orig, pred, reduction='none').mean(1)
            else:
                raise NotImplementedError(f"Unknown loss type {loss!r}")

            pred_errors += error.detach()

        return (pred_errors / len(ts)).cpu()


# TODO: check why performs bad
class CellJumpClassifier:
    """Diffusion classifier object.

    From a high level, the classifier works as follow

        1. Use all unique conditions as candidate.
        2. Evaluate reconstruction errors of cells with different candidate
           conditions.
        3. Select the top k conditions for each cell that resulted in the
           lowest errors and use these as the new candidate conditions.
           Note that k gradually decrease from round to round, and eventually
           drops to 1 in the final round to select the best matching condition
           for each cell.
        4. Repeat 2 and 3 until the last round as reached.

    Args:
        n_samples_list: List of samples to draw from in each round.
        to_kee_list: List of top conditions to keep for next round of
            evaluation.
        n_trials: Number of trials per sampled time points.
        loss: Type of loss to evaluate the error of the predictions given a
            particular condition.
        query_mode: What conditions to query for. "all" uses all possible
            combinations of the conditions, "seen" uses only the combinations
            seen during training, and "specified" uses the combinations passed
            to the :attr:`conditions` argument. "batch_all" and "batch_seen"
            are analogous to "all" and "seen" but only select from the batch.
        model: Model object to bind to. See
            :class:`scdiff.models.diffusion.ddpm.DDPM` for an example.

    Notes:
        There are several requirements for the binding model to work with the
        :class:`DiffusionClassifier` object.

            1. There should be a :attr:`diffusion_model` (or
               :attr:`model.diffusion_model`) attribute, which holds the
               denoising diffusion model.
            2. There should be a :attr:`num_timesteps` attribute about the
               maximum time step size of the diffusion model.
            3. There should be a :meth:`q_sample` method that performs forward
               sampling of the original data, i.e., adding noise.

    """

    def __init__(
        self,
        n_samples_list: List[int],
        to_keep_list: List[int],
        n_trials: int = 1,
        query_mode: CLF_QUERY_MODE = "all",
        inference_mask: bool = False,
        time_step_sampler: TS_SAMPLER_TYPE = "IterativeUniform",
        model: Optional[Any] = None,
    ):
        assert len(n_samples_list) == len(to_keep_list)
        assert to_keep_list[-1] == 1, "Last trial must only select one best matching condition."
        self.n_samples_list = n_samples_list
        self.to_keep_list = to_keep_list
        self.n_trials = n_trials
        self.query_mode = check_str_option("query_mode", query_mode, CLF_QUERY_MODE)
        self.inference_mask = inference_mask
        self._model = model

    @property
    def get_time_step_sampler(self):
        return self._time_step_sampler_cls

    @get_time_step_sampler.setter
    def get_time_step_sampler(self, val):
        if val == "IterativeUniform":
            self._time_step_sampler_cls = IterativeUniformTimeStepSampler
        elif val not in (opts := get_args(TS_SAMPLER_TYPE)):
            raise ValueError(f"Unknown time step sampler {val!r}, available options are {opts}")
        else:
            raise NotImplementedError(f"[DEVERROR] please implement {val}")

    @property
    def model(self):
        return self._model

    def __call__(
        self,
        x: torch.Tensor,
        x_conds: torch.Tensor,
        model: Optional[Any] = None,
        specified_conds: Optional[torch.Tensor] = None,
    ):
        """Predict conditions of each cell.

        Args:
            x_orig: Original expression values to be used for generating noised
                expression input to the diffusion model.
            x_conds: Conditions of all cells (i.e., rows of x_orig). These
                are the label to be predicted against.
            model: Model object to bind to. See
                :class:`scdiff.models.diffusion.ddpm.DDPM` for an example.
                Only needed if model was not binded during initialization.
            specified_conds: Specify the conditions to query for in "specified"
                query mode.

        Returns:
            Predicted conditions for each cell.

        """
        # Hook up with model
        if (model is None) and (self.model is None):
            raise ValueError("Model object not stored during init, please pass during call.")
        if (model is not None) and (self.model is not None):
            warnings.warn(
                f"Model object already specified during init: {self.model} but "
                f"also passed during call: {model}. Using the passed model "
                "in this call. Please remove duplicated model specification "
                "to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
        model = model or self.model
        diffusion_model, timesteps = self.get_assets_from_model(model)
        conditions = self._get_conditions(model, x_conds, specified_conds)

        # Set constants
        max_n_samples = max(self.n_samples_list)
        num_cells = len(x)
        num_conditions = len(conditions)
        num_t_splits = len(self.n_samples_list)
        n_samples_list = self.n_samples_list
        to_keep_list = self.to_keep_list

        if num_conditions < (max_to_keep := max(to_keep_list)):
            warnings.warn(
                f"Maximum conditions to keep ({max_to_keep}) exceeds the total "
                f"number of conditions available ({num_conditions}). Implicitly "
                f"setting max number of conditions to keep to {num_conditions}.",
                UserWarning,
                stacklevel=2,
            )
            to_keep_list = [min(num_conditions, i) for i in to_keep_list]

        # TODO: support other samplers
        time_step_sampler = IterativeUniformTimeStepSampler(
            timesteps,
            max_n_samples,
            self.n_trials,
        )

        full_error_tensor = torch.full((num_t_splits, num_cells, num_conditions), torch.nan)
        best_idxs = torch.arange(num_conditions).repeat(num_cells, 1)
        for i, (n_samples, n_to_keep) in enumerate(zip(n_samples_list, to_keep_list)):
            error_tensor = torch.zeros_like(best_idxs, dtype=torch.float)
            curr_t_to_eval = time_step_sampler(n_samples)

            # Perform one round of evaluation
            for j in tqdm.trange(
                best_idxs.shape[1],
                leave=False,
                desc=f"Round {i + 1} / {len(n_samples_list)}",
            ):
                query_conds = conditions[best_idxs[:, j]]
                error_tensor[:, j] = self.eval_error(model, x, curr_t_to_eval, query_conds,
                                                     self.inference_mask)

            # Aggregate evaluation results across rounds
            if i == 0:
                full_error_tensor[i] = error_tensor
                prev_size = len(curr_t_to_eval)
            else:
                curr_size = len(curr_t_to_eval)
                full_error_tensor[i].scatter_(dim=1, index=best_idxs, src=error_tensor)
                # Aggregate errors with previous runs
                full_error_tensor[i] = (
                    (full_error_tensor[i] * curr_size + full_error_tensor[i-1] * prev_size)
                    / (curr_size + prev_size)
                )
                prev_size += curr_size

            # Find best conditions for each cell
            new_best_vals, new_best_idxs = error_tensor.topk(n_to_keep, dim=1, largest=False)
            assert not torch.isnan(new_best_vals).any(), "Found nans in selected entries."
            # Convert idx back to original condition idx
            best_idxs = best_idxs.gather(dim=1, index=new_best_idxs)

        pred_conditions = conditions[best_idxs.flatten()]

        return pred_conditions

    def _get_conditions(
        self,
        model: torch.nn.Module,
        x_conds: torch.Tensor,
        specified_conds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # First prepare the unique conditions, either from the observations
        # from the batch, or from the full training dataset (recorded by the
        # unique_conditions attribute in the model object)
        if not self.query_mode.startswith("batch_"):
            unique_conditions = model.unique_conditions

        old_version_ckpt_flag = getattr(model, "unique_conditions") is None
        if self.query_mode.startswith("batch_") or old_version_ckpt_flag:
            if old_version_ckpt_flag:
                warnings.warn(
                    "The model ckpt being used is from an older version that "
                    "do not contain the 'unique_conditions'. Implicitly "
                    f"switching the query_mode from {self.query_mode} to "
                    f"batch_{self.query_mode}",
                    UserWarning,
                    stacklevel=2,
                )
            unique_conditions = x_conds.unique(dim=0)

        # The only difference is how we prepare the unique_conditions, which
        # we have handeled above
        query_mode = self.query_mode.replace("batch_", "")

        if query_mode == "all":
            individual_unique_conditions = [i.unique().tolist() for i in unique_conditions.T]
            return torch.tensor(
                list(itertools.product(*individual_unique_conditions)),
                device=unique_conditions.device,
            )
        elif query_mode == "seen":
            return unique_conditions
        elif query_mode == "specified":
            if specified_conds is None:
                raise ValueError("query_mode set to 'specified' but did not passed specified conditions")
            elif not isinstance(specified_conds, torch.Tensor):
                raise TypeError(f"Please pass specified contions as tensor, got {type(specified_conds)}")
            elif specified_conds.shape[1] != unique_conditions:
                raise ValueError(
                    f"Inconsistent condition type number. Got {specified_conds.shape[1]} "
                    f"conditions in the specified conditions, but model only recorded "
                    f"{unique_conditions.shape[1]} conditions.",
                )
            return specified_conds
        else:
            raise NotImplementedError(query_mode)

    @staticmethod
    def get_assets_from_model(model):
        if hasattr(model, "diffusion_model"):
            diffusion_model = model.diffusion_model
        else:
            diffusion_model = model.model.diffusion_model

        if hasattr(model, "num_timesteps"):
            timesteps = model.num_timesteps
        else:
            timesteps = model.timesteps

        return diffusion_model, timesteps

    @staticmethod
    @torch.inference_mode()
    def eval_error(
        model: torch.nn.Module,
        x_orig: torch.Tensor,
        ts: List[int],
        conditions: torch.Tensor,
        inference_mask: bool = False,
    ) -> torch.Tensor:
        device = x_orig.device
        pred_errors = torch.zeros(len(x_orig), device=device)

        for t in tqdm.tqdm(ts, leave=False, desc="Estimating errors"):
            t_input = torch.tensor([t], device=device)
            error = model.get_loss(x_orig, t_input, conditions=conditions, mask_flag=inference_mask, w_diff=0)
            pred_errors += error.detach()

        return (pred_errors / len(ts)).cpu()
