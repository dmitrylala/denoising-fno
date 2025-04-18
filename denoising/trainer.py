import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from timeit import default_timer

import optuna
import torch
from torch import nn

from .data.utils import Batch

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb

    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

__all__ = [
    'Trainer',
]


class Trainer:
    """
    Trainer class to train neural-operators on given datasets.

    .. note ::
        Trainer expects datasets to provide batches as key-value dictionaries, ex.:
        ``{'x': x, 'y': y}``, that are keyed to the arguments expected by models and losses.

    Parameters
    ----------
    model : nn.Module
    n_epochs : int
    wandb_log : bool, default is False
        whether to log results to wandb
    device : torch.device, or str
    eval_interval : int, default is 1
        how frequently to evaluate model and log training stats
    log_output : bool, default is False
        if True, and if wandb_log is also True, log output images to wandb
    verbose : bool, default is False

    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model: nn.Module,
        lr: float,
        n_epochs: int,
        wandb_log: bool = False,
        device: str = 'cpu',
        eval_interval: int = 1,
        log_output: bool = False,
        save_every: int | None = None,
        save_best: int | None = None,
        save_dir: str | Path = './ckpt',
        verbose: bool = False,
    ) -> None:
        self.model = model

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr / 10.0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30)

        # only log to wandb if a run is active
        self.wandb_log = False
        self.log_output = log_output
        if wandb_available:
            self.wandb_log = wandb_log and wandb.run is not None

        self.eval_interval = eval_interval
        self.verbose = verbose
        self.device = device

        # Track starting epoch for checkpointing/resuming
        self.start_epoch = 0
        self.n_epochs = n_epochs

        # attributes for checkpointing

        # if provided, interval at which to save checkpoints
        self.save_every = save_every

        # if provided, key of metric f"{loader_name}_{loss_name}"
        # to monitor and save model with best eval result
        # Overrides save_every and saves on eval_interval
        self.save_best = save_best

        # directory at which to save training states if
        # save_every and/or save_best is provided
        self.save_dir = save_dir

    def train(  # noqa: C901, D417, PLR0913
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loaders: dict[str, torch.utils.data.DataLoader],
        training_loss: Callable,
        eval_losses: dict[str, Callable] | None = None,
        trial=None,  # noqa: ANN001
        trial_obj: str = 'test_l2',
    ) -> dict[str, float]:
        """
        Trains the given model on the given dataset.

        If a device is provided, the model and data processor are loaded to device here.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        training_loss: function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        trial: optuna trial, optional

        Returns
        -------
        all_metrics: dict
            dictionary keyed f"{loader_name}_{loss_name}"
            of metric results for last validation epoch across
            all test_loaders

        """
        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, 'reduction') and training_loss.reduction == 'mean':
            warnings.warn(
                f'{training_loss.reduction=}. This means that the loss is '
                'initialized to average across the batch dim. The Trainer '
                'expects losses to sum across the batch dim.',
                stacklevel=3,
            )

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = {'l2': training_loss}

        # Load model to device
        self.model = self.model.to(self.device)

        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loaders:
                metrics.extend([f'{name}_{metric}' for metric in eval_losses])
            assert self.save_best in metrics, (  # noqa: S101
                f'Error: expected a metric of the form <loader_name>_<metric>, got {self.save_best}'
            )
            best_metric_value = float('inf')
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if self.verbose:
            print(f'Training on {len(train_loader.dataset)} samples')
            print(
                f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                f'         on resolutions {list(test_loaders)}.'
            )
            sys.stdout.flush()

        for epoch in range(self.start_epoch, self.n_epochs):
            train_err, avg_loss, epoch_train_time = self.train_one_epoch(
                epoch, train_loader, training_loss
            )
            epoch_metrics = {
                'train_err': train_err,
                'avg_loss': avg_loss,
                'epoch_train_time': epoch_train_time,
            }

            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(
                    epoch=epoch, eval_losses=eval_losses, test_loaders=test_loaders
                )

                # for optuna hyperparam optimization
                if trial is not None:
                    trial.report(float(eval_metrics[trial_obj]), epoch)

                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned

                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if self.save_best is not None and eval_metrics[self.save_best] < best_metric_value:
                    best_metric_value = eval_metrics[self.save_best]
                    self.checkpoint()

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None and epoch % self.save_every == 0:
                self.checkpoint()

        return epoch_metrics

    def train_one_epoch(
        self,
        epoch: int,
        train_loader: torch.utils.data.DataLoader,
        training_loss: Callable,
    ) -> dict[str, float | torch.Tensor]:
        """
        train_one_epoch trains self.model on train_loader for one epoch and returns training metrics.

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        training_loss: function
            cost function to minimize

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch

        """  # noqa: E501
        self.epoch = epoch
        avg_loss = 0
        self.model.train()
        t1 = default_timer()
        train_err = 0.0

        # track number of training examples in batch
        self.n_samples = 0

        for idx, sample in enumerate(train_loader):
            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1

        train_err /= len(train_loader)
        avg_loss /= self.n_samples

        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg['lr']
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch, time=epoch_train_time, avg_loss=avg_loss, train_err=train_err, lr=lr
            )

        return train_err, avg_loss, epoch_train_time

    def train_one_batch(
        self,
        idx: int,
        sample: Batch,
        training_loss: Callable,
    ) -> float | torch.Tensor:
        """
        Run one batch of input through model and return training loss on outputs.

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch
        training_loss: function
            cost function to minimize

        Returns
        -------
        loss: float | Tensor
            float value of training loss

        """
        self.optimizer.zero_grad(set_to_none=True)
        # load data to device
        sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        if isinstance(sample['y'], torch.Tensor):
            self.n_samples += sample['y'].shape[0]
        else:
            self.n_samples += 1

        out = self.model(**sample)

        if self.epoch == 0 and idx == 0 and self.verbose and isinstance(out, torch.Tensor):
            print(f'Raw outputs of shape {out.shape}')

        return training_loss(out, **sample)

    def log_training(
        self, epoch: int, time: float, avg_loss: float, train_err: float, lr: float | None = None
    ) -> None:
        """
        Basic method to log results from a single training epoch.

        Parameters
        ----------
        epoch: int
            training epoch index
        time: float
            training time of epoch
        avg_loss: float
            average train_err per individual sample
        train_err: float
            train error for entire epoch
        lr: float
            learning rate at current epoch

        """
        # accumulate info to log to wandb
        if self.wandb_log:
            values_to_log = {
                'train_err': train_err,
                'time': time,
                'avg_loss': avg_loss,
                'lr': lr,
            }

        msg = f'[{epoch}] time={time:.2f}, '
        msg += f'avg_loss={avg_loss:.4f}, '
        msg += f'train_err={train_err:.4f}'

        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=False)

    def evaluate_all(
        self,
        epoch: int,
        eval_losses: dict[str, Callable],
        test_loaders: dict[str, torch.utils.data.DataLoader],
    ) -> dict:
        # evaluate and gather metrics across each loader in test_loaders
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_metrics = self.evaluate(eval_losses, loader, log_prefix=loader_name)
            all_metrics.update(**loader_metrics)
        if self.verbose:
            self.log_eval(epoch=epoch, eval_metrics=all_metrics)
        return all_metrics

    def evaluate(
        self,
        loss_dict: dict[str, Callable],
        data_loader: torch.utils.data.DataLoader,
        log_prefix: str = '',
    ) -> dict:
        """
        Evaluates the model on a dictionary of losses.

        Parameters
        ----------
        loss_dict : dict of functions
            each function takes as input a tuple (prediction, ground_truth)
            and returns the corresponding loss
        data_loader : torch.utils.data.DataLoader
            data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict

        """
        # Ensure model and data processor are loaded to the proper device

        self.model = self.model.to(self.device)
        self.model.eval()

        errors = {f'{log_prefix}_{loss_name}': 0 for loss_name in loss_dict}

        # Warn the user if any of the eval losses is reducing across the batch
        for eval_loss in loss_dict.values():
            if hasattr(eval_loss, 'reduction') and eval_loss.reduction == 'mean':
                warnings.warn(
                    f'{eval_loss.reduction=}. This means that the loss is '
                    'initialized to average across the batch dim. The Trainer '
                    'expects losses to sum across the batch dim.',
                    stacklevel=3,
                )

        self.n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                return_output = False
                if idx == len(data_loader) - 1:
                    return_output = True
                eval_step_losses, outs = self.eval_one_batch(
                    sample, loss_dict, return_output=return_output
                )

                for loss_name, val_loss in eval_step_losses.items():
                    errors[f'{log_prefix}_{loss_name}'] += val_loss

        for key in errors:
            errors[key] /= self.n_samples

        # on last batch, log model outputs
        if self.log_output:
            errors[f'{log_prefix}_outputs'] = wandb.Image(outs)

        return errors

    def eval_one_batch(
        self,
        sample: dict,
        eval_losses: dict,
        return_output: bool = False,
    ) -> tuple[dict[str, float], torch.Tensor | None]:
        """
        eval_one_batch runs inference on one batch and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_output : bool
            whether to return model outputs for plotting
            by default False

        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs

        """
        # load data to device
        sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        self.n_samples += sample['y'].size(0)

        out = self.model(**sample)

        eval_step_losses = {}

        for loss_name, loss in eval_losses.items():
            val_loss = loss(out, **sample)
            eval_step_losses[loss_name] = val_loss

        if return_output:
            return eval_step_losses, out
        return eval_step_losses, None

    def log_eval(self, epoch: int, eval_metrics: dict) -> None:
        """
        log_eval logs outputs from evaluation on all test loaders to stdout and wandb.

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics collected during evaluation
            keyed f"{test_loader_name}_{metric}" for each test_loader

        """
        values_to_log = {}
        msg = ''
        for metric, value in eval_metrics.items():
            if isinstance(value, (float, torch.Tensor)):
                msg += f'{metric}={value:.4f}, '
            if self.wandb_log:
                values_to_log[metric] = value

        msg = 'Eval: ' + msg[:-2]  # cut off last comma+space
        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=True)

    def checkpoint(self) -> None:
        """Checkpoint saves current training state to a directory for resuming later. Only saves training state on the first GPU."""  # noqa: E501
        save_name = 'best_model' if self.save_best is not None else 'model'
        save_training_state(
            save_dir=self.save_dir,
            save_name=save_name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
        )
        if self.verbose:
            print(f'Saved training state to {self.save_dir}')


def save_training_state(  # noqa: D417, PLR0913
    save_dir: str | Path,
    save_name: str,
    model: nn.Module,
    optimizer: nn.Module | None = None,
    scheduler: nn.Module | None = None,
    epoch: int | None = None,
) -> None:
    """
    Save_training_state returns model and optional other training modules saved from prior training for downstream use.

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to load training state (model, optional optimizer, scheduler)
    save_name : str
        name of model to load

    """  # noqa: E501
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    manifest = {}

    model.save_checkpoint(save_dir, save_name)
    manifest['model'] = f'{save_name}_state_dict.pt'

    # save optimizer if state exists
    if optimizer is not None:
        optimizer_pth = save_dir / 'optimizer.pt'
        torch.save(optimizer.state_dict(), optimizer_pth)
        manifest['optimizer'] = 'optimizer.pt'

    if scheduler is not None:
        scheduler_pth = save_dir / 'scheduler.pt'
        torch.save(scheduler.state_dict(), scheduler_pth)
        manifest['scheduler'] = 'scheduler.pt'

    if epoch is not None:
        manifest['epoch'] = epoch

    torch.save(manifest, save_dir / 'manifest.pt')
