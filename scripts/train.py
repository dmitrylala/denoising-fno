#!/usr/bin/env -S uv run --script

from pathlib import Path

import torch
import wandb

from config import Environment, TrainConfig
from denoising.train import prepare_training
from denoising.utils import seed_everything

if __name__ == '__main__':
    cwd = Path.cwd()
    env = Environment(_env_file=cwd / 'env')
    wandb.login(key=env.wandb_api_key)

    run_idx = 30
    save_dir = cwd / 'notebooks/mri'

    cfg = TrainConfig(
        # Datasets params
        train_dset='mri_pm_train',
        test_dset='mri_pm_test',
        train_batch_size=128,
        test_batch_size=256,
        # Model params
        name_model='mri-hno',
        cfg_fno={
            'n_modes': (32, 32),
            'in_channels': 1,
            'hidden_channels': 32,
            'lifting_channel_ratio': 8,
            'projection_channel_ratio': 2,
            'out_channels': 1,
            'factorization': 'dense',
            'n_layers': 4,
            'rank': 0.42,
            'spectral': 'hartley',
        },
        # Run params
        random_seed=42,
        device='cuda:2',
        run_name=f'Run {run_idx}, HNO',
        save_weights_path=save_dir / f'run-{run_idx}-weights.pt',
        # Train params
        n_epochs=50,
        lr=1e-3,
        wandb_log=False,
        save_dir=save_dir / f'run-{run_idx}',
        verbose=True,
    )

    trainer, train_kwargs, run = prepare_training(env, cfg)

    seed_everything(cfg.random_seed)
    trainer.train(**train_kwargs)

    if run is not None:
        run.finish()

    torch.save(trainer.model.to('cpu').state_dict(), cfg.save_weights_path)
