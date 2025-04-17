import wandb
from neuralop.losses import H1Loss, LpLoss

from config import (
    Environment,
    TrainConfig,
    get_datasets_configs,
    get_model_configs,
    make_model_config,
    make_trainer_config,
    make_wandb_config,
)

from .data import DatasetRegistry
from .models import ModelRegistry
from .trainer import Trainer

__all__ = [
    'prepare_training',
]


def prepare_training(env: Environment, cfg: TrainConfig) -> tuple:
    # load datasets
    dataset_registry = DatasetRegistry()
    dataset_registry.load(get_datasets_configs(env.data), verbose=cfg.verbose)

    train_loader = dataset_registry.make_dl(
        cfg.train_dset, batch_size=cfg.train_batch_size, shuffle=True
    )
    test_loader = dataset_registry.make_dl(cfg.test_dset, batch_size=cfg.test_batch_size)

    # check dataloader work
    for batch in train_loader:
        x_size, y_size = batch.x.size(), batch.y.size()
        if cfg.verbose:
            print(x_size, y_size)  # noqa: T201
        break

    load_models_kwargs = {
        'random_seed': cfg.random_seed,
        'device': cfg.device,
        'verbose': cfg.verbose,
    }
    model_registry = ModelRegistry()
    if cfg.verbose:
        # load existing models
        model_registry.load(get_model_configs(env.weights), **load_models_kwargs)

    # create and get new model
    new_model = {cfg.name_model: make_model_config(cfg.cfg_fno, None, cfg.cls_model)}
    model_registry.load(new_model, **load_models_kwargs)
    model = model_registry[cfg.name_model]

    # init wandb run if needed
    run = None
    if cfg.wandb_log:
        wandb_cfg = make_wandb_config(run_name=cfg.run_name, tags=['MRI', 'no augs'])
        run = wandb.init(**wandb_cfg.model_dump())

    # create trainer
    trainer_cfg = make_trainer_config(
        cfg.n_epochs,
        cfg.lr,
        cfg.wandb_log,
        cfg.save_dir,
        cfg.verbose,
    )
    trainer = Trainer(model=model, device=cfg.device, **trainer_cfg.model_dump())
    print(f'Logging to wandb enabled: {trainer.wandb_log}')  # noqa: T201

    h1loss = H1Loss(d=2)
    l2loss = LpLoss(d=2, p=2)

    train_kwargs = {
        'train_loader': train_loader,
        'test_loaders': {'test': test_loader},
        'training_loss': h1loss,
        'eval_losses': {'h1': h1loss, 'l2': l2loss},
    }

    return trainer, train_kwargs, run
