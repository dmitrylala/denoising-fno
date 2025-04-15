from copy import deepcopy
from pathlib import Path

from denoising import make_model_config
from denoising.configs import ModelConfig


def get_model_configs(weights_dir: Path) -> dict[str, ModelConfig]:
    fno_cfg_mri = {
        'n_modes': (32, 32),
        'in_channels': 1,
        'hidden_channels': 32,
        'lifting_channel_ratio': 8,
        'projection_channel_ratio': 2,
        'out_channels': 1,
        'factorization': 'tucker',
        'n_layers': 4,
        'rank': 0.42,
    }

    fno_cfg_sidd = deepcopy(fno_cfg_mri)
    fno_cfg_sidd['in_channels'] = 3
    fno_cfg_sidd['out_channels'] = 3

    fno_cfg_sidd_more_layers = deepcopy(fno_cfg_sidd)
    fno_cfg_sidd_more_layers['n_layers'] = 16
    fno_cfg_sidd_more_layers['hidden_channels'] = 16

    fno_cfg_bsd = deepcopy(fno_cfg_mri)

    return {
        # models trained on MRI
        'mri-fno-neuralop': make_model_config(
            fno_cfg_mri,
            weights_dir / 'mri/run-6-weights.pt',
            'TFNO',
        ),
        'mri-fno-custom': make_model_config(
            fno_cfg_mri, weights_dir / 'mri/run-8-weights.pt', 'FNO'
        ),
        # trained in SIDD patches
        'sidd-fno-run2': make_model_config(
            fno_cfg_sidd,
            weights_dir / 'sidd_patches/run-2/model_state_dict.pt',
            'FNO',
        ),
        'sidd-fno-run3': make_model_config(
            fno_cfg_sidd,
            weights_dir / 'sidd_patches/run-3/model_state_dict.pt',
            'FNO',
        ),
        'sidd-fno-run4': make_model_config(
            fno_cfg_sidd_more_layers,
            weights_dir / 'sidd_patches/run-4-weights.pt',
            'FNO',
        ),
        # trained on BSD
        'bsd-fno': make_model_config(
            fno_cfg_bsd, weights_dir / 'bsd/run-2/model_state_dict.pt', 'FNO'
        ),
    }
