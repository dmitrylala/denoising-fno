from pathlib import Path

from .configs import FNODatasetConfig
from .utils import make_bsd_dset_config, make_fno_dset_config, make_load_params


def get_datasets_configs(data_dir: Path) -> dict[str, FNODatasetConfig]:
    mri_root = data_dir / 'MRI/IXI_0_1/255'
    bsd_root = data_dir / 'BSDS300-horizontal-synthetic'
    sidd_root = data_dir / 'SIDD_Small_sRGB_Only'

    mri_sketch_load_params = make_load_params('sketch', [145, 145], 'float32')
    mri_image_load_params = make_load_params('image', [145, 145], 'float32')
    mri_gt_load_params = make_load_params('gt', [255, 255], 'float32')
    pm_load_params = [mri_sketch_load_params, mri_image_load_params]
    gt_load_params = [mri_sketch_load_params, mri_gt_load_params]

    sidd_noisy_load_params = make_load_params('noisy', [512, 512, 3], 'uint8')
    sidd_gt_load_params = make_load_params('gt', [512, 512, 3], 'uint8')
    sidd_load_params = [sidd_noisy_load_params, sidd_gt_load_params]

    return {
        # MRI datasets
        'mri_pm_train': make_fno_dset_config(
            mri_root,
            data_dir / 'MRI/lists/IXI_0_1/train_pmLR_gibbsnoiseLR_train.csv',
            pm_load_params,
        ),
        'mri_pm_test': make_fno_dset_config(
            mri_root,
            data_dir / 'MRI/lists/IXI_0_1/train_pmLR_gibbsnoiseLR_val.csv',
            pm_load_params,
        ),
        'mri_gt_train': make_fno_dset_config(
            mri_root,
            data_dir / 'MRI/lists/IXI_0_1/train_gtLR_gibbsnoiseLR_train_train.csv',
            gt_load_params,
        ),
        'mri_gt_val': make_fno_dset_config(
            mri_root,
            data_dir / 'MRI/lists/IXI_0_1/train_gtLR_gibbsnoiseLR_train_val.csv',
            gt_load_params,
        ),
        'mri_gt_test': make_fno_dset_config(
            mri_root,
            data_dir / 'MRI/lists/IXI_0_1/train_gtLR_gibbsnoiseLR_val.csv',
            gt_load_params,
        ),
        # BSD datasets
        'bsd_synth_0.01_train': make_bsd_dset_config(bsd_root, 0.01, 'train'),
        'bsd_synth_0.01_test': make_bsd_dset_config(bsd_root, 0.01, 'test'),
        # SIDD datasets, patches
        'sidd_train': make_fno_dset_config(
            sidd_root / 'train',
            sidd_root / 'patches_train.csv',
            sidd_load_params,
            normalize=True,
        ),
        'sidd_test': make_fno_dset_config(
            sidd_root / 'val',
            sidd_root / 'patches_val.csv',
            sidd_load_params,
            normalize=True,
        ),
    }
