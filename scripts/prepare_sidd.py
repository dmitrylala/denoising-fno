#!/usr/bin/env -S uv run --script

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from denoising.configs import Environment

if TYPE_CHECKING:
    from asyncio.events import AbstractEventLoop


NOISY_PREFIX = 'NOISY'
GT_PREFIX = 'GT'

logger = logging.getLogger(__name__)


@dataclass
class ExtractOptions:
    input_folder: Path
    save_folder: Path
    img_list_path: Path
    save_csv_path: Path
    n_threads: int = 20
    compression_level: int = 3
    crop_size: int = 512
    step: int = 384
    thresh_size: int = 0


def worker(path: Path, opts: ExtractOptions) -> list[str]:
    """Worker for each process."""
    crop_size = opts.crop_size
    step = opts.step
    thresh_size = opts.thresh_size

    scene_name = path.parent.name
    extension = path.suffix
    keyword = path.name.split('_')[0]

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]

    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    saved_paths = []
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x : x + crop_size, y : y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            img_save_dir = opts.save_folder / scene_name
            img_save_dir.mkdir(exist_ok=True)
            img_name = f'{keyword}_patch{index:03d}{extension}'
            cv2.imwrite(
                img_save_dir / img_name,
                cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opts.compression_level],
            )
            saved_paths.append(str(Path(scene_name) / img_name))
    return saved_paths


def read_txt(path: str | Path) -> list[str]:
    with Path.open(path) as f:
        return [s.strip() for s in f.readlines()]


async def extract_subimages(opts: ExtractOptions) -> None:
    """Crop images to subimages."""
    if not opts.img_list_path.exists():
        msg = f"Image list {opts.img_list_path} doesn't exist"
        raise ValueError(msg)

    opts.save_folder.mkdir(exist_ok=True, parents=True)
    logger.info(f'Saving to: {opts.save_folder}')

    scene_names = read_txt(opts.img_list_path)
    logger.info(f'Got scenes: {len(scene_names)}')

    count_exist = 0
    existing_paths = []
    for scene_name in scene_names:
        for keyword in [NOISY_PREFIX, GT_PREFIX]:
            path = opts.input_folder / scene_name / f'{keyword}_SRGB_010.PNG'
            if path.exists():
                existing_paths.append(path)
                count_exist += 1
    logger.info(f'Existing image paths: {count_exist}')

    calls: list[partial[str]] = [partial(worker, path, opts) for path in existing_paths]
    loop: AbstractEventLoop = asyncio.get_running_loop()
    call_coros = []
    saved_paths: list[str] = []

    with ProcessPoolExecutor(opts.n_threads) as process_pool:
        for call in calls:
            call_coros.append(loop.run_in_executor(process_pool, call))  # noqa: PERF401

        pbar = tqdm(total=len(call_coros), unit='image', desc='Extract')
        for res in asyncio.as_completed(call_coros):
            saved_paths.extend(await res)
            pbar.update(1)
        pbar.close()
    logger.info('All processes done')

    noisy_paths = sorted(path for path in saved_paths if NOISY_PREFIX in path)
    gt_paths = sorted(path for path in saved_paths if GT_PREFIX in path)
    sample_list = pd.DataFrame({'noisy': noisy_paths, 'gt': gt_paths})
    sample_list.to_csv(opts.save_csv_path, index=False)
    logger.info(f'Got saved paths: {len(saved_paths)}, saved csv to {opts.save_csv_path}')


async def main() -> None:
    env = Environment(_env_file=Path(__file__).parent.parent / 'env')
    sidd_root = env.data / 'SIDD_Small_sRGB_Only'

    parts = ['train', 'val']

    for part in parts:
        opts = ExtractOptions(
            input_folder=sidd_root / 'Data',
            save_folder=sidd_root / part,
            img_list_path=sidd_root / f'Scene_Instances_{part}.txt',
            save_csv_path=sidd_root / f'patches_{part}.csv',
        )
        logger.info(f'Processing {part} part')
        try:
            await extract_subimages(opts)
        except Exception:
            logger.exception('Error extracting')
        else:
            logger.info('Success!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
