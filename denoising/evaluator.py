from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm

from .data import DatasetRegistry
from .data.utils import Batch
from .metrics import Metrics
from .models import ModelRegistry


class Evaluator:
    def __init__(
        self,
        models: ModelRegistry,
        datasets: DatasetRegistry,
        metrics: Metrics,
        device: str | torch.device = 'cpu',
    ) -> None:
        self.models = models
        self.datasets = datasets
        self.metrics = metrics
        self.device = device
        self._predicts_cache = {}

    def evaluate(
        self,
        model_names: list[str],
        dataset_name: str,
        batch_size: int = 128,
        skip_cache: bool = False,
    ) -> dict[str, float]:
        self._validate_eval_input(model_names, dataset_name)

        # instantiate Metric objects
        self.metrics.create(model_names)

        # make dataloaders for all predictors and infer models if needed
        predictions_dls = self._make_predictions_dls(
            model_names,
            dataset_name,
            batch_size,
            skip_cache,
        )

        # create dataloader for test dataset from registry
        dset_dl = self.make_dl_from_dset(dataset_name, batch_size)

        for name, preds_dl in predictions_dls.items():
            for pred_batch, dset_batch in zip(preds_dl, dset_dl, strict=False):
                # if using dataset as predictor
                if isinstance(pred_batch, Batch):
                    pred_batch = pred_batch.y  # noqa: PLW2901
                gt_batch = dset_batch.y

                # if gt_batch shape is bigger, downscale it to pred shape
                if pred_batch.size()[-1] < gt_batch.size()[-1]:
                    rs = Resize(size=pred_batch.size()[-1])
                    gt_batch = rs(gt_batch)
                self.metrics.update(name, pred_batch, gt_batch)

        # compute final float values from Metric objects
        return self.metrics.compute()

    @torch.no_grad()
    def inference(self, model_name: str, dataset_name: str, batch_size: int) -> torch.Tensor:
        model = self.models[model_name]
        model = model.to(self.device)
        dl = self.make_dl_from_dset(dataset_name, batch_size)

        res = []
        for batch in tqdm(dl):
            b = {k: v.to(self.device) for k, v in batch.items()}
            res.append(model(**b).to('cpu'))
        model = model.to('cpu')

        return torch.cat(res, dim=0)

    def make_dl_from_dset(self, name: str, batch_size: int) -> DataLoader:
        return self.datasets.make_dl(name, batch_size)

    def _make_predictions_dls(
        self,
        names: list[str],
        dataset_name: str,
        batch_size: int,
        skip_cache: bool = False,
    ) -> dict[str, DataLoader]:
        preds_dls = defaultdict(DataLoader)
        for name in names:
            # if name is in datasets, dataset.y will be used as predictions
            if name in self.datasets:
                preds_dls[name] = self.make_dl_from_dset(name, batch_size)
                continue

            key = (name, dataset_name)
            if key not in self._predicts_cache or skip_cache:
                print(f'Computing predicts for {key} from model')  # noqa: T201
                preds = self.inference(name, dataset_name, batch_size)
                self._predicts_cache[key] = preds
                print(f'Saved predicts for {key} to cache')  # noqa: T201
            else:
                print(f'Retrieving predicts for {key} from cache')  # noqa: T201

            preds_dls[name] = self._make_dl_from_cache(key, batch_size)
        return preds_dls

    def _make_dl_from_cache(self, key: tuple[str], batch_size: int) -> DataLoader:
        preds = self._predicts_cache[key]
        return DataLoader(preds, batch_size=batch_size, shuffle=False, pin_memory=True)

    def _validate_eval_input(self, model_names: list[str], dataset_name: str) -> None:
        if dataset_name not in self.datasets:
            msg = f"Can't find dataset {dataset_name} in dataset registry"
            raise KeyError(msg)

        for name in model_names:
            if name not in self.datasets and name not in self.models:
                msg = f"Can't find {name} in any registry"
                raise ValueError(msg)
