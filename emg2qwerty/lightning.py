# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    TCNEncoder,  # safe to keep even if you don't use it
)
from emg2qwerty.transforms import Transform


# -----------------------------
# DataModule
# -----------------------------
class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
        train_fraction: float = 1.0,
        subset_seed: int = 0,
    ) -> None:
        super().__init__()
        self.window_length = window_length
        self.padding = padding
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.train_fraction = train_fraction
        self.subset_seed = subset_seed

    def setup(self, stage: str | None = None) -> None:
        # Train: windowed, jittered; your custom augmentation lives in data.py and is toggled here
        full_train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                    augment=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )

        if self.train_fraction < 1.0:
            n_total = len(full_train_dataset)
            n_keep = max(1, int(round(n_total * self.train_fraction)))
            rng = np.random.default_rng(self.subset_seed)
            indices = np.sort(rng.choice(n_total, size=n_keep, replace=False))
            self.train_dataset = Subset(full_train_dataset, indices.tolist())
        else:
            self.train_dataset = full_train_dataset

        print("TRAIN DATASET TYPE:", type(self.train_dataset))
        if hasattr(self.train_dataset, "datasets"):
            ds0 = self.train_dataset.datasets[0]
        else:
            ds0 = self.train_dataset

        print("TRAIN jitter:", ds0.jitter)
        print("TRAIN augment:", ds0.augment)

        # Val: windowed, no jitter/augment
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )

        # Test: whole session, no padding
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


# -----------------------------
# Shared CTC + metrics logic
# -----------------------------
class _BaseCTCModule(pl.LightningModule):
    def __init__(self, decoder: DictConfig) -> None:
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["inputs"]  # (T, N, ...)
        targets = batch["targets"]  # (S, N)
        input_lengths = batch["input_lengths"]  # (N,)
        target_lengths = batch["target_lengths"]  # (N,)
        N = int(input_lengths.shape[0])

        emissions = self.forward(inputs)  # (T_out, N, C)

        # Generic length correction (works for CNN encoders that shrink T, and for Transformer/TCN that keep T)
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        # Safety checks
        assert (emission_lengths > 0).all(), "Some emission lengths <= 0"
        assert (emission_lengths <= emissions.shape[0]).all(), "Some emission lengths > T_out"
        assert (target_lengths <= emission_lengths).all(), "Some targets longer than emissions"

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),  # (S, N) -> (N, S)
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        # Decode for CER metrics
        preds = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()

        for i in range(N):
            tgt = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=preds[i], target=tgt)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("val", batch)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step("test", batch)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# -----------------------------
# Baseline CNN: TDSConv + CTC
# -----------------------------
class TDSConvCTCModule(_BaseCTCModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__(decoder=decoder)
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )

        self.encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)       # (T, N, C)
        x = self.encoder(x)             # (T', N, C)
        emissions = self.classifier(x)  # (T', N, classes)
        return emissions


# -----------------------------
# CNN + BiGRU (or LSTM) + CTC
# -----------------------------
class TDSRNNCTCModule(_BaseCTCModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        rnn_type: str = "gru",          # "gru" or "lstm"
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        bidirectional: bool = True,
        rnn_dropout: float = 0.1,
        optimizer: DictConfig | None = None,
        lr_scheduler: DictConfig | None = None,
        decoder: DictConfig | None = None,
    ) -> None:
        assert decoder is not None, "decoder config must be provided"
        super().__init__(decoder=decoder)
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )

        self.encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )

        rnn_dropout_eff = rnn_dropout if rnn_layers > 1 else 0.0

        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=num_features,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                dropout=rnn_dropout_eff,
                bidirectional=bidirectional,
                batch_first=False,  # input is (T, N, C)
            )
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=num_features,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                dropout=rnn_dropout_eff,
                bidirectional=bidirectional,
                batch_first=False,  # input is (T, N, C)
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        rnn_out_features = rnn_hidden * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs).contiguous()   # (T, N, num_features)
        x = self.encoder(x).contiguous()         # (T', N, num_features)

        # Short/windowed batches: use normal GPU GRU
        if self.training or x.shape[0] < 50000:
            self.rnn.flatten_parameters()
            x, _ = self.rnn(x)
        else:
            # Full-session test fallback: run GRU on CPU to avoid cuDNN failure
            orig_device = x.device

            self.rnn.cpu()
            x_cpu = x.float().cpu()

            with torch.no_grad():
                x_cpu, _ = self.rnn(x_cpu)

            x = x_cpu.to(orig_device)
            self.rnn.to(orig_device)

        emissions = self.classifier(x)           # (T', N, num_classes)
        return emissions


# -----------------------------
# CNN + Transformer Encoder + CTC
# (defaults allow you to reuse baseline config; just override module._target_)
# -----------------------------
class TDSTransformerCTCModule(_BaseCTCModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        # Transformer params (all optional defaults)
        tfm_layers: int = 2,
        tfm_heads: int = 8,
        tfm_ff_mult: int = 4,
        tfm_dropout: float = 0.1,
        optimizer: DictConfig | None = None,
        lr_scheduler: DictConfig | None = None,
        decoder: DictConfig | None = None,
    ) -> None:
        assert decoder is not None, "decoder config must be provided"
        super().__init__(decoder=decoder)
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )

        self.encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=tfm_heads,
            dim_feedforward=tfm_ff_mult * num_features,
            dropout=tfm_dropout,
            activation="gelu",
            batch_first=False,  # expects (T, N, C)
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=tfm_layers)

        self.classifier = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)       # (T, N, C)
        x = self.encoder(x)             # (T', N, C)
        x = self.transformer(x)         # (T', N, C)
        emissions = self.classifier(x)  # (T', N, classes)
        return emissions


# -----------------------------
# Optional: TCN + CTC (keep if you want)
# -----------------------------
class TCNCTCModule(_BaseCTCModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        tcn_num_blocks: int,
        tcn_kernel_size: int,
        tcn_dropout: float,
        tcn_dilation_base: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__(decoder=decoder)
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )

        self.encoder = TCNEncoder(
            num_features=num_features,
            num_blocks=tcn_num_blocks,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
            dilation_base=tcn_dilation_base,
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)       # (T, N, C)
        x = self.encoder(x)             # (T, N, C)
        emissions = self.classifier(x)  # (T, N, classes)
        return emissions