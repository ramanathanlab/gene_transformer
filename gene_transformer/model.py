import os
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.lr_schedules import WarmupLR
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.profiler import PyTorchProfiler
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

from gene_transformer.blast import BLASTCallback
from gene_transformer.config import ModelSettings, PathLike, throughput_config
from gene_transformer.dataset import FastaDataset
from gene_transformer.utils import (
    LoadDeepSpeedStrategy,
    LoadPTCheckpointStrategy,
    ModelLoadStrategy,
    SequenceGenerationCallback,
    ThroughputMonitor,
)


class DNATransformer(pl.LightningModule):

    cfg: ModelSettings
    train_dataset: FastaDataset
    val_dataset: FastaDataset
    test_dataset: FastaDataset

    def __init__(self, cfg: ModelSettings) -> None:
        super().__init__()
        self.save_hyperparameters(cfg.dict())
        self.cfg = cfg
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(str(self.cfg.tokenizer_file))
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # loads from a json file like this: https://huggingface.co/google/reformer-enwik8/blob/main/config.json
        self.base_config = AutoConfig.from_pretrained(self.cfg.model_config_json)
        self.model = AutoModelForCausalLM.from_config(self.base_config)

    # def configure_sharded_model(self):
    #     self.model = AutoModelForCausalLM.from_config(self.base_config)

    def get_dataset(self, data_path: PathLike) -> FastaDataset:
        """Helper function to generate dataset."""
        return FastaDataset(
            data_path,
            block_size=self.cfg.block_size,
            tokenizer=self.tokenizer,
            kmer_size=self.cfg.kmer_size,
            small_subset=self.cfg.small_subset,
        )

    def get_dataloader(self, dataset: FastaDataset, shuffle: bool) -> DataLoader:
        """Helper function to generate dataloader."""
        return DataLoader(
            dataset,
            shuffle=shuffle,
            drop_last=True,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_data_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        self.train_dataset = self.get_dataset(self.cfg.train_file)
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        self.val_dataset = self.get_dataset(self.cfg.val_file)
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        self.test_dataset = self.get_dataset(self.cfg.test_file)
        return self.get_dataloader(self.test_dataset, shuffle=False)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> GPT2DoubleHeadsModelOutput:  # type: ignore[override]
        return self.model(x, labels=x, **kwargs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:  # type: ignore[override]
        outputs = self(batch)
        loss = outputs.loss
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> DeepSpeedCPUAdam:
        # optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.cfg.learning_rate)
        optimizer = FusedAdam(self.parameters(), lr=self.cfg.learning_rate)
        if self.cfg.warm_up_lr is not None:
            scheduler = WarmupLR(
                optimizer,
                warmup_min_lr=self.cfg.warm_up_lr.min_lr,
                warmup_max_lr=self.cfg.learning_rate,
                warmup_num_steps=self.cfg.warm_up_lr.num_steps,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric) -> None:
        scheduler.step()


def train(cfg: ModelSettings) -> None:
    if cfg.load_from_checkpoint_pt is not None:
        load_strategy = LoadPTCheckpointStrategy(cfg.load_from_checkpoint_pt, cfg=cfg)
        model = load_strategy.get_model(DNATransformer)
    elif cfg.load_from_checkpoint_dir is not None:
        # Check if loading from checkpoint - this assumes that you're
        # loading from a sharded DeepSpeed checkpoint!!!
        load_strategy = LoadDeepSpeedStrategy(cfg.load_from_checkpoint_dir, cfg=cfg)
        model = load_strategy.get_model(DNATransformer)
        print(f"Loaded existing model at checkpoint {cfg.load_from_checkpoint_dir}....")
    else:
        model = DNATransformer(cfg)

    # Setup wandb
    wandb_logger = None
    if cfg.wandb_active:
        print("Using Weights and Biases for logging...")
        wandb_logger = WandbLogger(project=cfg.wandb_project_name)

    callbacks: List[Callback] = []
    if cfg.checkpoint_dir is not None:
        callbacks.append(
            ModelCheckpoint(dirpath=cfg.checkpoint_dir, save_last=True, verbose=True)
        )

    if cfg.wandb_active:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    if cfg.enable_blast:
        assert cfg.checkpoint_dir is not None
        callbacks.append(
            BLASTCallback(
                block_size=cfg.block_size,
                database_file=cfg.blast_validation_file,
                output_dir=cfg.checkpoint_dir / "blast",
                blast_exe_path=cfg.blast_exe_path,
                num_blast_seqs_per_gpu=cfg.num_blast_seqs_per_gpu,
                node_local_path=cfg.node_local_path,
            )
        )

    if cfg.num_test_seqs_per_gpu:
        assert cfg.checkpoint_dir is not None
        callbacks.append(
            SequenceGenerationCallback(
                block_size=cfg.block_size,
                num_test_seqs_per_gpu=cfg.num_blast_seqs_per_gpu,
                output_dir=cfg.checkpoint_dir / "generated",
                custom_seq_name=cfg.custom_seq_name,
            )
        )

    if cfg.compute_throughput:
        # Remove other callbacks
        callbacks = [ThroughputMonitor(cfg.batch_size, cfg.num_nodes)]

    profiler = None
    if cfg.profiling_path:
        profiler = PyTorchProfiler(
            dirpath=cfg.profiling_path,
            profiler_kwargs={
                "activities": [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                "schedule": torch.profiler.schedule(wait=0, warmup=1, active=3),
                "on_trace_ready": torch.profiler.tensorboard_trace_handler("./"),
            },
        )

    trainer = pl.Trainer(
        # use all available gpus
        gpus=-1,
        default_root_dir=str(cfg.checkpoint_dir),
        # Use NVMe offloading on other clusters see more here:
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed-infinity-nvme-offloading
        strategy=DeepSpeedPlugin(
            stage=3,
            # offload_optimizer=True,
            # offload_parameters=True,
            # remote_device="cpu",
            # offload_params_device="cpu",
            # offload_optimizer_device="nvme",
            # nvme_path="/tmp",
            logging_batch_size_per_gpu=cfg.batch_size,
            # add the option to load a config from json file with more deepspeed options
            # note that if supplied all defaults are ignored - model settings defaults this arg to None
            # config=cfg.deepspeed_cfg_file
        ),
        callbacks=callbacks,
        # max_steps=cfg.training_steps,
        logger=wandb_logger,
        profiler=profiler,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        num_sanity_val_steps=0,
        precision=cfg.precision,
        max_epochs=cfg.epochs,
        num_nodes=cfg.num_nodes,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        # plugins=[SLURMEnvironment(auto_requeue=False)]
    )

    trainer.fit(model)
    if cfg.compute_throughput:
        return

    # continue on if a normal training run - testing and inference mode
    trainer.test(model)

    if trainer.is_global_zero:
        print("Completed training.")


def generate_embeddings(
    model: DNATransformer, dataloader: DataLoader, compute_mean: bool = False
) -> np.ndarray:
    """Output embedding array of shape (num_seqs, block_size, hidden_dim)."""
    embeddings = []
    for batch in tqdm(dataloader):
        batch = batch.cuda()
        outputs = model(batch, output_hidden_states=True)
        # outputs.hidden_states: (batch_size, sequence_length, hidden_size)
        emb = outputs.hidden_states[0].detach().cpu().numpy()
        if compute_mean:
            # Compute average over sequence length
            emb = np.mean(emb, axis=1)
        embeddings.append(emb)

    embeddings = np.concatenate(embeddings)  # type: ignore
    return embeddings


# TODO: Make separate files for training and inference
def inference(
    model_load_strategy: ModelLoadStrategy,
    fasta_file: str,
    output_path: Optional[PathLike] = None,
    compute_mean: bool = False,
) -> np.ndarray:
    """Output embedding array of shape (num_seqs, block_size, hidden_dim)."""
    model: DNATransformer = model_load_strategy.get_model(DNATransformer)
    model.cuda()
    dataset = model.get_dataset(fasta_file)
    dataloader = model.get_dataloader(dataset, shuffle=False)
    print(f"Running inference with dataset length {len(dataloader)}")
    embeddings = generate_embeddings(model, dataloader, compute_mean)
    print(f"Embeddings shape: {embeddings.shape}")
    if output_path:
        assert Path(output_path).suffix == ".npy"
        np.save(output_path, embeddings)
    return embeddings


def test(cfg: ModelSettings) -> None:
    """Run test dataset after loading from checkpoint"""
    if cfg.load_from_checkpoint_dir is None:
        raise ValueError("load_from_checkpoint_dir must be set in the config file")
    load_strategy = LoadDeepSpeedStrategy(cfg.load_from_checkpoint_dir, cfg=cfg)
    model = load_strategy.get_model(DNATransformer)
    model.cuda()

    trainer = pl.Trainer(
        gpus=-1,
        default_root_dir=str(cfg.checkpoint_dir),
        strategy=DeepSpeedPlugin(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            remote_device="cpu",
            offload_params_device="cpu",
            logging_batch_size_per_gpu=cfg.batch_size,
        ),
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        num_sanity_val_steps=2,
        precision=cfg.precision,
        max_epochs=cfg.epochs,
        num_nodes=cfg.num_nodes,
    )

    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--inference_fasta", default="")
    parser.add_argument("--inference_mean", action="store_true")
    parser.add_argument("--inference_model_load", default="pt", help="deepspeed or pt")
    parser.add_argument(
        "--inference_pt_file",
        help="Path to pytorch model weights if inference_model_load==pt",
    )
    parser.add_argument(
        "--inference_output_path", default="./embeddings.npy", type=Path
    )
    args = parser.parse_args()
    config = ModelSettings.from_yaml(args.config)

    # Setup torch environment
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.set_num_threads(config.num_data_workers)  # type: ignore[attr-defined]
    pl.seed_everything(0)

    # check if we're computing throughput - this means a new config with specific settings - default is false
    if config.compute_throughput:
        warnings.warn(
            "You are running in compute throughput mode - running for 6 epochs to compute samples per second. "
            "No validation or test sets run. No model checkpointing."
        )
        # new config definition
        config = throughput_config(config)

    if args.mode == "train":
        train(config)
    elif args.mode == "test":
        test(config)
    elif args.mode == "inference" and not config.compute_throughput:
        if not args.inference_fasta:
            raise ValueError("Must provide a fasta file to run inference on.")

        if args.inference_output_path.exists():
            raise FileExistsError(
                f"inference_output_path: {args.inference_output_path} already exists!"
            )

        if args.inference_model_load == "pt":
            model_strategy = LoadPTCheckpointStrategy(args.pt_file, cfg=config)
        elif args.inference_model_load == "deepspeed":
            if config.load_from_checkpoint_dir is None:
                raise ValueError(
                    "load_from_checkpoint_dir must be set in the config file"
                )
            model_strategy = LoadDeepSpeedStrategy(
                config.load_from_checkpoint_dir, cfg=config
            )
        else:
            raise ValueError(
                f"Invalid inference_model_load {args.inference_model_load}"
            )
        inference(
            model_strategy,
            args.inference_fasta,
            args.inference_output_path,
            args.inference_mean,
        )
