import os
import warnings
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.lr_schedules import WarmupLR
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

from gene_transformer.blast import ParallelBLAST
from gene_transformer.config import ModelSettings, PathLike, throughput_config
from gene_transformer.dataset import FASTADataset, IndividualFastaDataset
from gene_transformer.utils import (
    ThroughputMonitor,
    generate_dna_to_stop,
    seqs_to_fasta,
    tokens_to_sequences,
)

SequenceDataset = Union[FASTADataset, IndividualFastaDataset]


class DNATransformer(pl.LightningModule):

    cfg: ModelSettings
    train_dataset: SequenceDataset
    val_dataset: SequenceDataset
    test_dataset: SequenceDataset

    def __init__(self, cfg: ModelSettings) -> None:
        super().__init__()
        self.save_hyperparameters(cfg.dict())
        self.cfg = cfg
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(str(self.cfg.tokenizer_file))
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        base_config = AutoConfig.from_pretrained(
            self.cfg.model_name,
            vocab_size=self.tokenizer.vocab_size,
            feed_forward_size=self.cfg.block_size,
            axial_pos_embds=False,
            # local_chunk_length=100,
            # lsh_attn_chunk_length=100,
            axial_pos_shape=(128, 94),
            # max_position_embeddings=cfg.block_size,
            # max_position_embeddings=self.cfg.block_size,
        )
        self.model = AutoModelForCausalLM.from_config(base_config)

        # To validate generated sequences
        # TODO: make sure temp files are outputting to node local
        if self.cfg.enable_blast:
            self.blast = ParallelBLAST(
                database_file=self.cfg.blast_validation_file,
                blast_dir=self.cfg.checkpoint_dir / "blast",
                blast_exe_path=self.cfg.blast_exe_path,
                num_workers=min(10, self.cfg.num_blast_seqs_per_gpu),
                node_local_path=self.cfg.node_local_path,
            )

        # Collect generated sequences at each epoch end
        self.final_sequences: Dict[str, List[str]] = {}

    def get_dataset(self, data_path: PathLike) -> SequenceDataset:
        """Helper function to generate dataset."""
        if self.cfg.genome_level:
            return IndividualFastaDataset(
                data_path,
                block_size=self.cfg.block_size,
                tokenizer=self.tokenizer,
                kmer_size=self.cfg.kmer_size,
                small_subset=self.cfg.small_subset,
            )
        return FASTADataset(
            data_path,
            block_size=self.cfg.block_size,
            tokenizer=self.tokenizer,
            kmer_size=self.cfg.kmer_size,
        )

    def get_dataloader(self, dataset: SequenceDataset, shuffle: bool) -> DataLoader:
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
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:  # type: ignore[override]
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log(
            "test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
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

    def validation_epoch_end(self, val_step_outputs: List[torch.FloatTensor]) -> None:  # type: ignore[override]
        # NOTE: BLAST must be installed locally in order for this to work properly.
        if not self.cfg.enable_blast:
            return

        # Generate sequences and run blast across all ranks,
        # then gather mean, max for logging on rank 0.

        # Don't do anything to the validation step outputs, we're using this
        # space to generate sequences and run blast in order to monitor the
        # similarity to training sequences
        tokens = generate_dna_to_stop(
            self.model,
            self.tokenizer,
            num_seqs=self.cfg.num_blast_seqs_per_gpu,
            max_length=self.cfg.block_size,
        )
        sequences = tokens_to_sequences(tokens, self.tokenizer)

        prefix = f"globalstep{self.global_step}"
        max_scores, mean_scores = self.blast.run(sequences, prefix)
        metrics = np.max(max_scores), np.mean(mean_scores)
        # Wait until all ranks meet up here
        self.trainer._accelerator_connector.strategy.barrier()  # type: ignore[attr-defined]
        metrics = self.all_gather(metrics)
        try:
            max_score, mean_score = metrics[0].max().cpu(), metrics[1].mean().cpu()
        except AttributeError as exc:
            # getting a weird numpy error when running validation on the protein sequences so catching here
            print("Attribute error when trying to move tensor to CPU... Error:", exc)
            max_score, mean_score = metrics[0].max(), metrics[1].mean()
        self.log("val/max_blast_score", max_score, logger=True, prog_bar=True)
        self.log("val/mean_blast_score", mean_score, logger=True, prog_bar=True)
        if self.trainer.is_global_zero:  # type: ignore[attr-defined]
            # Will move blast results (fasta and csv file) from the node
            # where rank-0 runs to the file system (will also move files
            # written by other ranks on the node)
            self.blast.backup_results()

    def test_epoch_end(self, outputs: List[torch.FloatTensor]) -> None:  # type: ignore[override]
        if not self.cfg.num_test_seqs_per_gpu:
            return None

        tokens = generate_dna_to_stop(
            self.model,
            self.tokenizer,
            num_seqs=self.cfg.num_test_seqs_per_gpu,
            max_length=self.cfg.block_size,
        )

        # Wait until all ranks meet up here
        self.trainer._accelerator_connector.strategy.barrier()  # type: ignore[attr-defined]
        # sequences after all_gather is shape (world_size, num_test_seqs_per_gpu, block_size)
        tokens = self.all_gather(tokens)

        if self.trainer.is_global_zero:  # type: ignore[attr-defined]
            # Concatenate over world size
            tokens = tokens.view(-1, self.cfg.block_size)
            sequences = tokens_to_sequences(tokens, self.tokenizer)
            # sequences = np.concatenate(sequences.cpu().numpy())
            print(f"sequences {len(sequences)}")
            self.final_sequences[f"globalstep{self.global_step}"] = sequences


def load_from_deepspeed(
    cfg: ModelSettings,
    checkpoint_dir: Path,
    checkpoint: str = "last.ckpt",
    model_weights: str = "last.pt",
) -> DNATransformer:
    """Utility function for deepspeed conversion"""
    # first convert the weights
    save_path = str(checkpoint_dir / checkpoint)
    output_path = str(checkpoint_dir / model_weights)
    # perform the conversion
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
    # load model
    model = DNATransformer.load_from_checkpoint(output_path, strict=False, cfg=cfg)
    return model  # type: ignore[no-any-return]


def train(cfg: ModelSettings) -> None:
    # Check if loading from checkpoint - this assumes that you're
    # loading from a sharded DeepSpeed checkpoint!!!
    if cfg.load_from_checkpoint_dir is not None:
        model = load_from_deepspeed(
            cfg=cfg, checkpoint_dir=cfg.load_from_checkpoint_dir
        )
        print(f"Loaded existing model at checkpoint {cfg.load_from_checkpoint_dir}....")
        try:
            model.model.lm_head.bias.data = torch.zeros_like(
                model.model.lm_head.bias.data
            )
        except Exception as exc:
            print("Couldn't set bias equal to zeros.", exc)
            pass
    else:
        model = DNATransformer(cfg)

    # Setup wandb
    wandb_logger = None
    if cfg.wandb_active:
        print("Using Weights and Biases for logging...")
        wandb_logger = WandbLogger(project=cfg.wandb_project_name)

    callbacks: List[Callback] = []
    callbacks.append(
        ModelCheckpoint(dirpath=cfg.checkpoint_dir, save_last=True, verbose=True)
    )

    if cfg.wandb_active:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

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
        check_val_every_n_epoch=cfg.check_val_every_n_epoch
        # plugins=[SLURMEnvironment(auto_requeue=False)]
    )
    trainer.fit(model)
    if cfg.compute_throughput:
        return

    # continue on if a normal training run - testing and inference mode
    trainer.test(model)

    if trainer.is_global_zero:
        print("Completed training.")

    if trainer.is_global_zero and cfg.num_test_seqs_per_gpu:
        save_path = cfg.checkpoint_dir / "generated"
        save_path.mkdir(exist_ok=True)
        for name, seqs in model.final_sequences.items():
            seqs_to_fasta(
                seqs, save_path / f"{name}.fasta", custom_seq_name=cfg.custom_seq_name
            )
        print(f"Saved final generated sequences to {save_path}")


class ModelLoadStrategy(ABC):
    @abstractmethod
    def get_model(self) -> DNATransformer:
        """Load and return a model object."""


class LoadDeepSpeedStrategy(ModelLoadStrategy):
    def __init__(self, cfg: ModelSettings) -> None:
        self.cfg = cfg

    def get_model(self) -> DNATransformer:
        if self.cfg.load_from_checkpoint_dir is None:
            raise ValueError("load_from_checkpoint_dir must be set in the config file")
        model = load_from_deepspeed(
            cfg=self.cfg, checkpoint_dir=self.cfg.load_from_checkpoint_dir
        )
        return model


class LoadPTCheckpointStrategy(ModelLoadStrategy):
    def __init__(self, cfg: ModelSettings, pt_file: str) -> None:
        self.cfg = cfg
        self.pt_file = pt_file

    def get_model(self) -> DNATransformer:
        model = DNATransformer.load_from_checkpoint(
            self.pt_file, strict=False, cfg=self.cfg
        )
        return model


def generate_embeddings(model: DNATransformer, dataloader: DataLoader) -> np.ndarray:
    """Output embedding array of shape (num_seqs, block_size, hidden_dim)."""
    embeddings = []
    for batch in tqdm(dataloader):
        batch = batch.cuda()
        outputs = model(batch, output_hidden_states=True)
        # outputs.hidden_states: (batch_size, sequence_length, hidden_size)
        embeddings.append(outputs.hidden_states[0].detach().cpu().numpy())

    embeddings = np.concatenate(embeddings)  # type: ignore
    return embeddings


# TODO: Make separate files for training and inference
def inference(
    model_load_strategy: ModelLoadStrategy,
    fasta_file: str,
    output_path: Optional[PathLike] = None,
) -> np.ndarray:
    """Output embedding array of shape (num_seqs, block_size, hidden_dim)."""
    model = model_load_strategy.get_model()
    model.cuda()
    dataset = model.get_dataset(fasta_file)
    dataloader = model.get_dataloader(dataset, shuffle=False)
    print(f"Running inference with dataset length {len(dataloader)}")
    embeddings = generate_embeddings(model, dataloader)
    print(f"Embeddings shape: {embeddings.shape}")
    if output_path:
        assert Path(output_path).suffix == ".npy"
        np.save(output_path, embeddings)
    return embeddings


def test(cfg: ModelSettings) -> None:
    """Run test dataset after loading from checkpoint"""
    model = LoadDeepSpeedStrategy(cfg).get_model()
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
    elif args.mode == "inference" and not config.compute_throughput:
        if not args.inference_fasta:
            raise ValueError("Must provide a fasta file to run inference on.")

        if args.inference_output_path.exists():
            raise FileExistsError(
                f"inference_output_path: {args.inference_output_path} already exists!"
            )

        if args.inference_model_load == "pt":
            model_strategy = LoadPTCheckpointStrategy(config, args.pt_file)
        elif args.inference_model_load == "deepspeed":
            model_strategy = LoadDeepSpeedStrategy(config)
        else:
            raise ValueError(
                f"Invalid inference_model_load {args.inference_model_load}"
            )
        inference(model_strategy, args.inference_fasta, args.inference_output_path)
