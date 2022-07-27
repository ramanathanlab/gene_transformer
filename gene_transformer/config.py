"""Model configuration."""
import json
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseSettings as _BaseSettings

_T = TypeVar("_T")

PathLike = Union[str, Path]


class BaseSettings(_BaseSettings):
    """Base settings to provide an easier interface to read/write YAML files."""

    def dump_yaml(self, cfg_path: PathLike) -> None:
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class WarmupLRSettings(BaseSettings):
    """Learning rate warm up settings."""

    min_lr: float = 5e-8
    """The starting learning rate."""
    num_steps: int = 50000
    """Steps to warm up for."""


class ModelSettings(BaseSettings):
    """Settings for the DNATransformer model."""

    # logging settings
    wandb_active: bool = True
    """Whether to use wandb for logging."""
    wandb_project_name: str = "codon_transformer"
    """Wandb project name to log to."""
    checkpoint_dir: Path = Path("codon_transformer")
    """Checkpoint directory to backup model weights."""
    node_local_path: Optional[Path] = None
    """A node local storage option to write temporary files to."""
    num_nodes: int = 1
    """The number of compute nodes used for training."""
    compute_throughput: bool = False
    """Flag for profiling - uses small subset to compute average throughput over 5 epochs after pinning."""
    profiling_path: Optional[Path] = None
    """Set to path if we want to run pytorch profiler"""

    # data settings
    tokenizer_file: Path = Path("tokenizer_files/codon_wordlevel_100vocab.json")
    """Path to the tokenizer file."""
    train_file: Path
    """Path to the training data."""
    val_file: Path
    """Path to the validation data."""
    test_file: Path
    """Path to the testing data."""
    kmer_size: int = 3
    """Size of kmer to use for tokenization."""
    genome_level: bool = False
    """Whether or not to use the genome-scale dataset class."""
    small_subset: int = 0
    """Only applies when :obj:`genome_level` is true. Uses the full dataset by default."""

    # blast settings
    enable_blast: bool = False
    """Whether or not to run BLAST during validation steps."""
    blast_validation_file: Path = Path("blast_file.fasta")
    """Path to fasta file to BLAST against."""
    num_blast_seqs_per_gpu: int = 5
    """Number of BLAST jobs per GPU/rank."""
    blast_exe_path: Path = Path("blastn")
    """Path to the BLAST executable, defaults to current conda environment."""

    # model settings
    model_name: str = "gpt2"
    """Name of the huggingface model to use."""
    batch_size: int = 8
    """Training micro-batch size."""
    epochs: int = 5
    """Number of training epochs."""
    block_size: int = 512
    """Block size to specify sequence length passed to the transformer."""
    accumulate_grad_batches: int = 1
    """Number of batches to accumulate before gradient updates."""
    learning_rate: float = 5e-5
    """Learning rate to use for training."""
    precision: int = 16
    """Training precision."""
    warm_up_lr: Optional[WarmupLRSettings] = None
    """If specified, will use a learning rate warmup scheduler."""
    load_from_checkpoint_dir: Optional[Path] = None
    """If specified, will load a model weight checkpoint to resume training from."""
    deepspeed_cfg_file: Optional[Path] = None
    """The deepspeed configuration file (currently unused)."""
    check_val_every_n_epoch: int = 1
    """Run validation every n number of epochs"""

    # generation settings
    num_test_seqs_per_gpu: int = 8
    """Number of sequences to generate per GPU when testing."""
    custom_seq_name: str = "SyntheticSeq"
    """Custum sequence name to write into fasta files for generate sequences."""

    # training ops (see PyTorch DataLoader for details.)
    num_data_workers: int = 4
    """Number of subprocesses to use for data loading."""
    prefetch_factor: int = 4
    """Number of batches loaded in advance by each worker."""
    pin_memory: bool = True
    """If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them."""
    persistent_workers: bool = True
    """If True, the data loader will not shutdown the worker processes after a dataset has been consumed once."""


def throughput_config(cfg: ModelSettings) -> ModelSettings:
    new_config = cfg.copy()
    new_config.epochs = 6
    new_config.check_val_every_n_epoch = 7
    new_config.num_test_seqs_per_gpu = 0
    new_config.small_subset = 10000
    new_config.profiling_path = None
    return new_config


if __name__ == "__main__":
    settings = ModelSettings(
        train_file=Path("train.fasta"),
        val_file=Path("val.fasta"),
        test_file=Path("test.fasta"),
    )
    settings.dump_yaml("settings_template.yaml")
