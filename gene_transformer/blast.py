"""Defining blast utilities to monitor training"""
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore[import]
import pytorch_lightning as pl
from Bio import SeqIO  # type: ignore[import]
from Bio.Seq import Seq  # type: ignore[import]
from Bio.SeqRecord import SeqRecord  # type: ignore[import]
from pytorch_lightning.callbacks import Callback

from gene_transformer.utils import generate_dna, tokens_to_sequences


class ParallelBLAST:
    """Class to handle blasting a group of sequences against
    a sequence database in parallel"""

    def __init__(
        self,
        database_file: Path,
        output_dir: Path,
        blast_exe_path: Path = Path("blastn"),
        num_workers: int = 1,
    ) -> None:
        """Runs BLAST using the blastn conda utility.

        Parameters
        ----------
        database_file : Path
            The fasta file containing sequences to blast against.
        output_dir : Path
            Output directory to write fasta files and blast csv files.
        blast_exe_path : Path, optional
            Path to blast executable, by default Path("blastn")
        num_workers : int, optional
            Number of threads used to spawn blast processes, by default 1
        """
        self.database_file = database_file
        self.output_dir = output_dir
        self.blast_exe_path = blast_exe_path
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

    def _blast(self, sequence: str, prefix: str) -> Tuple[float, float]:
        """Blast :obj:`sequence` agaisnt :obj:`database_file` using
        the blastn executable in a subprocess call.

        Parameters
        ----------
        sequence : str
            Sequence to blast.
        prefix : str
            Filename prefix for fasta file and blast csv file.

        Returns
        -------
        Tuple[float, float]
            Top score, mean score
        """

        # Write a temporary fasta file
        seq_hash = hash(sequence)
        temp_fasta = self.output_dir / f"{prefix}-seq-{seq_hash}.fasta"
        temp_csv = self.output_dir / f"{prefix}-blast-{seq_hash}.csv"
        SeqIO.write(SeqRecord(Seq(sequence)), temp_fasta, "fasta")
        print("running blast...")
        # Run local blastn given parameters in init, REQUIRES LOCAL INSTALLATION OF BLAST
        command = "{} -query {} -subject {} -out {} -outfmt 10".format(
            self.blast_exe_path, temp_fasta, self.database_file, temp_csv
        )
        subprocess.run(command, shell=True)

        try:
            # Read in csv where blast results were stored and take
            # column which specifies the scores
            scores = pd.read_csv(temp_csv, header=None)[11].values
        except pd.errors.EmptyDataError:
            print(f"WARNING: blast did not find a match {temp_csv}")
            sys.stdout.flush()
            # Remove files, no need to backup empty files
            temp_fasta.unlink()
            temp_csv.unlink()
            return -1.0, -1.0

        max_score: float = scores[0]
        mean_score: float = np.mean(scores)
        return max_score, mean_score

    def run(self, sequences: List[str], prefix: str) -> Tuple[List[float], List[float]]:
        max_scores, mean_scores = [], []
        futures = [self._executor.submit(self._blast, seq, prefix) for seq in sequences]
        for fut in futures:
            max_score, mean_score = fut.result()
            max_scores.append(max_score)
            mean_scores.append(mean_score)
        return max_scores, mean_scores


class BLASTCallback(Callback):
    """Custom callback in order to evaluate generated sequences with BLAST."""

    def __init__(
        self,
        block_size: int,
        database_file: Path,
        output_dir: Path,
        blast_exe_path: Path = Path("blastn"),
        num_blast_seqs_per_gpu: int = 1,
        node_local_path: Optional[Path] = None,
    ) -> None:
        """Generates sequences and BLAST's them against a database after each epoch.

        Parameters
        ----------
        block_size: int
            Block size to specify sequence length passed to the transformer.
        database_file : Path
            The fasta file containing sequences to blast against.
        output_dir : Path
            Output directory to write fasta files and blast csv files.
        blast_exe_path : Path, optional
            Path to blast executable, by default Path("blastn")
        num_blast_seqs_per_gpu : int, optional
            Number of sequences to generate and BLAST on each GPU, by default 1
        node_local_path : Optional[Path], by default None
            If passed, will write temporary blast files to this node local
            directory.
        """
        super().__init__()

        self.block_size = block_size
        self.num_blast_seqs_per_gpu = num_blast_seqs_per_gpu
        self.node_local_path = node_local_path

        # Default temp dir to the file system blast dir
        self.output_dir = output_dir
        self.temp_dir = output_dir

        if self.node_local_path is not None:
            self.temp_dir = self.node_local_path / "blast"
            self.temp_dir.mkdir(exist_ok=True)

            # Copy other files to node local storage
            database_file = self._copy_to_node_local(database_file)
            # Copy shared object file for blast
            # self._copy_to_node_local(blast_exe_path.parent / "libblastinput.so")
            # blast_exe_path = self._copy_to_node_local(blast_exe_path)

        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.blast = ParallelBLAST(
            database_file=database_file,
            output_dir=self.temp_dir,
            blast_exe_path=blast_exe_path,
            num_workers=num_blast_seqs_per_gpu,
        )

    def _copy_to_node_local(self, file: Path) -> Path:
        assert self.node_local_path is not None
        dst = self.node_local_path / file.name
        return dst if dst.exists() else shutil.copy(file, dst)

    def _backup_results(self) -> None:
        """Move node local files to :obj:`output_dir`."""
        if self.node_local_path is not None:
            # Bulk move of blast files
            command = f"mv {self.temp_dir / '*'} {self.output_dir}"
            subprocess.run(command, shell=True)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """BLAST generated sequences and collect statistics.

        Generate sequences and run blast across all ranks,
        then gather mean, max for logging on rank 0.
        """

        # Don't do anything to the validation step outputs, we're using this
        # space to generate sequences and run blast in order to monitor the
        # similarity to training sequences
        print("generating dna")
        sys.stdout.flush()
        tokens = generate_dna(
            pl_module.model,
            pl_module.tokenizer,
            num_seqs=self.num_blast_seqs_per_gpu,
            max_length=self.block_size,
        )
        sequences = tokens_to_sequences(tokens, pl_module.tokenizer)

        prefix = f"globalstep{pl_module.global_step}"
        print("dna generated")
        sys.stdout.flush()
        max_scores, mean_scores = self.blast.run(sequences, prefix)
        metrics = np.max(max_scores), np.mean(mean_scores)
        print("blast done")
        sys.stdout.flush()
        # Wait until all ranks meet up here
        trainer._accelerator_connector.strategy.barrier()
        print("barrier done")
        sys.stdout.flush()
        metrics = pl_module.all_gather(metrics)
        print("gather done")
        sys.stdout.flush()
        if trainer.is_global_zero:
            max_score, mean_score = metrics[0].max().item(), metrics[1].mean().item()
            # TODO: Test the above line and if it works, then remove the commented out code below
            # try:
            #     max_score, mean_score = metrics[0].max().cpu(), metrics[1].mean().cpu()
            # except AttributeError as exc:
            #     # getting a weird numpy error when running validation on the protein sequences so catching here
            #     print("Attribute error when trying to move tensor to CPU... Error:", exc)
            #     max_score, mean_score = metrics[0].max(), metrics[1].mean()

            pl_module.log("val/max_blast_score", max_score, prog_bar=True)
            pl_module.log("val/mean_blast_score", mean_score, prog_bar=True)
            # Will move blast results (fasta and csv file) from the node
            # where rank-0 runs to the file system (will also move files
            # written by other ranks on the node)
            self._backup_results()
