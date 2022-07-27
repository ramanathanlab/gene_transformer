from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict

import torch
from Bio import SeqIO  # type: ignore[import]
from natsort import natsorted
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from gene_transformer.config import PathLike


def group_by_kmer(s: SeqIO.SeqRecord, n: int) -> str:
    seq = str(s.seq)
    return " ".join(seq[i : i + n] for i in range(0, len(seq), n))


def _write_fasta_file(seq: SeqIO.SeqRecord, output_file: Path) -> None:
    SeqIO.write(seq, str(output_file), "fasta")


def write_individual_fasta_files(
    fasta_file: Path, output_dir: Path, num_workers: int = 1
) -> None:
    output_dir.mkdir(exist_ok=True)
    seqs = list(SeqIO.parse(fasta_file, "fasta"))
    output_files = [output_dir / f"sequence-{i}.fasta" for i in range(len(seqs))]
    print(f"Number of sequences: {len(seqs)}")
    chunksize = max(1, len(seqs) // num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in executor.map(
            _write_fasta_file, seqs, output_files, chunksize=chunksize
        ):
            pass


class FastaDataset(Dataset):
    def __init__(
        self,
        fasta_dir: PathLike,
        block_size: int,
        tokenizer: PreTrainedTokenizerFast,
        kmer_size: int = 3,
        small_subset: int = 0,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.kmer_size = kmer_size

        self.files = natsorted(Path(fasta_dir).glob("*.fasta"))

        # default of zero will not call this logic
        if small_subset:
            self.files = self.files[:small_subset]

        self.pad_sequence = partial(
            torch.nn.functional.pad, value=tokenizer.pad_token_id
        )

        # Cache the samples in memory
        self.samples: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # tokenize on the fly
        try:
            return self.samples[idx]
        except KeyError:
            sequence = list(SeqIO.parse(self.files[idx], "fasta"))[0]
            encoded_sequence = torch.Tensor(
                self.tokenizer.encode(
                    group_by_kmer(sequence, self.kmer_size),
                    max_length=self.block_size,
                    padding="max_length",
                )
            ).long()
            self.samples[idx] = encoded_sequence
            return encoded_sequence
