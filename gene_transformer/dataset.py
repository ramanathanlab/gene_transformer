from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch
from Bio import SeqIO  # type: ignore[import]
from mpire import WorkerPool
from mpire.utils import make_single_arguments
from natsort import natsorted
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from gene_transformer.config import PathLike


def group_by_kmer(s: SeqIO.SeqRecord, n: int) -> str:
    seq = str(s.seq)
    return " ".join(seq[i : i + n] for i in range(0, len(seq), n))


# TODO: We may need to revisit to see how the file system handles this.
class IndividualFastaDataset(Dataset):
    def __init__(
        self,
        fasta_dir: PathLike,
        block_size: int,
        tokenizer: PreTrainedTokenizerFast,
        kmer_size: int = 3,
        small_subset: int = 0,
    ):
        print("Individual fasta dataset")
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

        # TODO: Test if this caching mechanism solves the OOM
        # initialize reading from fasta files
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


class FASTADataset(Dataset):  # type: ignore[type-arg]
    def __init__(
        self,
        fasta_file: PathLike,
        block_size: int,
        tokenizer: PreTrainedTokenizerFast,
        kmer_size: int = 3,
    ) -> None:
        """PyTorch Dataset that tokenizes sequences by codon.

        Parameters
        ----------
        fasta_file : str
            Path to fasta file to read sequence from.
        block_size : int
            max_length of :obj:`tokenizer` encoder.
        tokenizer : PreTrainedTokenizerFast
            Converts raw strings to tokenized tensors.
        kmer_size : int
            Kmer length to tokenize by.
        """

        def _single_encode(sequence: SeqIO.SeqRecord) -> List[int]:
            return tokenizer.encode(  # type: ignore[no-any-return]
                group_by_kmer(sequence, kmer_size),
                # return_tensors="pt", # currently not returning torch tensors since it causes memory issues
                max_length=block_size,
                padding="max_length",
            )

        # Read in the sequences from the fasta file, convert to
        # codon string, tokenize, and collect in tensor
        print(f"Processing {fasta_file}...")
        parsed_seqs = list(SeqIO.parse(str(fasta_file), "fasta"))
        samples = []
        num_chunks = 50000
        for chunk in tqdm(list(chunks(parsed_seqs, num_chunks))):
            with WorkerPool(n_jobs=4) as pool:
                # need make_single_arguments otherwise map unpacks the seqs
                results = pool.map(
                    _single_encode,
                    make_single_arguments(chunk),
                    progress_bar=False,
                    iterable_len=num_chunks,
                )
                samples.extend(results)
        self.sequences = torch.Tensor(samples)
        print("Encoded all sequences.")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx].long()


def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
