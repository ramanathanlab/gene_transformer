from argparse import ArgumentParser
from pathlib import Path

from gene_transformer.dataset import write_individual_fasta_files

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--fasta", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("-n", "--num_workers", type=int, default=1)
    args = parser.parse_args()

    write_individual_fasta_files(args.fasta, args.output_dir, args.num_workers)
