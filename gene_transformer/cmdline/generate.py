from argparse import ArgumentParser
from pathlib import Path

from gene_transformer.config import ModelSettings
from gene_transformer.model import DNATransformer
from gene_transformer.utils import (
    LoadDeepSpeedStrategy,
    non_redundant_generation,
    seqs_to_fasta,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-o", "--output_fasta", type=Path, required=True)
    parser.add_argument("-n", "--num_seqs", type=int, required=True)
    parser.add_argument("-s", "--name_prefix", type=str, default="SyntheticSeq")
    args = parser.parse_args()

    # Load the model settings file
    config = ModelSettings.from_yaml(args.config)

    if config.load_from_checkpoint_dir is None:
        raise ValueError("load_from_checkpoint_dir must be set in the config file")

    # Load the model into GPU memory
    load_strategy = LoadDeepSpeedStrategy(config.load_from_checkpoint_dir, cfg=config)
    model = load_strategy.get_model(DNATransformer)
    model.cuda()

    # Generate sequences using the model
    results = non_redundant_generation(model, model.tokenizer, num_seqs=args.num_seqs)
    unique_seqs, all_seqs = results["unique_seqs"], results["all_generated_seqs"]
    print(f"Proportion of unique seqs: {len(unique_seqs) / len(all_seqs)}")

    # Write fasta with unique sequences to disk
    seqs_to_fasta(unique_seqs, args.output_fasta, custom_seq_name=args.name_prefix)
