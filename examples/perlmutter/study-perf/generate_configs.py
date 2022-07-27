from pathlib import Path
from argparse import ArgumentParser
from itertools import product
from gene_transformer.config import ModelSettings


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=Path,
        required=True,
        help="Directory to write config files to.",
    )
    parser.add_argument("--tokenizer_file", type=Path, required=True)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--val_file", type=Path, required=True)
    parser.add_argument("--test_file", type=Path, required=True)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    args.config_dir.mkdir(exist_ok=True)

    model_names = ["reformer"]  # "gpt-neox"]
    num_nodes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    num_params = ["25M", "250M", "2.5B", "20B"]
    block_sizes = [2048, 10240]  # protein vs genome scale

    model_architectures = {
        "reformer": {
            "25M": "architectures/reformer_24,313,413.json",
            "250M": "architectures/reformer_243,038,277.json",
            "2.5B": "architectures/reformer_2,551,869,509.json",
            "20B": "architectures/reformer_20,405,579,845.json",
        },
        "gpt-neox": {
            "25M": "architectures/neox_25,351,168.json",
            "250M": "architectures/neox_260,668,800.json",
            "2.5B": "architectures/neox_2,417,053,728.json",
            "20B": "architectures/neox_9,558,835,200.json",  # TODO: Update to 20B
        },
    }

    experiment_combinations = list(
        product(model_names, num_nodes, num_params, block_sizes)
    )

    for experiment in experiment_combinations:
        model_name, nodes, params, block_size = experiment
        experiment_name = f"{model_name}_{nodes}nodes_{params}_{block_size}"
        print(experiment_name)
        config = ModelSettings(
            checkpoint_dir=None,
            node_local_path=Path("/tmp"),
            num_nodes=nodes,
            compute_throughput=True,
            tokenizer_file=args.tokenizer_file,
            train_file=args.train_file,
            val_file=args.val_file,
            test_file=args.test_file,
            model_config_json=Path(model_architectures[model_name][params]),
            batch_size=4,  # TODO: Set based on max size possible
            block_size=block_size,
            num_test_seqs_per_gpu=0,
        )
        config_path = args.config_dir / f"{experiment_name}.yaml"
        config.dump_yaml(config_path)


if __name__ == "__main__":
    main()
