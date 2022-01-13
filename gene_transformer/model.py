from transformers import TransfoXLLMHeadModel, TransfoXLConfig
from tokenizers import Tokenizer
# import pytorch_lightning as pl
# from transformers import PreTrainedTokenizerFast
# from aitextgen.TokenDataset import TokenDataset
# from torch.utils.data import DataLoader
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
# import torch
# import numpy as np
# from torch.utils.data import Subset
from transformers import AdamW
# from argparse import ArgumentParser
# from config import ModelSettings
# import os
# from utils import generate_dna_to_stop, generate_fasta_file, seqs_to_fasta
# from blast import BlastRun
# from tqdm import tqdm
# from pathlib import Path
# from Bio import SeqRecord
# import statistics
# from pytorch_lightning.utilities import rank_zero_only
#
# NUM_DATA_WORKERS = 4

import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# NEW
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from argparse import ArgumentParser
from config import ModelSettings
from aitextgen.TokenDataset import TokenDataset
import pickle

# from https://github.com/spellml/deeplab-voc-2012/blob/master/models/2_pytorch_distributed_model.py
def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)


def get_dataloader(config, rank, world_size):
    try:
        with open('/tmp/pickled_train_dataloader.pkl', 'rb') as f:
            dataloader = pickle.load(f)
            return dataloader
    except FileNotFoundError:
        dataset = TokenDataset(config.train_file, tokenizer_file=config.tokenizer_file, block_size=config.block_size)
        sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
        with open('/tmp/pickled_train_dataloader.pkl', 'wb') as f:
            b = pickle.dumps(dataloader)
            f.write(b)
        return dataloader


def get_model(config):
    if config.use_pretrained:
        model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
    else:
        base_config = TransfoXLConfig()
        model = TransfoXLLMHeadModel(base_config)
    return model


def train(rank, config, world_size):
    # NEW
    init_process(rank, world_size)
    print(f"Rank {rank}/{world_size} training process initialized.\n")

    # NEW
    # Since this is a single-instance multi-GPU training script, it's important that only one
    # process handle downloading of the data, to:
    #
    # * Avoid race conditions implicit in having multiple processes attempt to write to the same
    #   file simultaneously.
    # * Avoid downloading the data in multiple processes simultaneously.
    #
    # Since the data is cached on disk, we can construct and discard the dataloader and model in
    # the master process only to get the data. The other processes are held back by the barrier.
    if rank == 0:
        get_dataloader(config, rank, world_size)
        get_model(config)
    dist.barrier()
    print(f"Rank {rank}/{world_size} training process passed data download barrier.\n")

    model = get_model(config)
    model.cuda(rank)
    model.train()

    # NEW
    model = DistributedDataParallel(model, device_ids=[rank])

    dataloader = get_dataloader(config, rank, world_size)

    # since the background class doesn't matter nearly as much as the classes of interest to the
    # overall task a more selective loss would be more appropriate, however this training script
    # is merely a benchmark so we'll just use simple cross-entropy loss
    # criterion = nn.CrossEntropyLoss()

    # NEW
    # Since we are computing the average of several batches at once (an effective batch size of
    # world_size * batch_size) we scale the learning rate to match.
    optimizer = AdamW(model.parameters(), lr=5e-5 * world_size)

    # writer = SummaryWriter(f'/spell/tensorboards/model_2')

    for epoch in range(1, NUM_EPOCHS + 1):
        losses = []

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.cuda(rank)

            outputs = model(batch)
            loss = outputs.losses.mean()
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            # if rank == 0:
            #     writer.add_scalar('training loss', curr_loss)
            losses.append(curr_loss)

        # print(
        #     f'Finished epoch {epoch}, rank {rank}/{world_size}. '
        #     f'Avg Loss: {np.mean(losses)}; Median Loss: {np.min(losses)}.\n'
        # )

        if rank == 0 and epoch % 5 == 0:
            if not os.path.exists('/tmp/checkpoints/'):
                os.mkdir('/tmp/checkpoints/')
            torch.save(model.state_dict(), f'/spell/checkpoints/model_{epoch}.pth')
    torch.save(model.state_dict(), f'/spell/checkpoints/model_final.pth')


# NEW
NUM_EPOCHS = 1
WORLD_SIZE = torch.cuda.device_count()


def main(config):
    mp.spawn(train,
             args=(config, WORLD_SIZE),
             nprocs=WORLD_SIZE,
             join=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config = ModelSettings.from_yaml(args.config)
    main(config)


# class DNATransform(pl.LightningModule):
#     def __init__(self, config):
#         super(DNATransform, self).__init__()
#         self.config = config
#         self.batch_size = config.batch_size
#         self.tokenizer = Tokenizer.from_file(config.tokenizer_file)
#         self.fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
#         self.final_sequences = []
#         if config.small_subset:
#             self.train_dataset = Subset(TokenDataset(config.train_file, tokenizer_file=config.tokenizer_file,
#                                                      block_size=config.block_size), np.arange(5000))
#             self.val_dataset = Subset(TokenDataset(config.val_file, tokenizer_file=config.tokenizer_file,
#                                                    block_size=config.block_size), np.arange(1000))
#             self.test_dataset = Subset(TokenDataset(config.test_file, tokenizer_file=config.tokenizer_file,
#                                                     block_size=config.block_size), np.arange(1000))
#         else:
#             self.train_dataset = TokenDataset(config.train_file, tokenizer_file=config.tokenizer_file,
#                                               block_size=config.block_size)
#             self.val_dataset = Subset(TokenDataset(config.val_file, tokenizer_file=config.tokenizer_file,
#                                                    block_size=config.block_size), np.arange(1000))
#             self.test_dataset = Subset(TokenDataset(config.test_file, tokenizer_file=config.tokenizer_file,
#                                                     block_size=config.block_size), np.arange(1000))
#         # pdb.set_trace()
#         if config.use_pretrained:
#             self.model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
#         else:
#             base_config = TransfoXLConfig()
#             self.model = TransfoXLLMHeadModel(base_config)
#
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=NUM_DATA_WORKERS,
#                           prefetch_factor=4,
#                           pin_memory=True, persistent_workers=True, shuffle=True)
#
#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=NUM_DATA_WORKERS, prefetch_factor=4,
#                           pin_memory=True, persistent_workers=True, shuffle=False)
#
#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=NUM_DATA_WORKERS,
#                           prefetch_factor=4,
#                           pin_memory=True, persistent_workers=True, shuffle=False)
#
#     def forward(self, x):
#         return self.model(x, labels=x)
#
#     def training_step(self, batch, batch_idx):
#         x = batch
#         outputs = self(x)
#         loss = outputs.losses.mean()
#         # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         # self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         self.log("train/loss", loss)
#         # wandb.log({"train_loss": loss, 'random_value': 1})
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x = batch
#         outputs = self(x)
#         loss = outputs.losses.mean()
#         # self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         # wandb.log({"val_loss": loss})
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         x = batch
#         outputs = self(x)
#         loss = outputs.losses.mean()
#         # self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         # wandb.log({"test_loss": loss})
#         return loss
#
#     def configure_optimizers(self):
#         return AdamW(self.model.parameters(), lr=5e-5)
#         # return FusedAdam(self.parameters())
#
#     def validation_epoch_end(self, val_step_outputs):
#         """NOTE: BLAST must be installed locally in order for this to work properly."""
#         if not self.config.enable_blast:
#             return
#         # don't do anything to the validation step outputs, we're using this space to generate sequences and run blast
#         # in order to monitor the similarity to training sequences
#         generated = generate_dna_to_stop(self.model, self.fast_tokenizer, num_seqs=self.config.num_blast_seqs_per_gpu,
#                                          biopy_seq=False)
#         blast_scores = []
#         temp_fasta_dir = Path(
#             str(self.config.checkpoint_dir)
#             + "/blast_runs_globalstep{}/".format(self.global_step))
#         temp_csv_dir = temp_fasta_dir
#         try:
#             os.makedirs(temp_fasta_dir)
#         except FileExistsError:
#             pass
#
#         for n, sequence in tqdm(enumerate(generated)):
#             print("Blasting sequence {}...".format(sequence))
#             run = BlastRun(
#                 sequence,
#                 self.config.blast_validation_file,
#                 temp_fasta_dir=temp_fasta_dir,
#                 temp_csv_dir=temp_csv_dir
#             )
#             run.run_blast()
#             run.get_scores()
#             score = run.get_mean_score()
#             blast_scores.append(score)
#         # calculate mean and max score
#         mean_score = statistics.mean(blast_scores)
#         max_score = max(blast_scores)
#         self.log("val/mean_blast_score", float(mean_score), logger=True)
#         self.log("val/max_blast_score", float(max_score), logger=True)
#
#     def test_epoch_end(self, outputs):
#         if self.config.generate_upon_completion:
#             generated = generate_dna_to_stop(self.model, self.fast_tokenizer,
#                                              num_seqs=self.config.num_blast_seqs_per_gpu,
#                                              biopy_seq=True)
#             self.final_sequences.extend(generated)
#             # save_path = Path(self.config.checkpoint_dir) / Path("final_generated_sequences.fasta")
#             # seqs_to_fasta(generated, save_path)
#             # print("Saved final generated sequences to ", save_path)
#
#
# if __name__ == "__main__":
#     os.environ["TOKENIZERS_PARALLELISM"] = "true"
#     torch.set_num_threads(NUM_DATA_WORKERS)
#     pl.seed_everything(0)
#     parser = ArgumentParser()
#     parser.add_argument("-c", "--config", required=True)
#     args = parser.parse_args()
#     config = ModelSettings.from_yaml(args.config)
#     model = DNATransform(config)
#     if config.wandb_active:
#         print("Using Weights and Biases for logging...")
#         wandb_logger = WandbLogger(project=config.wandb_project_name)
#     else:
#         wandb_logger = None
#     checkpoint_callback = ModelCheckpoint(dirpath=config.checkpoint_dir,
#                                           every_n_train_steps=config.val_check_interval,
#                                           save_last=True, monitor="val/loss", mode="min",
#                                           filename='codon-transformer-{step:02d}-{val/loss:.2f}', verbose=True)
#     trainer = pl.Trainer(gpus=-1, default_root_dir=config.checkpoint_dir, strategy="ddp_spawn",
#                          callbacks=[checkpoint_callback], max_steps=config.training_steps, logger=wandb_logger,
#                          profiler="simple", val_check_interval=config.val_check_interval,
#                          accumulate_grad_batches=config.accumulate_grad_batches, num_sanity_val_steps=2)
#     trainer.fit(model)
#     trainer.test(model)
#     print("Completed training.")
#     if config.generate_upon_completion:
#         save_path = Path(config.checkpoint_dir) / Path("final_generated_sequences.fasta")
#         seqs = model.final_sequences
#         print("Length of final sequence list: ", len(seqs))
#         seqs_to_fasta(seqs, save_path)
#         print("Saved final generated sequences to ", save_path)
