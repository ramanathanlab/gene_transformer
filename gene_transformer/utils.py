from Bio import SeqIO  # type: ignore[import]
from Bio.Seq import Seq  # type: ignore[import]
from Bio.SeqRecord import SeqRecord  # type: ignore[import]
import torch
from transformers import PreTrainedTokenizerFast

# from config import ModelSettings
# from model import DNATransform


# global variables
stop_codons = set("TAA", "TAG", "TGA")

def generate_dna_to_stop(
    model: torch.nn.Module,
    fast_tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
    top_k: int = 50,
    top_p: float = 0.95,
    num_seqs: int = 5,
    biopy_seq: bool = False,
):
    # generate the tokenized output
    output = model.generate(
        fast_tokenizer.encode("ATG", return_tensors="pt").cuda(),
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_seqs,
    )
    # convert from tokens to string
    seqs = [fast_tokenizer.decode(i, skip_special_tokens=True) for i in output]
    seq_strings = []
    for s in seqs:
        # break into codons
        dna = s.split(" ")
        # iterate through until you reach a stop codon
        for n, i in enumerate(dna):
            if i in stop_codons:
                break
        # get the open reading frame
        to_stop = dna[:n+1]
        # create the string and append to list
        seq_strings.append("".join(to_stop))
    # convert to biopython objects if requested
    if biopy_seq:
        seq_strings = [Seq(s) for s in seq_strings]
    return seq_strings


def seqs_to_fasta(seqs, file_name):
    records = [
        SeqRecord(
            i,
            id="MDH_SyntheticSeq_{}".format(seq),
            name="MDH_sequence",
            description="synthetic malate dehydrogenase",
        )
        for seq, i in enumerate(seqs)
    ]

    with open(file_name, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")


def generate_fasta_file(
    file_name,
    model,
    fast_tokenizer,
    max_length: int = 512,
    top_k: int = 50,
    top_p: float = 0.95,
    num_seqs: int = 5,
    translate_to_protein: bool = False,
):
    # generate seq objects
    generated = generate_dna_to_stop(
        model,
        fast_tokenizer,
        max_length=max_length,
        top_k=top_k,
        top_p=top_p,
        num_seqs=num_seqs,
        biopy_seq=True,
    )
    if translate_to_protein:
        generated = [s.translate() for s in generated]
    # generate seq records
    seqs_to_fasta(generated, file_name)
