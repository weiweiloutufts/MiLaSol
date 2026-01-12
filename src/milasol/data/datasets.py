import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import random

# Define amino acid mapping
aa_vocab = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "-": 0,
    "X": 21,
}


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def random_deletion(seq, p=0.05):
    return "".join([aa for aa in seq if random.random() > p])


def random_substitution(seq, p=0.05):
    return "".join(
        [aa if random.random() > p else random.choice(AMINO_ACIDS) for aa in seq]
    )


def random_masking(seq, p=0.05, mask_token="X"):
    return "".join([aa if random.random() > p else mask_token for aa in seq])


def collate_fn(batch):
    sequences, esm_embs, prot_embs, raygun_embs, solu_feats, labels, idx = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    max_len = 1241
    if max_len is not None:
        if sequences_padded.size(1) < max_len:
            pad_size = max_len - sequences_padded.size(1)
            padding = torch.full(
                (sequences_padded.size(0), pad_size), 0, dtype=sequences_padded.dtype
            )
            sequences_padded = torch.cat([sequences_padded, padding], dim=1)
        elif sequences_padded.size(1) > max_len:
            sequences_padded = sequences_padded[:, :max_len]
    esm_embs = torch.stack(esm_embs)
    prot_embs = torch.stack(prot_embs)
    raygun_embs = torch.stack(raygun_embs)
    solu_feats = torch.stack(solu_feats)
    labels = torch.stack(labels)
    idx = torch.as_tensor(idx, dtype=torch.long)
    return sequences_padded, esm_embs, prot_embs, raygun_embs, solu_feats, labels, idx


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequence_file,
        esm_file,
        prot_file,
        raygun_file,
        feats_file=None,
        label_file=None,
        augment=False,
    ):
        self.df_sequences = pd.read_csv(sequence_file, header=None)
        self.esm_embeddings = torch.tensor(
            pd.read_csv(esm_file).values, dtype=torch.float
        )
        self.prot_embeddings = torch.tensor(
            pd.read_csv(prot_file).values, dtype=torch.float
        )
        self.raygun_embeddings = torch.tensor(
            pd.read_csv(raygun_file).values, dtype=torch.float
        )
        if feats_file is not None:
            self.solu_feats = torch.tensor(
                pd.read_csv(feats_file).values, dtype=torch.float
            )
            assert len(self.solu_feats) == len(
                self.df_sequences
            ), "Mismatch between Solu features and other inputs."
        else:
            self.solu_feats = None
        if label_file is not None:
            self.labels = torch.tensor(
                pd.read_csv(label_file, header=None).values, dtype=torch.float
            )
            assert (
                len(self.df_sequences)
                == len(self.esm_embeddings)
                == len(self.esm_embeddings)
                == len(self.labels)
            ), "Mismatch in number of samples between files."
        else:
            self.labels = None
            assert (
                len(self.df_sequences)
                == len(self.esm_embeddings)
                == len(self.esm_embeddings)
            ), "Mismatch in number of samples between files."

        self.augment = augment

    def __len__(self):
        return len(self.df_sequences)

    def __getitem__(self, idx):
        sequence_str = self.df_sequences.iloc[idx, 0]
        seq = sequence_str
        if self.augment:
            # Apply augmentations randomly
            if random.random() < 0.5:
                seq = random_deletion(sequence_str)
            if random.random() < 0.5:
                seq = random_substitution(sequence_str)
            if random.random() < 0.5:
                seq = random_masking(sequence_str)
        sequence_tensor = torch.tensor(
            [aa_vocab.get(aa, 21) for aa in seq], dtype=torch.long
        )

        esm_output = self.esm_embeddings[idx]
        prot_output = self.prot_embeddings[idx]
        raygun_output = self.raygun_embeddings[idx]
        if self.labels is not None and self.solu_feats is not None:
            # Training with labels + extra features
            label = self.labels[idx]
            solu_output = self.solu_feats[idx]
            return (
                sequence_tensor,
                esm_output,
                prot_output,
                raygun_output,
                solu_output,
                label,
                idx,
            )

        elif self.labels is not None:
            # Training with labels, but no solubility features
            label = self.labels[idx]
            return (
                sequence_tensor,
                esm_output,
                prot_output,
                raygun_output,
                torch.tensor(-1),
                label,
                idx,
            )

        else:
            # Inference: no labels
            return (
                sequence_tensor,
                esm_output,
                prot_output,
                raygun_output,
                torch.tensor(-1),
                torch.tensor(-1),
                idx,
            )
