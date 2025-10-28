import torch
from numpy.typing import NDArray
import numpy as np
import numpy.random as random
from jaxtyping import Int
def load_batch(
        tokens: NDArray,
        batch_size: int,
        context_length: int,
        device="cpu"
    ) -> tuple[Int[torch.Tensor, "batch_size context_length"]]:
    """
    Load data from sequential token integers into to tensors ready for model input.
    Args:
        tokens: a numpy array of integer tokens
        batch_size: the number of example to split the data into
        context_length: the length of each example
        device_string: device on which to place the resulting tensors
    Returns:
        A tuple of tensors, each of size batch_size x context_length
        The first containing the token sequence examples
        The second containing the correct next token prediction
    """

    # Unpopulated tensors to be returned
    sequences = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    labels = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)

    # Generate random sample indices
    max_index = len(tokens) - context_length - 1
    random_indices = random.randint(0, max_index + 1, size=batch_size)

    # Sample and populate tensors
    for (sample_number, sample_index) in enumerate(random_indices):
        sequence = torch.from_numpy(tokens[sample_index: sample_index + context_length].copy()).to(device, dtype=torch.long)
        label = torch.from_numpy(tokens[sample_index + 1: sample_index + context_length + 1].copy()).to(device, dtype=torch.long)

        sequences[sample_number] = sequence
        labels[sample_number] = label

    return (sequences, labels)


import numpy as np
import torch
from typing import Tuple, List

class ConversationBatchLoader:
    def __init__(self, file_path: str, batch_size: int, context_length: int, device: torch.device):
        self.tokens = np.memmap(file_path, dtype=np.uint16, mode='r')
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

        # Define special tokens
        self.END_TOKEN = 0
        self.ME_TOKEN = 1
        self.THEM_TOKEN = 2

        # Precompute message boundaries for speed
        self.boundaries = self._compute_boundaries()

    def _compute_boundaries(self) -> List[Tuple[int, int]]:
        """
        Precompute (start, end) indices of all <|Me|> and <|Them|> segments,
        skipping <|endoftext|> boundaries.
        """
        boundaries = []
        i = 0
        while i < len(self.tokens):
            if self.tokens[i] == self.END_TOKEN:
                i += 1
                continue
            if self.tokens[i] in (self.ME_TOKEN, self.THEM_TOKEN):
                start = i
                i += 1
                while i < len(self.tokens) and self.tokens[i] not in (self.ME_TOKEN, self.THEM_TOKEN, self.END_TOKEN):
                    i += 1
                end = i
                boundaries.append((start, end))
            else:
                i += 1
        return boundaries

    def load_batch(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Load a batch of message-aligned samples.
        Each sequence contains as many *full messages* as will fit
        within context_length, without splitting any.

        Returns:
            padded_seqs: (batch, max_len) tensor of inputs
            padded_labels: (batch, max_len) tensor of labels
            valid_lengths: list[int] of number of unmasked (real) tokens
        """
        batch_sequences = []
        batch_labels = []
        valid_lengths = []

        chosen_segments = np.random.choice(len(self.boundaries), self.batch_size, replace=True)

        for seg_idx in chosen_segments:
            start, _ = self.boundaries[seg_idx]
            pos = start
            collected = []

            while pos < len(self.tokens):
                if self.tokens[pos] == self.END_TOKEN:
                    break

                if self.tokens[pos] in (self.ME_TOKEN, self.THEM_TOKEN):
                    msg_start = pos
                    pos += 1
                    while pos < len(self.tokens) and self.tokens[pos] not in (self.ME_TOKEN, self.THEM_TOKEN, self.END_TOKEN):
                        pos += 1
                    msg_end = pos
                    msg = self.tokens[msg_start:msg_end]

                    # Stop if adding this message would overflow context_length
                    if len(collected) + len(msg) > self.context_length:
                        break

                    collected.extend(msg)
                else:
                    pos += 1

            if not collected:
                collected = [self.END_TOKEN]

            seq = torch.tensor(collected, dtype=torch.long, device=self.device)
            label = torch.roll(seq.clone(), shifts=-1, dims=0)
            label[-1] = self.END_TOKEN

            batch_sequences.append(seq)
            batch_labels.append(label)
            valid_lengths.append(len(seq))  # number of unmasked tokens

        max_len = max(valid_lengths)
        padded_seqs = torch.full((self.batch_size, max_len), self.END_TOKEN, dtype=torch.long, device=self.device)
        padded_labels = torch.full((self.batch_size, max_len), self.END_TOKEN, dtype=torch.long, device=self.device)

        for i, (seq, label) in enumerate(zip(batch_sequences, batch_labels)):
            l = valid_lengths[i]
            padded_seqs[i, :l] = seq
            padded_labels[i, :l] = label

        return padded_seqs, padded_labels, valid_lengths
