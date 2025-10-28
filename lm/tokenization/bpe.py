import json
import os
from pathlib import Path
import pickle
from typing import BinaryIO
import numpy as np
import regex as re
from collections import Counter
from multiprocessing import cpu_count, Pool
from collections.abc import Iterator, Iterable
import cProfile
import pstats

PRE_TOKENIZATION_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PRE_TOKENIZATION_REGEX = re.compile(PRE_TOKENIZATION_REGEX)

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # Vocabulary Initialization:
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for (i, special_token) in enumerate(special_tokens):
        vocab[i + 256] = special_token.encode("utf-8")

    # Pre-tokenization
    word_counts = _get_pre_tokenized_data(input_path, special_tokens)

    # Keep track of the values that were merged
    merges: list[tuple[bytes, bytes]] = []

    # Iteratively merge until our vocab size is reached
    while len(vocab) < vocab_size:

        # Count highest pairs
        # Merge all instances of highest pairs
        pair_counter = Counter()

        # Go through every word
        for (word, count) in word_counts.items():

            # Go through each pair within the word
            # Increment counter for pair according to instances of word
            for pair in zip(word, word[1:]):
                pair_counter[pair] += count

        # Add the most common pair as our newest merge
        top_counted_pair = max(pair_counter.items(), key=lambda pair_count: (pair_count[1], pair_count[0]))[0]
        merges.append(top_counted_pair)
        
        # Merge all instances of this pair within the pre_tokenized_data
        merge = b''.join(top_counted_pair)
        new_word_counts: Counter[tuple[bytes]] = Counter()
        for (word, count) in word_counts.items():
            new_word = []
            i = 0
            while i < len(word):
                if i+1 < len(word) and (word[i], word[i+1]) == top_counted_pair:
                   new_word.append(merge)
                   i += 2
                else:
                   new_word.append(word[i])
                   i += 1
            new_word_counts[tuple(new_word)] = count
        
        word_counts = new_word_counts

        # Add our new merge to the vocab
        vocab[len(vocab)] = merge
    
    return (vocab, merges)


def _get_pre_tokenized_data(input_path: str |os.PathLike, special_tokens: list[str]) -> Counter[tuple[bytes]]:
    """
    Splits a corpus into pretokens (to be further tokenized by BPE).
    """

    num_processes = cpu_count()

    with open(input_path, "rb") as f:

        boundaries = _find_chunk_boundaries(
            f, num_processes, special_tokens[0].encode("utf-8")
        )

        args = []
        for i in range(len(boundaries) - 1):
            args.append((input_path, special_tokens, boundaries[i], boundaries[i + 1]))

        with Pool(num_processes) as pool:
            results = pool.map(_process_chunk, args)

        aggregated_counter = Counter()

        for counter in results:
            aggregated_counter.update(counter)

        return aggregated_counter
    

def _process_chunk(args: tuple[str, list[str], int, int]) -> Counter[tuple[bytes]]: 
    """
    Pretokenize a single chunk of text and return the counted words.
    """
    input_path, special_tokens, begin_index, end_index = args

    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    escaped_special_tokens = "|".join(escaped_special_tokens)
    compiled_escaped_special_tokens = re.compile(f"({escaped_special_tokens})")

    with open(input_path, "br") as f:
        f.seek(begin_index)
        chunk_text = f.read(end_index - begin_index).decode("utf-8", errors="ignore")

        counted_words: Counter[tuple[bytes]] = Counter()
        split_text = compiled_escaped_special_tokens.split(chunk_text)
        for split in split_text:
            if split not in special_tokens and split.strip():
                for match in COMPILED_PRE_TOKENIZATION_REGEX.finditer(split):
                    counted_words.update([tuple(bytes([b]) for b in match.group().encode("utf-8"))])

        return counted_words


def _find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.reverse_vocab: dict[bytes, int] = {}
        for (key, value) in self.vocab.items():
            self.reverse_vocab[value] = key

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        
        if vocab_filepath.endswith(".json"):
            with open(vocab_filepath, encoding="utf-8") as f:
                vocab_data = json.load(f)
                vocab = {}
                for k, v in vocab_data.items():
                    token_id = int(k)
                    if isinstance(v, str):
                        try:
                            vocab[token_id] = bytes.fromhex(v)
                        except ValueError:
                            vocab[token_id] = v.encode("utf-8")
                    elif isinstance(v, list):
                        vocab[token_id] = bytes(v)
                    else:
                        vocab[token_id] = v
        else:
            with open(vocab_filepath, "rb") as f:
                vocab = pickle.load(f)

        if merges_filepath.endswith(".json"):
            with open(merges_filepath, encoding="utf-8") as f:
                merges_data = json.load(f)
                merges = []
                for merge in merges_data:
                    if isinstance(merge[0], str):
                        merge_tuple = (merge[0].encode("utf-8"), merge[1].encode("utf-8"))
                    elif isinstance(merge[0], list):
                        merge_tuple = (bytes(merge[0]), bytes(merge[1]))
                    else:
                        merge_tuple = merge
                    merges.append(merge_tuple)
        else:
            with open(merges_filepath, "rb") as f:
                merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:

        splits = self._split(text)

        encoded_text: list[int] = []
        for split in splits:
            if len(split) == 0: 
                continue
            if self.special_tokens and split in self.special_tokens:
                encoded_text.append(self.reverse_vocab[split.encode("utf-8")])
            else:
                words = COMPILED_PRE_TOKENIZATION_REGEX.finditer(split)
                for match in words:
                    bpe_encoded = self._encode_text_bytes(tuple(bytes([b]) for b in match.group().encode("utf-8")))
                    encoded_text.extend(bpe_encoded)
        return encoded_text
    
    def _split(self, text) -> list[str]:
        # Split on special tokens
        splits = [text]
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_token_regex = re.compile(f"({"|".join(re.escape(special_token) for special_token in sorted_special_tokens)})")
            splits = special_token_regex.split(text)

        return splits
    
    def _encode_text_bytes(self, text_bytes: tuple[bytes]) -> list[int]:

        # Apply merges
        for merge in self.merges:   
            merged: list[bytes] = []
            i = 0
            while i < len(text_bytes):
                if i < len(text_bytes) - 1 and text_bytes[i] == merge[0] and text_bytes[i+1] == merge[1]:
                    merged.append(text_bytes[i] + text_bytes[i+1])
                    i += 2
                else:
                    merged.append(text_bytes[i])
                    i += 1
            text_bytes = tuple(merged)

        # Map final merges to vocab
        token_list: list[int] = []
        
        for vocab_bytes in text_bytes:
            token_list.append(self.reverse_vocab[vocab_bytes])

        return token_list
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.

        This method processes the input line by line to minimize memory usage,
        making it suitable for large files that don't fit in memory.

        Args:
            iterable: An iterable of strings (e.g., file handle)

        Yields:
            Token IDs one at a time
        """
        buffer = ""

        for line in iterable:
            buffer += line

            while buffer:
                if len(buffer) > 8192:  # Process in 8KB chunks
                    # Find last newline in first 8KB
                    chunk_end = buffer.rfind("\n", 0, 8192)
                    if chunk_end == -1:
                        # No newline found, take a smaller chunk at word boundary
                        chunk_end = buffer.rfind(" ", 0, 4096)
                        if chunk_end == -1:
                            chunk_end = 4096  # Force split if no word bounary

                    chunk = buffer[: chunk_end + 1]
                    buffer = buffer[chunk_end + 1 :]

                    token_ids = self.encode(chunk)
                    yield from token_ids
                else:
                    break

        if buffer:
            token_ids = self.encode(buffer)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:

        decoded_bytes: list[bytes] = []
        for id in ids:
            
            vocab_bytes = self.vocab[id]
            decoded_bytes.append(vocab_bytes)

        return b"".join(decoded_bytes).decode("utf-8", errors='replace')
    
    def encode_dataset_to_numpy(
        self,
        file_path: str | Path,
        output_path: str | Path,
        max_vocab_size: int = 65536,  # uint16 max value
    ) -> None:
        """
        Encode a large dataset to numpy array with memory-efficient streaming.

        Args:
            tokenizer: The tokenizer to use
            file_path: Path to input text file
            output_path: Path to save encoded numpy array
            chunk_size: Size of chunks to process at once (bytes)
            max_vocab_size: Maximum vocabulary size (for uint16 validation)
        """
        print(f"Encoding dataset {file_path} to {output_path}...")

        all_token_ids: list[int] = []

        with open(file_path, encoding="utf-8") as f:
            token_count = 0
            for token_id in self.encode_iterable(f):
                all_token_ids.append(token_id)
                token_count += 1

                if token_count % 1_000_000 == 0:
                    print(f"  Processed {token_count:,} tokens...")

        print(f"Total tokens: {len(all_token_ids):,}")

        max_token_id = max(all_token_ids)
        if max_token_id >= max_vocab_size:
            raise ValueError(f"Token ID {max_token_id} exceeds uint16 range [0, {max_vocab_size - 1}]")

        token_array = np.array(all_token_ids, dtype=np.uint16)

        np.save(output_path, token_array)

        print(f"Saved {len(token_array):,} tokens to {output_path}")
        print(f"Array shape: {token_array.shape}")
        print(f"Array dtype: {token_array.dtype}")
        print(f"File size: {os.path.getsize(str(output_path) + '.npy') / (1024 * 1024):.1f} MB")


if __name__ == "__main__":


    # profiler = cProfile.Profile()
    # profiler.enable()
   
    t = Tokenizer.from_files("../output/openwebtext_vocab.json", "../output/openwebtext_merges.pkl")
    t.encode_dataset_to_numpy("../data/imessages_sft.txt", "../data/encoded/imessages_sft.npy")

    # profiler.disable()

    # stats = pstats.Stats(profiler)
    # stats.sort_stats("tottime")

    # print("\nTop 10 functions by total time:")
    # stats.print_stats(10)


