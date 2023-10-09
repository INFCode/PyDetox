from collections import Counter, defaultdict
from typing import DefaultDict, List, Tuple
import logging as lg
import os
from pathlib import Path
import requests

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from nltk import TreebankWordTokenizer
from tqdm import tqdm


class ParaDetoxDataset(Dataset):
    _X: List[List[int]]
    _Y: List[List[int]]

    def __init__(self, X: List[List[int]], Y: List[List[int]]):
        assert len(X) == len(
            Y
        ), f"Data and labels should have same length, but {len(X)} vs {len(Y)} received."

        super().__init__()

        self._X = X
        self._Y = Y

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, index):
        return self._X[index], self._Y[index]


class Vocabulary:
    pad_token: str = "<TOK_PAD>"
    start_token: str = "<TOK_SOS>"
    end_token: str = "<TOK_EOS>"
    unknown_token: str = "<TOK_UNK>"
    special_tokens: List[str] = [pad_token, start_token, end_token, unknown_token]

    _vocab: List[str]
    _index: DefaultDict[str, int]

    def __init__(self, tokens: List[str], unknown_threshold: int = 3) -> None:
        word_freq = dict(Counter(tokens))
        self._vocab = self.special_tokens + [
            k for k, v in word_freq.items() if v >= unknown_threshold
        ]
        self._index = defaultdict(
            lambda: self._vocab.index(self.unknown_token),  # default to unknown
            zip(self._vocab, range(len(self._vocab))),
        )

    def __len__(self) -> int:
        return len(self._vocab)

    def tok2idx(self, token: str) -> int:
        return self._index[token]

    def idx2tok(self, index: int) -> str:
        return self._vocab[index]

    def encode_sents(self, sentences: List[List[str]]):
        return [[self.tok2idx(word) for word in sentence] for sentence in sentences]

    def decode_sents(self, sentences: List[List[int]]):
        return [[self.idx2tok(idx) for idx in sentence] for sentence in sentences]


def preprocess_dataset(path_to_dataset: str) -> Tuple[DataLoader, Vocabulary]:
    path = _prepare_path(path_to_dataset)

    lg.info("Loading dataset")
    toxic, neutral = _load_dataset(path)

    lg.info("Tokenizing dataset")
    tokenizer = TreebankWordTokenizer()
    toxic_tokens = tokenizer.tokenize_sents(toxic)
    neutral_tokens = tokenizer.tokenize_sents(neutral)

    flatten_tokens = [t for sentence in toxic_tokens + neutral_tokens for t in sentence]
    lg.info("Building vocabulary")
    vocab = Vocabulary(flatten_tokens, unknown_threshold=2)

    lg.info("Encoding dataset")
    X = vocab.encode_sents(toxic_tokens)
    Y = vocab.encode_sents(neutral_tokens)

    lg.info("Building dataloader")
    dataloader = _make_dataloader(X, Y, vocab)

    return dataloader, vocab


def _load_dataset(path: Path) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(path, delimiter="\t", header=0)
    lg.debug(f"{df.head()=}")

    toxic_col = "toxic"
    neutral_col = [f"neutral{i}" for i in range(1, 4)]

    toxic = []
    neutral = []

    for _, row in tqdm(df.iterrows()):
        toxic_sentence = row[toxic_col]
        neutral_sentences = row[neutral_col]
        neutral_sentences = neutral_sentences[neutral_sentences.notna()].tolist()
        for n in neutral_sentences:
            toxic.append(toxic_sentence)
            neutral.append(n)

    return toxic, neutral


def _make_dataloader(
    X: List[List[int]], Y: List[List[int]], vocab: Vocabulary
) -> DataLoader:
    dataset = ParaDetoxDataset(X, Y)
    pad_token_index = vocab.tok2idx(vocab.pad_token)

    # define a collate function that pads the batch to the longest one
    def collate_fn(samples: List[Tuple[List[int], List[int]]]):
        lengths = [(len(s[0]), len(s[1])) for s in samples]
        lengths_X, lengths_Y = list(zip(*lengths))  # unzips into 2 lists
        longest_X, longest_Y = max(lengths_X), max(lengths_Y)
        batch_size = len(samples)

        # dtype have to be int as expected by the embedding layer
        padded_X = (
            torch.ones((batch_size, longest_X), dtype=torch.int32) * pad_token_index
        )
        padded_Y = (
            torch.ones((batch_size, longest_Y), dtype=torch.int32) * pad_token_index
        )

        # copy over the actual sequences
        for i in range(batch_size):
            x, y = samples[i]
            x_l, y_l = lengths[i]
            padded_X[i, :x_l] = torch.as_tensor(x)
            padded_Y[i, :y_l] = torch.as_tensor(y)

        return padded_X, padded_Y, lengths_X, lengths_Y

    dataloader = DataLoader(dataset, collate_fn=collate_fn)
    return dataloader


def _prepare_path(path_to_dataset: str) -> Path:
    path = Path(path_to_dataset)
    if path_to_dataset[-1] == "/":
        path = path / "paradetox.tsv"

    lg.info("Looking at data path {str(path)}.")

    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        # file not downloaded yet
        lg.info("Dataset not found in given path. Download from GitHub.")
        _download_dataset(path)

    return path


def _download_dataset(path: Path):
    # Define the URL of the file you want to download
    url = (
        "https://raw.githubusercontent.com/s-nlp/paradetox/main/paradetox/paradetox.tsv"
    )

    # Create parent folders if they don't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Download the file
    response = requests.get(url, timeout=1000)

    if response.status_code == 200:
        with open(path, "wb") as file:
            file.write(response.content)
        lg.info(f"File downloaded successfully to {path}")
    else:
        lg.warning(f"Failed to download the file. Status code: {response.status_code}")


if __name__ == "__main__":
    lg.basicConfig(level=lg.DEBUG)
    dl, vocab = preprocess_dataset("./data/")

    for i in range(20):
        print(f"{vocab.idx2tok(i)}", end=" ")
    print()

    for i, o, il, ol in dl:
        print(f"{i}")
        print(f"{vocab.decode_sents(i)}")
        break
