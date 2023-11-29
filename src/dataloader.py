from dataclasses import dataclass
from random import Random
from typing import Callable, Dict, Optional, Union, List
from pathlib import Path
import csv
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
)

import torch
from torch.utils.data import DataLoader

from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)
from transformers import AutoTokenizer

from util import flatten, unflatten, relative_to_project_root
import random
import numpy as np


class DataLoaderGroup:
    train: DataLoader
    test: DataLoader
    validation: DataLoader

    def __init__(
        self, datasets: DatasetDict, collator: Callable, batch_size: int
    ) -> None:
        def seed_init_fn(x):
            seed = 42 + x
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        self.train = DataLoader(
            datasets["train"],  # type: ignore
            collate_fn=collator,
            shuffle=True,
            batch_size=batch_size,
            worker_init_fn=seed_init_fn,
        )
        self.validation = DataLoader(
            datasets["validation"],  # type: ignore
            collate_fn=collator,
            batch_size=batch_size,
            worker_init_fn=seed_init_fn,
        )
        self.test = DataLoader(
            datasets["test"],  # type: ignore
            collate_fn=collator,
            batch_size=batch_size,
            worker_init_fn=seed_init_fn,
        )


@dataclass
class DataLoaderConfig:
    batch_size: int
    tokenizer_source: str
    train_weight: int = 90
    test_weight: int = 8
    validation_weight: int = 2
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path: Optional[Union[str, Path]] = None

    def split_dataset(
        self,
        train: Dataset,
        test: Optional[Dataset] = None,
        validation: Optional[Dataset] = None,
    ) -> DatasetDict:
        if test is None:
            train_vs_test_val = 1 - self.train_weight / (
                self.train_weight + self.test_weight + self.validation_weight
            )
            train_test = train.train_test_split(test_size=train_vs_test_val)
            train = train_test["train"]
            test = train_test["test"]
        if validation is None:
            test_vs_val = self.test_weight / (self.test_weight + self.validation_weight)
            test_validation = test.train_test_split(test_size=test_vs_val)
            test = test_validation["test"]
            validation = test_validation["train"]
        dataset_dict = {
            "train": train,
            "validation": validation,
            "test": test,
        }

        for k, v in dataset_dict.items():
            print(f"Size of {k} : {len(v)}")

        splitted_dataset = DatasetDict(dataset_dict)
        return splitted_dataset

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer_source)


def paradetox_tsv_reader(file_path: Union[Path, str]) -> Dict[str, List]:
    with open(file_path, "r", encoding="utf-8") as file:
        # Initialize the CSV reader with tab delimiter
        reader = csv.reader(file, delimiter="\t")
        # Skip the header row
        next(reader, None)
        result = {"input": [], "outputs": []}
        for row in reader:
            result["input"].append(row[0])
            result["outputs"].append([output for output in row[1:] if output])

    return result


def load_paradetox_dataset(path: Optional[Union[Path, str]]) -> Dataset:
    path = path or relative_to_project_root("data/paradetox/paradetox.tsv")
    return Dataset.from_dict(paradetox_tsv_reader(path))


@dataclass
class DataCollatorForParaDetox:
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    seed: int = 42
    generator: Random = Random(seed)
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        sampled_features = []
        sampled_labels = []
        for f in features:
            sample_idx = self.generator.randint(0, len(f["labels"]) - 1)
            # process labels seperately (see https://github.com/huggingface/transformers/issues/20182)
            sampled_features.append(
                {k: v[sample_idx] for k, v in f.items() if k != "labels"}
            )
            sampled_labels.append(f["labels"][sample_idx])

        # Apply Padding
        padded_inputs = self.tokenizer.pad(
            sampled_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        padded_labels = self.tokenizer.pad(
            {"input_ids": sampled_labels},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=False,  # No attention mask for labels
            return_tensors="pt",
        )
        padded_inputs["labels"] = padded_labels["input_ids"]

        for k, v in padded_inputs.items():
            if isinstance(v, torch.Tensor):
                padded_inputs[k] = v.to(device=self.device)

        return padded_inputs


def paradetox_dataloader(cfg: DataLoaderConfig) -> DataLoaderGroup:
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_source)
    dataset = load_paradetox_dataset(cfg.path)

    def tokenize_function(examples: Dict[str, List]):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        inputs: List[str] = examples["input"]
        outputs: List[List[str]] = examples["outputs"]
        inputs_duplicated = [i for i, o in zip(inputs, outputs) for _ in o]

        # Flatten
        outputs_flatten, partition = flatten(outputs)
        assert len(inputs_duplicated) == len(
            outputs_flatten
        ), f"{len(inputs_duplicated)} vs. {len(outputs_flatten)}"

        # Tokenize
        tokenized = tokenizer(
            inputs_duplicated, text_target=outputs_flatten, truncation=True
        )

        # unflatten
        result = {k: unflatten(v, partition) for k, v in tokenized.items()}

        for i in result["input_ids"]:
            assert len(i) != 0
        return result

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )
    # tokenized_dataset.set_format("torch")
    splitted_dataset = cfg.split_dataset(tokenized_dataset)

    collator = DataCollatorForParaDetox(tokenizer, cfg.device)
    return DataLoaderGroup(splitted_dataset, collator, cfg.batch_size)


@dataclass
class DataCollatorForJigsaw:
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        tags = [feature.pop("tags") for feature in features]

        # Apply Padding
        padded_inputs = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        padded_inputs["tags"] = torch.tensor(tags).to(device=self.device)

        for k, v in padded_inputs.items():
            if isinstance(v, torch.Tensor):
                padded_inputs[k] = v.to(device=self.device)

        return padded_inputs


def jigsaw_dataloader(cfg: DataLoaderConfig) -> DataLoaderGroup:
    tokenizer = cfg.get_tokenizer()
    path = cfg.path or relative_to_project_root("data/jigsaw")
    dataset = load_dataset("jigsaw_toxicity_pred", data_dir=str(path))
    assert isinstance(dataset, DatasetDict)

    def tokenize_function(examples: Dict[str, List]):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        inputs: List[str] = examples["comment_text"]
        tag_names = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
        length = len(inputs)
        tags = [[examples[name][i] for name in tag_names] for i in range(length)]

        # Tokenize
        tokenized = tokenizer(inputs, truncation=True)

        result = dict(tokenized.items())
        result["tags"] = tags
        return result

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # tokenized_dataset.set_format("torch")

    accepted_keys = ["input_ids", "attention_mask", "tags"]
    for key in tokenized_dataset["train"].features.keys():
        if key not in accepted_keys:
            tokenized_dataset = tokenized_dataset.remove_columns(key)

    splitted_dataset = cfg.split_dataset(
        tokenized_dataset["train"], tokenized_dataset["test"]
    )

    collator = DataCollatorForJigsaw(tokenizer, cfg.device)
    return DataLoaderGroup(splitted_dataset, collator, cfg.batch_size)


if __name__ == "__main__":
    cfg = DataLoaderConfig(batch_size=8, tokenizer_source="facebook/bart-large")
    dlg = jigsaw_dataloader(cfg)
    print(f"Train split size:\t\t{len(dlg.train)}")
    print(f"Test split size:\t\t{len(dlg.test)}")
    print(f"Validation split size:\t{len(dlg.validation)}")

    tokenizer = cfg.get_tokenizer()
    for batch in dlg.train:
        decoded_input = tokenizer.decode(
            batch["input_ids"][0], skip_special_tokens=True
        )
        print(decoded_input)
        if "labels" in batch.keys():
            decoded_label = tokenizer.decode(
                batch["labels"][0], skip_special_tokens=True
            )
            print(decoded_label)
        break
