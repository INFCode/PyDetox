from dataclasses import dataclass
from random import Random
from typing import Callable, Dict, Optional, Union, List
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
)

import csv
import torch
from torch.utils.data import DataLoader

from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)
from transformers import BartTokenizer
from pathlib import Path

from util import flatten, unflatten, relative_to_project_root


class DataLoaderGroup:
    train: DataLoader
    test: DataLoader
    validation: DataLoader

    def __init__(
        self, datasets: DatasetDict, collator: Callable, batch_size: int
    ) -> None:
        self.train = DataLoader(
            datasets["train"],  # type: ignore
            collate_fn=collator,
            # shuffle=True,
            batch_size=batch_size,
        )
        self.validation = DataLoader(
            datasets["validation"],  # type: ignore
            collate_fn=collator,
            batch_size=batch_size,
        )
        self.test = DataLoader(
            datasets["test"],  # type: ignore
            collate_fn=collator,
            batch_size=batch_size,
        )


def paradetox_tsv_reader(file_path: Path | str) -> Dict[str, List]:
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


def load_paradetox_dataset() -> Dataset:
    path = relative_to_project_root("data/paradetox/paradetox.tsv")
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

        return padded_inputs


def paradetox_dataloader(
    tokenizer: PreTrainedTokenizerBase, device: torch.device
) -> DataLoaderGroup:
    dataset = load_paradetox_dataset()

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
    train_test = tokenized_dataset.train_test_split(test_size=0.12)
    test_validation = train_test["test"].train_test_split(test_size=0.8)
    splitted_dataset = DatasetDict(
        {
            "train": train_test["train"],
            "validation": test_validation["train"],
            "test": test_validation["test"],
        }
    )

    collator = DataCollatorForParaDetox(tokenizer, device)
    return DataLoaderGroup(splitted_dataset, collator, 2)


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

        return padded_inputs


def jigsaw_dataloader(
    tokenizer: PreTrainedTokenizerBase, device: torch.device
) -> DataLoaderGroup:
    path = relative_to_project_root("data/jigsaw")
    dataset = load_dataset("jigsaw_toxicity_pred", data_dir=str(path))
    # dataset_train = load_dataset(
    #    "jigsaw_toxicity_pred", data_dir=str(path), split="train[:100]"
    # )
    # dataset_test = load_dataset(
    #    "jigsaw_toxicity_pred", data_dir=str(path), split="test[:20]"
    # )
    # dataset = DatasetDict({"train": dataset_train, "test": dataset_test})
    assert isinstance(dataset, DatasetDict)
    print(dataset.keys())

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

    test_validation = tokenized_dataset["test"].train_test_split(test_size=0.8)
    splitted_dataset = DatasetDict(
        {
            "train": tokenized_dataset["train"],
            "validation": test_validation["train"],
            "test": test_validation["test"],
        }
    )

    collator = DataCollatorForJigsaw(tokenizer, device)
    return DataLoaderGroup(splitted_dataset, collator, 2)


if __name__ == "__main__":
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dlg = jigsaw_dataloader(tokenizer, device)
    print(f"Train split size:\t\t{len(dlg.train)}")
    print(f"Test split size:\t\t{len(dlg.test)}")
    print(f"Validation split size:\t{len(dlg.validation)}")

    for batch in dlg.test:
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
