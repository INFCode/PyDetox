import dataclasses
from dataclasses import dataclass, field, fields
import argparse
from argparse import Namespace
from itertools import product
from typing import Any, Dict, Generator, List, Type, get_args, get_origin


@dataclass
class ConfigBase:
    @classmethod
    def parse(cls):
        parser = argparse.ArgumentParser()
        tunable = {}
        for field in fields(cls):
            args = [f"--{field.name.replace('_', '-')}"]
            kwargs: Dict[str, Any] = {
                "help": field.metadata.get("help", ""),
            }

            if field.default == dataclasses.MISSING:
                kwargs["required"] = True
            else:
                kwargs["default"] = field.default

            is_list = get_origin(field.type) == List
            allow_tuning = field.metadata.get("tunable", False)

            if allow_tuning and is_list:
                # List of list is not supported
                # as it's hard to parse
                raise ValueError("Lists cannot be tunable.")

            if allow_tuning or is_list:
                kwargs["nargs"] = "+"

            if is_list:
                kwargs["type"] = get_args(field.type)[0]
            else:
                kwargs["type"] = field.type

            tunable[field.name] = allow_tuning

            parser.add_argument(*args, **kwargs)
        args = parser.parse_args()

        result = ParseResult(args, tunable, cls)

        return result


@dataclass
class ParseResult:
    args: Namespace
    tunable: Dict[str, bool]
    target_type: Type[ConfigBase]

    def enumerate(self) -> Generator[ConfigBase, None, None]:
        ns_dict = vars(self.args)

        keys = ns_dict.keys()
        value_lists = [v if self.tunable[k] else [v] for k, v in ns_dict.items()]

        for combination in product(*value_lists):
            # Yield a new dictionary for each combination
            yield self.target_type(**dict(zip(keys, combination)))


# Define a dataclass that will store your configuration parameters
@dataclass
class ParallelConfig(ConfigBase):
    missing: int
    epochs: int = field(default=10, metadata={"help": "Number of epochs"})
    batch_size: int = field(default=32, metadata={"help": "Batch size"})
    learning_rate: float = field(
        default=0.001, metadata={"help": "Learning rate", "tunable": True}
    )
    dataset_path: str = field(
        default="data/paradetox/paradetox.py", metadata={"help": "Path to the dataset"}
    )
    save_model: bool = field(
        default=True, metadata={"help": "Flag to save the model after training"}
    )


if __name__ == "__main__":
    result = ParallelConfig.parse()
    for cfg in result.enumerate():
        print(cfg)
