from dataclasses import dataclass, field, fields
import argparse


# Define a dataclass that will store your configuration parameters
@dataclass
class ParallelConfig:
    epochs: int = field(default=10, metadata={"help": "Number of epochs"})
    batch_size: int = field(default=32, metadata={"help": "Batch size"})
    learning_rate: float = field(default=0.001, metadata={"help": "Learning rate"})
    dataset_path: str = field(
        default="data/paradetox/paradetox.py", metadata={"help": "Path to the dataset"}
    )
    save_model: bool = field(
        default=True, metadata={"help": "Flag to save the model after training"}
    )

    # A method to create an argparse parser based on the dataclass fields
    @classmethod
    def create_parser(cls):
        parser = argparse.ArgumentParser()
        for field in fields(cls):
            field_name = field.name
            field_type = field.type
            field_default = (
                field.default if field.default != dataclass._MISSING_TYPE else None
            )
            field_help = field.metadata.get("help", "")
            field_required = field.default == dataclass._MISSING_TYPE

            # Add argument to the parser based on the presence of default value
            if field_required:
                parser.add_argument(
                    f"--{field_name}", type=field_type, required=True, help=field_help
                )
            else:
                parser.add_argument(
                    f"--{field_name}",
                    type=field_type,
                    default=field_default,
                    help=field_help,
                )
        return parser

    # A method to parse command line arguments into an instance of the Config dataclass
    @classmethod
    def from_command_line(cls):
        parser = cls.create_parser()
        args = parser.parse_args()
        return cls(**vars(args))


# To use the Config dataclass for parsing command line arguments, you can use the following:
# config = Config.from_command_line()
# print(config)
