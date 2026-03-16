import json

import torch


class Config:
    """Load and expose a named section from ``configs.json``."""

    def __init__(self, section: str, file_path: str = "configs.json") -> None:
        with open(file_path, "r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        self._config = raw_config.get(section, {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_property(self, property_name):
        return self._config.get(property_name)

    def set_property(self, property_name, value) -> None:
        self._config[property_name] = value

    def update_from_args(self, args) -> None:
        for name in ["learning_rate", "batch_size", "epochs", "weight_decay", "patience", "pred_lambda"]:
            if hasattr(args, name):
                value = getattr(args, name)
                if value is not None:
                    self.set_property(name, value)


class Create(Config):
    def __init__(self) -> None:
        super().__init__("create")

    @property
    def slice_size(self):
        return self.get_property("slice_size")

    @property
    def joern_cli_dir(self):
        return self.get_property("joern_cli_dir")


class Data(Config):
    def __init__(self, section: str) -> None:
        super().__init__(section)

    @property
    def cpg(self):
        return self.get_property("cpg")

    @property
    def raw(self):
        return self.get_property("raw")

    @property
    def input(self):
        return self.get_property("input")

    @property
    def model(self):
        return self.get_property("model")

    @property
    def tokens(self):
        return self.get_property("tokens")


class Paths(Data):
    def __init__(self) -> None:
        super().__init__("paths")

    @property
    def joern(self):
        return self.get_property("joern")


class Files(Data):
    def __init__(self) -> None:
        super().__init__("files")


class Embed(Config):
    def __init__(self) -> None:
        super().__init__("embed")

    @property
    def nodes_dim(self):
        return self.get_property("nodes_dim")

    @property
    def edge_type(self):
        return self.get_property("edge_type")


class Process(Config):
    def __init__(self) -> None:
        super().__init__("process")

    @property
    def epochs(self):
        return self.get_property("epochs")

    @property
    def patience(self):
        return self.get_property("patience")

    @property
    def batch_size(self):
        return self.get_property("batch_size")

    @property
    def dataset_ratio(self):
        return self.get_property("dataset_ratio")

    @property
    def shuffle(self):
        return self.get_property("shuffle")


class BertGGNN(Config):
    def __init__(self) -> None:
        super().__init__("bertggnn")

    @property
    def learning_rate(self):
        return self.get_property("learning_rate")

    @property
    def weight_decay(self):
        return self.get_property("weight_decay")

    @property
    def pred_lambda(self):
        return self.get_property("pred_lambda")

    @property
    def model(self):
        return self.get_property("model")
