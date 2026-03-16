# DGV

DGV is the open-source implementation of our dual-view graph vulnerability detection framework. This repository provides the core model, the public training and evaluation pipeline, and the Joern export script used in preprocessing.

## Repository layout

```text
.
|-- training/
|-- joern/
|-- loss_functions/
|-- models/
|-- utils/
|-- configs.py
|-- configs.json
|-- requirements.txt
`-- run_dgv.py
```

## Datasets

The experiments in our work are built on the following public vulnerability detection datasets:

- Devign: https://github.com/epicosy/devign
- ReVeal: https://github.com/VulDetProject/ReVeal
- DiverseVul: https://github.com/wagner-group/diversevul

Please follow the corresponding dataset licenses, access policies, and preprocessing requirements when preparing data for reproduction.

## Environment

```bash
pip install -r requirements.txt
```

## Joern setup

This repository ships the Joern export script at `joern/graph-for-funcs.sc`.

Install Joern separately from the official project:

- https://joern.io/
- https://github.com/joernio/joern

After installing Joern, update `create.joern_cli_dir` in `configs.json` so it points to your local Joern CLI directory.

## Data expectation

`run_dgv.py` expects preprocessed PyTorch Geometric samples stored as `.pkl` files under `data/input/` by default. Each record should provide:

- `input`: a `torch_geometric.data.Data` object
- `target`: binary label
- `func`: source code string

## Train

```bash
python run_dgv.py --train --input_path data/input --epochs 100 --batch_size 8
```

## Test

```bash
python run_dgv.py --test --input_path data/input --checkpoint data/model/DualGraphVulD_best.pt
```

