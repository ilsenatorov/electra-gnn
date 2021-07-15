# ELECTRA-GNN


## Required packages

* torch
* torch_geometric
* rdkit
* pytorch_lightning
## Pretraining

Run the script with `python pretrain.py [DATA_FILE]`

The `[DATA_FILE]` has to be a csv file with column labeled `smiles`, which will contain the SMILES representation of the molecules

Model data will be logged to `tb_logs` folder (including checkpoints).

## Finetuning

Run the script with `python finetune.py [DATA_FILE] --pretrained_model_file [PRETRAIN_CHECKPOINT]`

The `[DATA_FILE]` should contain the `smiles` column, as well as a `y` column, which will label be used as the target value

In case of classification problem (`y` denotes the class) use the `--task class` argument, in case of regression `--task reg`