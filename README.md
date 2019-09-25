# ExpressGNN

This is an implementation of the ExpressGNN proposed in the paper "Efficient Probabilistic Logic Reasoning with Graph Neural Networks".

## Requirements
- python 3.7
- pytorch 1.1
- scikit-learn
- networkx
- tqdm

## Quick Start
The following command starts the inference on the Kinship-S1 dataset on GPU:
```
python -m main.train -data_root data/kinship/S1 -slice_dim 8 -batchsize 16 -use_gcn 1 -embedding_size 64 -gcn_free_size 32 -load_method 0 -exp_folder exp -exp_name kinship -device cuda
```

To run ExpressGNN on the FB15K-237 dataset on GPU, use the follwoing command line:
```
python -m main.train -data_root data/fb15k-237 -rule_filename cleaned_rules_weight_larger_than_0.9.txt -slice_dim 16 -batchsize 16 -use_gcn 1 -num_hops 1 -embedding_size 128 -gcn_free_size 127 -patience 20 -lr_decay_patience 100 -entropy_temp 1 -load_method 1 -exp_folder exp -exp_name freebase -device cuda
```
