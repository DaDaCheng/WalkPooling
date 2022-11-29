# WalkPooling


## About

This is the source code for paper _Neural Link Prediction with Walk Pooling_.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neural-link-prediction-with-walk-pooling-1/link-prediction-on-cora)](https://paperswithcode.com/sota/link-prediction-on-cora?p=neural-link-prediction-with-walk-pooling-1)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neural-link-prediction-with-walk-pooling-1/link-prediction-on-pubmed)](https://paperswithcode.com/sota/link-prediction-on-pubmed?p=neural-link-prediction-with-walk-pooling-1)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neural-link-prediction-with-walk-pooling-1/link-prediction-on-citeseer)](https://paperswithcode.com/sota/link-prediction-on-citeseer?p=neural-link-prediction-with-walk-pooling-1)

## Requirements

python>=3.3.7

torch>=1.9.0

torch-cluster>=1.5.9

torch-geometric>=2.0.0

torch-scatter>=2.0.8

torch-sparse>=0.6.11

tqdm

This code was tested on macOS and Linux.

## Run

### Quick start

	python ./src/main.py --data-name USAir

### Parameters

#### Data and sample

`--data-name`: supported data:

1. Without node attributes: USAir NS Power Celegans Router PB Ecoli Yeast

2. With node attributes: cora citeseer pubmed

`--use-splitted`: when it is `True`, we use the splitted data from [SEAL](https://github.com/muhanzhang/SEAL). When it is `False`, we will randomly split train, validation and test data.

`--data-split-num`: the index of splitted data when `--use-splitted` is `True`. From 1 to 10.

`--test-ratio` and `--val-ratio`: Test ratio and validation ratio of the data set when `--use-splitted` is False. Defaults are `0.1` and `0.05` respectively.

`--observe-val-and-injection`: whether to contain the validation set in the observed graph and apply injection trick.

`--practical-neg-sample`: whether only see the train positive edges when sampling negative.

`--num-hops`: number of hops in sampling subgraph. Default is `2`.

`--max-nodes-per-hop`: When the graph is too large or too dense, we need max node per hop threshold to avoid OOM. Default is `None`.


#### Hyperparameters

`--init-attribute`: the initial attribute for graphs without node attributes. options: `n2v`, `one_hot`, `spc`, `ones`, `zeros`, `None`. Default is `ones`.

`--init-representation`: node feature representation . options:  `gic`, `vgae`, `argva`, `None`. Default is `None`.

`--drnl`: whether to use drnl labeling. Default is `False`.

`--seed`: random seed. Default is `1`.

`--lr`: learning rate. Default is `0.00005`.

`-heads`: using multi-heads in the attention link weight encoder. Default is `2`.

`--hidden-channels`: Default is `32`.

`--batch-size`: Default is `32`.

`--epoch-num`: Default is `50`.



## Reproducibility


Reproduce Table 1, 2, 3, 4 in the paper.

	./bash/run.sh

## Reference


If you find our work useful in your research, please cite our paper:

```
@article{pan2021neural,
  title={Neural Link Prediction with Walk Pooling},
  author={Pan, Liming and Shi, Cheng and Dokmani{\'c}, Ivan},
  journal={arXiv preprint arXiv:2110.04375},
  year={2021}
}
```

