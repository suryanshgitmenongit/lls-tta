Temporary development branch for stta paper:

# Test Time Domain Generalization

Repo for performing test time adaptation on domain generalization benchmarks. Built on top of [T3A](https://github.com/matsuolab/T3A) (itself built on top of [DomainBed](https://github.com/facebookresearch/DomainBed)).

## Setting up environment:
In your envs folder:
```
module load python/3.8  # if in compute canada, else ignore this step
virtualenv --no-download ttdg
source ttdg/bin/activate
pip install -r <path to requirements.txt>
```

## Data setup
Use the original T3A or Domainbed README to download the dataset into data folders.

## Running code
### Single runs
Feel free to use original T3A code for single runs. Example usage below:

TTDG is performed in two steps:

#### 1. Source Pretraining
```
python -m domainbed.scripts.train\
       --data_dir /home-local2/sasah14.extra.nobkp/scratch/data/\
       --output_dir /home-local2/sasah14.extra.nobkp/scratch/out/ttdg/single_run/debug\
       --algorithm ERM\
       --dataset PACS\
       --hparams "{\"backbone\": \"resnet18-BN\", \"batchsize\": 32}"\
       --test_env 0\
       --wandb_name te0_PACS_resnet18-BN_ERM_0
```
Wandb logs offline. Remove `wandb_name` to disable wandb logging.

#### 2. Target Adaptation
```
python -m domainbed.scripts.unsupervised_adaptation\
       --input_dir=/home-local2/sasah14.extra.nobkp/scratch/out/ttdg/single_run/debug\
       --adapt_algorithm=T3A\
       --wandb_name te0_PACS_resnet18-BN_ERM_0
```
Wandb logs offline for IIDAccuracy and Oracle model selection startegies. Remove `wandb_name` to disable wandb logging.

### Sweep runs
The launch file has been modified substantially for easy use. Use the launch file for triggering sweep runs.
```
bash scripts/launch.sh <I:mode> <II:backbone> <III:n_trials_from> <IV:n_trials> <V:command launcher> <VI:dataset>  <VII:data path> <VIII:sweep path> <IX: sweep name> <X: DG algorithm> <XI: TTA algorithm(s)>
```

TTDG is performed in two steps:

#### 1. Source Pretraining
(exactly like Domainbed-only or DG-only setup):
```
bash scripts/launch.sh pretrain resnet18-BN 10 1 multi_gpu PACS iid_data_path iid_sweep_path debug ERM_TTT ""
```

#### 2. Target Adaptation
(Test time adaptation, similar to setup used in T3A)
```
# for all TTA algorithms
bash scripts/launch.sh unsup resnet18-BN 10 1 multi_gpu PACS iid_data_path iid_sweep_path debug ERM_TTT ""
# for specific TTA algorithm
bash scripts/launch.sh unsup resnet18-BN 10 1 multi_gpu PACS iid_data_path iid_sweep_path debug ERM_TTT T3A
```

### Sweep runs on compute canada
Use cc_run script interactively by first allocating
```
salloc --account=def-chgag196 --gres=gpu:1 --cpus-per-task=3 --mem=32000M --time=2:00:00
bash cc_run.sh
```

or submitting a job by:
```
sbatch cc_run.sh
```

## Collecting sweep results
```
# for all TTA algorithms
python -m domainbed.scripts.collect_results\
       --input_dir=/home-local2/sasah14.extra.nobkp/scratch/out/ttdg/sweep/debug/
# for specific TTA algorithm
python -m domainbed.scripts.collect_results\
       --input_dir=/home-local2/sasah14.extra.nobkp/scratch/out/ttdg/sweep/debug/
       --adapt_algorithms T3A
```

The original README is below:

---
---

# Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization

This codebase is the official implementation of `Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization` (NeurIPS2021, Spotlight, [url](<https://openreview.net/forum?id=e_yvNqkJKAW&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2021%2FConference%2FAuthors%23your-submissions)>). 
This codebase is mainly based on [DomainBed](https://github.com/facebookresearch/DomainBed), with following modifications:

* enable to use various backbone networks including Big Transfer (BiT), Vision Transformers (ViT, DeiT, HViT), and MLP-Mixer. 
* enable to test test-time adaptation method (T3A and [Tent](https://github.com/DequanWang/tent)). 

## Installation

#### CUDA/Python
```
git clone git@github.com:matsuolab/Domainbed_contrib.git
cd Domainbed_contrib/docker
docker build -t {image_name} .
docker run -it -h `hostname` --runtime=nvidia -v /path/to/Domainbed_contrib /path/to/anyware --shm-size=40gb --name {container_name} {image_name}
```

#### Python libralies
We use `pipenv` for package management. 
```
cd /path/to/Domainbed_contrib
pip install pipenv
pipenv install
pipenv shell
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
```

## Quick start
#### (1) Downlload the datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path --dataset pacs
```
Note: change `--dataset pacs` for downloading other datasets (e.g., `vlcs`, `office_home`, `terra_incognita`). 


#### (2) Train a model on source domains
```sh
python -m domainbed.scripts.train\
       --data_dir /my/datasets/path\
       --output_dir /my/pretrain/path\
       --algorithm ERM\
       --dataset PACS\
       --hparams "{\"backbone\": \"resnet50\"}" 
```
This scripts will produce new directory `/my/pretrain/path`, which include the full training log. 

Note: change `--dataset PACS` for training on other datasets (e.g., `VLCS`, `OfficeHome`, `TerraIncognita`). 

Note: change `--hparams "{\"backbone\": \"resnet50\"}"` for using other backbones (e.g., `resnet18`, `ViT-B16`, `HViT`). 


#### (3) Evaluate model with test time adaptation (Table 1, Table 2, Figure 2)
```sh
python -m domainbed.scripts.unsupervised_adaptation\
       --input_dir=/my/pretrain/path\
       --adapt_algorithm=T3A
```
This scripts will produce a new file in `/my/pretrain/path`, whose name is `results_{adapt_algorithm}.jsonl`. 

Note: change `--adapt_algorithm=T3A` for using other test time adaptation methods (`T3A`, `Tent`, or `TentClf`). 



#### (4) Evaluate model with fine-tuning classifier(Figure 1)
```sh
python -m domainbed.scripts.supervised_adaptation\
       --input_dir=/my/pretrain/path\
       --ft_mode=clf
```
This scripts will produce a new file in `/my/pretrain/path`, whose name is `results_{ft_mode}.jsonl`. 


## Available backbones

* resnet18
* resnet50
* BiT-M-R50x3
* BiT-M-R101x3
* BiT-M-R152x2
* ViT-B16
* ViT-L16
* DeiT
* Hybrid ViT (HViT)
* MLP-Mixer (Mixer-L16)

## Reproducing results
#### Table 1 and Figure 2 (Tuned ERM and CORAL)

You can use `scripts/hparam_search.sh`. Specifically, for each dataset and base algorithm, you can just type a following command.
```
sh scripts/hparam_search.sh resnet50 PACS ERM
```
Note that, it automatically starts 240 jobs, and take many times to finish. 


#### Table 2 and Figure 1 (ERM with various backbone)

You can use `scripts/launch.sh`. Specifically, for each backbone, you can just type following three commands. 
```
sh scripts/launch.sh pretrain resnet50 10 3 local
sh scripts/launch.sh sup resnet50 10 3 local
sh scripts/launch.sh unsup resnet50 10 3 local
```


#### Other results
For table 1, we used scores reported by [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434). 
Full results for the reported scores in LaTeX format available [here](domainbed/results/2020_10_06_7df6f06/results.tex).
Note: We only used scores for VLCS, PACS, OfficeHome, and TerraIncognita. We used the resutls with `IIDAccuracySelectionMethod`. 


## License

This source code is released under the MIT license, included [here](LICENSE).
