# Data Augmentation Causal Aplicado a Sistemas de Recomendação

Equipe: Aian Shay, Elves Rodrigues, Gabriel Luz

Este trabalho tem como objetivo reimplementar o framework CPR proposto pelo paper [Top-N Recommendation with Counterfactual User Preference Simulation](https://arxiv.org/pdf/2109.02444.pdf) e avaliar o desempenho no dataset MIND, construído no artigo [MIND: A Large-scale Dataset for News Recommendation](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf). O desempenho foi testado nos modelos de recomendação NMRS, LSTUR e DAE-GRU. A intuição básica é enriquecer os dados com novas interações contrafactuais usuário-item, produzidas a partir do framework causal de Pearl e, assim, treinar os modelos de recomendação de forma mais robusta.

Todo os códigos e notebooks utilizados estão nas pastas `src` e `notebooks`.


## How to run

Este projeto foi feito utilizando Jupyter Notebooks e o Google Colab e depois adaptado para rodar com o **Lightning-Hydra-Template**, para rodar:

Install dependencies

```bash
# clone project
git clone https://github.com/AegoCausalML/RecSysDataAugmentation.git
cd RecSysDataAugmentation
# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv
# install pytorch according to instructions
# https://pytorch.org/get-started/
# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu
# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

Notebooks ...
Dar mais instrucoes.
