#!/usr/bin/env bash -l
# sets up conda environment in linux 
# WARNING: HAS NOT BEEN TESTED, SORRY! PLEASE LMK
# IF YOU TRIED THIS AND IT DOESN'T WORK
python -m venv .neufa_finetune
source .venv/bin/activate
pip install torch
pip install pandas numpy matplotlib
pip install librosa 
pip install tqdm
pip install torchaudio
pip install jupyter
pip install swig sequitur-g2p
git clone --recursive https://github.com/petronny/g2p