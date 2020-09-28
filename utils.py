from fastai.vision.all import *
from fastaudio.core.all import *

import wandb
from fastai.callback.wandb import *


path = untar_data(URLs.ESC50)

def get_data(sample_rate=16000, 
             item_tfms=None, 
             batch_tfms=None, 
             fold=1,
             batch_size=32,
             path=path,
             seed=1):
    set_seed(seed, True)
    df = pd.read_csv(path/'meta'/'esc50.csv')
    splitter = IndexSplitter(df[df.fold == fold].index)
    audio_block = AudioBlock(sample_rate=sample_rate)
    data_block = DataBlock(
        blocks=(audio_block, CategoryBlock),
        get_x=ColReader('filename', pref=path/'audio'),
        get_y=ColReader('category'),
        splitter=splitter,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms)
    data = data_block.dataloaders(df, bs=batch_size)
    return data

def get_learner(data, arch, n_channels=1, pretrained=True):
    return cnn_learner(data, arch,
                       config=cnn_config(n_in=n_channels),
                       pretrained=pretrained,
                       loss_fn=CrossEntropyLossFlat, 
                       metrics=accuracy).to_fp16()