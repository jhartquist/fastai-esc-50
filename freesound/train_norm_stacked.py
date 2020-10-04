from fastai.vision.all import *
from fastaudio.core.all import *

import wandb
from fastai.callback.wandb import *

from sklearn.model_selection import StratifiedKFold

from utils import *

assert torch.cuda.is_available()

config = SimpleNamespace(
    # signal
    sample_rate=44100,
    duration=12,

    # spectrum
    n_fft=4096,
    n_mels=256,
    f_max=20000,
    hop_length=441,
    win_length_1=882,
    win_length_2=1764,
    win_length_3=3528,
    
    # model
    arch='resnet18',
    
    # training
    n_epochs=1,
    batch_size=32,
    mix_up=0.4,
    learning_rate=1e-2,

    # data
    num_folds=8,
    fold=0,
)

wandb.init(
    config=config, 
    project='fastaudio-freesound',
    save_code=True,
)

config = wandb.config
print("Config:", json.dumps(config.as_dict(), indent=2))

@patch_to(AudioTensor, cls_method=True)
def create(cls, fn, cache_folder=None, **kwargs):
    "Creates audio tensor from file"
    if cache_folder is not None:
        fn = cache_folder / fn.name
    try:
        sig, sr = torchaudio.load(fn, **kwargs)
    except:
        sig, sr = torch.zeros([1, 0]), config.sample_rate
    return cls(sig, sr=sr)

def get_df(csv_path, audio_path):
    df = pd.read_csv(csv_path)
    df['path'] = str(audio_path) + '/' + df.fname
    return df

path = Path('data/freesound')
df = get_df(path/'train.csv', path/'audio_train')
df_test = get_df(path/'sample_submission.csv', path/'audio_test')

skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=1)
val_idx_folds = [val_idx for trn_idx, val_idx in skf.split(df.index, df.label)]
val_idx = val_idx_folds[config.fold]


to_spectrum = StackedMelSpectrogram(
    n_fft=config.n_fft,
    n_mels=config.n_mels,
    sample_rate=config.sample_rate,
    win_lengths=[
        config.win_length_1,
        config.win_length_2,
        config.win_length_3,
    ],
    hop_length=config.hop_length,
    f_max=config.f_max)
audio_block = AudioBlock(
    sample_rate=config.sample_rate,
    crop_signal_to=config.duration * 1000)

dblock = DataBlock(blocks=(audio_block, CategoryBlock),  
                   get_x=ColReader('path'),
                   get_y=ColReader('label'),
                   splitter=IndexSplitter(val_idx),
                   batch_tfms = [to_spectrum])
dls = dblock.dataloaders(df, bs=config.batch_size)

stats = StatsRecorder()
with torch.no_grad():
    for x,y in iter(dls.train):
        stats.update(x)
        
spec_normalize = SpecNormalize(stats.mean, stats.std)
dls.after_batch.add(spec_normalize)

arch = eval(config.arch)
learn = cnn_learner(dls,
                    arch, 
                    normalize=False,
                    loss_fn=CrossEntropyLossFlat,
                    metrics=accuracy).to_fp16()


cbs = [
    MixUp(config.mix_up), 
    WandbCallback(log_model=False, log_preds=False),
]

learn.fine_tune(config.n_epochs, 
                base_lr=config.learning_rate,
                cbs=cbs)

test_dl = dls.test_dl(df_test)
test_preds = learn.get_preds(dl=test_dl)[0].numpy()

fname = f'fold_{config.fold}of{config.num_folds}_{config.n_epochs}_basic_norm.npy'
np.save(fname, test_preds)

wandb.finish()