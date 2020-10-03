from utils import *
assert torch.cuda.is_available()

run_config = dict(
    # spectrum
    sample_rate=44100,
    n_fft=4096,
    n_mels=224,
    hop_length=441,
    
    win_length_1=882,
    win_length_2=1764,
    win_length_3=3528,
    
    f_max=20000,

    # model
    arch='resnet18',

    # training
    learning_rate=1e-2,
    n_epochs=1,
    batch_size=32,
    mix_up=0.1,
    norm=True,
    
    # data
    trial_num=1,
    fold=1,
)

run = wandb.init(
    project='fastai-esc-50',
    config=run_config,
    save_code=True,
)

config = wandb.config

print("Config:", json.dumps(config.as_dict(), indent=2))

win_lengths = [
    config.win_length_1,
    config.win_length_2,
    config.win_length_3,
]

to_spectrum = StackedMelSpectrogram(
    n_fft=config.n_fft,
    n_mels=config.n_mels,
    sample_rate=config.sample_rate,
    win_lengths=win_lengths,
    hop_length=config.hop_length,
    f_max=config.f_max,
)

batch_tfms = [to_spectrum]

data = get_data(batch_tfms=batch_tfms, 
                sample_rate=config.sample_rate,
                batch_size=config.batch_size,
                fold=config.fold,
                seed=config.trial_num)

if config.norm:
    global_stats  = StatsRecorder()
    with torch.no_grad():
        for idx,(x,y) in enumerate(iter(data.train)):
            # update normalization statistics
            global_stats.update(x)
        
    global_mean,global_std = global_stats.mean,global_stats.std
    print(global_mean, global_std)
    norm = SpecNormalize(global_mean,  global_std,  axes=(0,2,3))
    batch_tfms.append(norm)

    data = get_data(batch_tfms=batch_tfms, 
                    sample_rate=config.sample_rate,
                    batch_size=config.batch_size,
                    fold=config.fold,
                    seed=config.trial_num)
    
    
arch = eval(config.arch)

learn = get_learner(data, arch, 
                    normalize=(not config.norm),
                    n_channels=3)

cbs = []
if config.mix_up: 
    cbs.append(MixUp(config.mix_up))
cbs.append(WandbCallback(log_model=False, log_preds=False))

learn.fine_tune(config.n_epochs, 
                base_lr=config.learning_rate, 
                cbs=cbs)

wandb.finish()