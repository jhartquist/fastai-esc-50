from utils import *
assert torch.cuda.is_available()

run_config = dict(
    # spectrum
    sample_rate=44100,
    hop_length=441,     # 10ms
    win_length_1=882,   # 20ms
    win_length_2=1764,  # 40ms
    win_length_3=3528,  # 80ms
    n_fft=8192,
    n_mels=256,
    f_max=15000,

    # model
    arch='resnet18',

    # training
    learning_rate=1e-2,
    n_epochs=80,
    batch_size=128,
    mix_up=0.4,
    
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

data = get_data(batch_tfms=[to_spectrum], 
                sample_rate=config.sample_rate,
                batch_size=config.batch_size,
                fold=config.fold,
                seed=config.trial_num)

arch = eval(config.arch)

learn = get_learner(data, arch, n_channels=3)

cbs = [
    MixUp(config.mix_up),
    WandbCallback(log_model=False, log_preds=False),
]

learn.fine_tune(config.n_epochs, 
                base_lr=config.learning_rate, 
                cbs=cbs)

wandb.finish()
