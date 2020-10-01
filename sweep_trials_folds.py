from utils import *
assert torch.cuda.is_available()

run_config = dict(
    # spectrum
    sample_rate=22050,
    hop_length=221,  # 20ms
    win_length=1103, # 50ms
    n_fft=2048,
    n_mels=224,

    # model
    arch='resnet18',

    # training
    learning_rate=1e-2,
    n_epochs=80,
    batch_size=64,
    mix_up=0.1,
    
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

audio_config = AudioConfig.BasicMelSpectrogram(
    sample_rate=config.sample_rate,
    hop_length=config.hop_length,
    win_length=config.win_length,
    n_fft=config.n_fft,
    n_mels=config.n_mels,
    normalized=True,
)

to_spectrum = AudioToSpec.from_cfg(audio_config)
data = get_data(batch_tfms=[to_spectrum], 
                sample_rate=config.sample_rate,
                batch_size=config.batch_size,
                fold=config.fold,
                seed=config.trial_num)

arch = eval(config.arch)

learn = get_learner(data, arch)

cbs = [
    MixUp(),
    WandbCallback(log_model=False, log_preds=False),
]

learn.fine_tune(config.n_epochs, 
                base_lr=config.learning_rate, 
                cbs=cbs)

wandb.finish()
