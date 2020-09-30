from utils import *
assert torch.cuda.is_available()

run_config = dict(
    # spectrum
    sample_rate=32000,
    n_fft=4096,
    n_mels=128,
    hop_length_ms=10,
    win_length_ms=50,

    # model
    arch='resnet18',

    # training
    learning_rate=1e-2,
    n_epochs=20,
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

hop_length = int(config.hop_length_ms / 1000 * config.sample_rate)
win_length = int(config.win_length_ms / 1000 * config.sample_rate)

print(f"Hop: {hop_length}, Win: {win_length}")

audio_config = AudioConfig.BasicMelSpectrogram(
    sample_rate=config.sample_rate,
    hop_length=hop_length,
    win_length=win_length,
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

cbs = []
if config.mix_up: 
    cbs.append(MixUp(config.mix_up))
cbs.append(WandbCallback(log_model=False, log_preds=False))

learn.fine_tune(config.n_epochs, 
                base_lr=config.learning_rate, 
                cbs=cbs)

wandb.finish()