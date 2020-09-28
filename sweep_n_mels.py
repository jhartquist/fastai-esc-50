from utils import *
assert torch.cuda.is_available()

# should be fixed by wandb==0.10.3
# https://github.com/wandb/client/issues/1248#issuecomment-696933966
os.environ['WANDB_SAVE_CODE'] = 'true'

run_config = dict(
    # spectrum
    sample_rate=16000,
    hop_length=160,
    win_length=800,
    n_fft=2048,
    n_mels=512,

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

cbs = []
if config.mix_up: 
    cbs.append(MixUp(config.mix_up))
cbs.append(WandbCallback(log_model=False, log_preds=False))

learn.fine_tune(config.n_epochs, 
                base_lr=config.learning_rate, 
                cbs=cbs)

wandb.finish()