from utils import *
assert torch.cuda.is_available()

run_config = dict(
    # spectrum
    sample_rate=44100,
    n_fft=4096,
    n_mels=224,
    hop_length_ms=10,
    win_length_ms=50,
    f_max=20000,

    # model
    arch='resnet18',

    # training
    learning_rate=1e-2,
    n_epochs=20,
    batch_size=64,
    mix_up=0.1,
    norm=False,
    
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
    f_max=config.f_max,
)

to_spectrum = AudioToSpec.from_cfg(audio_config)

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

learn = get_learner(data, arch, normalize=(not config.norm))

cbs = []
if config.mix_up: 
    cbs.append(MixUp(config.mix_up))
cbs.append(WandbCallback(log_model=False, log_preds=False))

learn.fine_tune(config.n_epochs, 
                base_lr=config.learning_rate, 
                cbs=cbs)

wandb.finish()