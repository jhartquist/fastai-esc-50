from fastai.vision.all import * 
from fastaudio.core.all import *
from fastaudio.augment.all import *

# from https://github.com/fastaudio/Audio-Competition/blob/master/ESC-50-baseline-1Fold.ipynb
def CrossValidationSplitter(col='fold', fold=1):
    "Split `items` (supposed to be a dataframe) by fold in `col`"
    def _inner(o):
        assert isinstance(o, pd.DataFrame), "ColSplitter only works when your items are a pandas DataFrame"
        col_values = o.iloc[:,col] if isinstance(col, int) else o[col]
        valid_idx = (col_values == fold).values.astype('bool')
        return IndexSplitter(mask2idxs(valid_idx))(o)
    return _inner

class StackedMelSpecs(Transform):
    "Stacks Mel spectrograms with different resolutions into a single image."
    
    def __init__(self, n_fft, n_mels, sample_rate, win_lengths, hop_length, f_max=None):
        store_attr()
        # mel spectrum extractors
        assert max(win_lengths) <= n_fft
        self.specs = [AudioToSpec.from_cfg(
            AudioConfig.BasicMelSpectrogram(n_fft=n_fft,
                                            hop_length=hop_length,
                                            win_length=win_length,
                                            normalized=True,
                                            n_mels=n_mels,
                                            f_max=f_max,
                                            sample_rate=sample_rate)) 
                      for win_length in win_lengths]

    def encodes(self, x: AudioTensor) -> AudioSpectrogram:       
        return AudioSpectrogram(torch.cat([spec(x) for spec in self.specs], axis=1))
    
def mapk_np(preds, targs, k=3):
    preds = np.argsort(-preds, axis=1)[:, :k]
    score = 0.0
    for i in range(k):
        num_hits = (preds[:, i] == targs).sum()
        score += num_hits * (1.0 / (i+1.0))
    score /= preds.shape[0]
    return score

def mapk(preds, targs, k=3):
    return mapk_np(to_np(preds), to_np(targs), k)


# https://enzokro.dev/spectrogram_normalizations/2020/09/10/Normalizing-spectrograms-for-deep-learning.html
class SpecNormalize(Normalize):
    "Normalize/denorm batch of `TensorImage`"
    def encodes(self, x:TensorImageBase): return (x-self.mean) / self.std
    def decodes(self, x:TensorImageBase):
        f = to_cpu if x.device.type=='cpu' else noop
        return (x*f(self.std) + f(self.mean))
    
class StatsRecorder:
    def __init__(self, red_dims=(0,2,3)):
        """Accumulates normalization statistics across mini-batches.
        ref: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        self.red_dims = red_dims # which mini-batch dimensions to average over
        self.nobservations = 0   # running number of observations

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        # initialize stats and dimensions on first batch
        if self.nobservations == 0:
            self.mean = data.mean(dim=self.red_dims, keepdim=True)
            self.std  = data.std (dim=self.red_dims,keepdim=True)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            if data.shape[1] != self.ndimensions:
                raise ValueError('Data dims do not match previous observations.')
            
            # find mean of new mini batch
            newmean = data.mean(dim=self.red_dims, keepdim=True)
            newstd  = data.std(dim=self.red_dims, keepdim=True)
            
            # update number of observations
            m = self.nobservations * 1.0
            n = data.shape[0]

            # update running statistics
            tmp = self.mean
            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = torch.sqrt(self.std)
                                 
            # update total number of seen samples
            self.nobservations += n