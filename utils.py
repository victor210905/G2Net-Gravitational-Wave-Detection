import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import scipy.signal
import scipy.interpolate
from scipy.signal.windows import tukey

try:
    from nnAudio.Spectrogram import CQT1992v2
except ImportError:
    pass

SOS_FILTER = scipy.signal.butter(4, [20, 500], btype="bandpass", output="sos", fs=2048)
NORMALIZATION_FACTOR = np.sqrt((500 - 20) / (2048 / 2))
WINDOW_TUKEY = tukey(4096, alpha=0.1)


class ProjectConfig:
    """Centralized configuration for paths."""
    ROOT_DIR = '/kaggle/working'
    DATA_DIR = '/kaggle/input/g2net-gravitational-wave-detection/train'
    CSV_PATH = '/kaggle/input/g2net-gravitational-wave-detection/training_labels.csv'

def seed_everything(seed=42):
    """Ensures reproducibility across runs."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True 
        torch.backends.cudnn.deterministic = False

# --- Signal Processing (High Performance) ---

def apply_bandpass(x):
    return scipy.signal.sosfiltfilt(SOS_FILTER, x) / NORMALIZATION_FACTOR

def apply_whitening(x, sr=2048):
    # 1. Windowing
    x = x * WINDOW_TUKEY
    
    # 2. Calculate PSD (Power Spectral Density)
    freqs, psd = scipy.signal.welch(x, fs=sr, nperseg=sr, window='hann')
    
    # 3. FFT (Fast Fourier Transform)
    x_f = np.fft.rfft(x)
    freqs_f = np.fft.rfftfreq(len(x), d=1/sr)
    
    valid_indices = (freqs_f >= freqs.min()) & (freqs_f <= freqs.max())
    x_f_whitened = np.zeros_like(x_f)
    
    # 4. Interpolation
    psd_values = np.interp(freqs_f[valid_indices], freqs, psd)
    
    # 5. Normalize & IFFT
    x_f_whitened[valid_indices] = x_f[valid_indices] / np.sqrt(psd_values + 1e-20)
    return np.fft.irfft(x_f_whitened, n=len(x))

def apply_timeshift(x, max_shift=0.2, sr=2048):
    shift_amt = int(random.random() * max_shift * sr)
    if random.random() > 0.5:
        shift_amt = -shift_amt
    return np.roll(x, shift_amt, axis=-1)

# --- Model Components ---

class WaveToImage(nn.Module):
    def __init__(self, sr=2048, fmin=20, fmax=1024, hop_length=64, device='cuda'):
        super().__init__()
        self.transform = CQT1992v2(sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length, 
                                   output_format="Magnitude", verbose=False)
    
    def forward(self, x):
        batch_size, channels, time_steps = x.shape
        x = x.view(batch_size * channels, time_steps)
        images = self.transform(x)
        _, h, w = images.shape
        return images.view(batch_size, channels, h, w)

class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, stage='train', augment=False):
        self.df = dataframe
        self.root_dir = root_dir
        self.stage = stage
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row['id'])
        # Organization of G2Net: a/b/c/abc0123.npy
        path_part = f"{file_id[0]}/{file_id[1]}/{file_id[2]}"
        file_path = os.path.join(self.root_dir, path_part, f"{file_id}.npy")
        
        try:
            waves = np.load(file_path).astype(np.float32)
            
            if self.stage == 'train' and self.augment:
                waves = apply_timeshift(waves)

            cleaned_waves = []
            for i in range(3):
                w = waves[i]
                w = apply_bandpass(w)
                w = apply_whitening(w)
                w = w / (np.std(w) + 1e-20) 
                
                cleaned_waves.append(w)
            
            waves = np.stack(cleaned_waves)
            waves = torch.tensor(waves, dtype=torch.float32)
            
        except FileNotFoundError:
            waves = torch.zeros((3, 4096), dtype=torch.float32)

        if self.stage != 'test':
            return waves, torch.tensor(row['target'], dtype=torch.float32)
        return waves
