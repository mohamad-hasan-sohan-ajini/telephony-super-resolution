import json
from multiprocessing import Pool
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


def compute_gain_vector(freq: torch.Tensor, f_start: float = 2000, f_end: float = 3500):
    gain = 0.5 + 0.5 * torch.cos(
        torch.pi / 2 * (freq.abs() - f_start) / (f_end - f_start)
    )
    high_mask = freq.abs() > f_end
    gain[high_mask] = 0.0
    low_mask = freq.abs() < f_start
    gain[low_mask] = 1.0
    return gain


def apply_channel_effect(wav_path):
    save_path = Path("wav8k")
    down_sampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
    up_sampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
    cutoff_freq = 3500

    x, fs = torchaudio.load(wav_path)
    x = down_sampler(x)
    x = up_sampler(x)

    # fft
    fft = torch.fft.fft(x.squeeze(0))

    # Frequency bin resolution
    n = len(x.squeeze(0))
    freqs = torch.fft.fftfreq(n, d=1 / fs)

    # Create gain vector
    gain = compute_gain_vector(freqs)

    # apply mask
    fft_filtered = fft * gain

    # Inverse FFT (back to time domain)
    filtered_waveform = torch.fft.ifft(fft_filtered).real.unsqueeze(0)

    filtered_waveform = torchaudio.functional.lowpass_biquad(
        filtered_waveform,
        fs,
        cutoff_freq=cutoff_freq,
    )

    # Save result
    torchaudio.save(save_path / wav_path.name, filtered_waveform, fs)


base_path = Path("wav16k/")
files = sorted(base_path.glob("*.wav"))
print(len(files))
with open("filelist.json", "w") as f:
    json.dump([str(i) for i in files], f, indent=4)

# with Pool(6) as p:
#     for wav_path in tqdm(p.imap(apply_channel_effect, files), total=len(files)):
#         pass

for wav_path in tqdm(files):
    apply_channel_effect(wav_path)
