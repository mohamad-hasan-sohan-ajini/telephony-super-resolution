import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda", index=0)

# # Dataset


class PairedAudioDataset(Dataset):
    def __init__(
        self,
        json_path,
        wav8k_dir="wav8k",
        wav16k_dir="wav16k",
        audio_duration=1.0,
        sample_rate=16000,
    ):
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.num_samples = int(audio_duration * sample_rate)

        self.wav8k_dir = Path(wav8k_dir)
        self.wav16k_dir = Path(wav16k_dir)

        with open(json_path) as f:
            self.filenames = json.load(f)

    def __len__(self):
        return len(self.filenames)

    def _load_and_crop(self, wav_name):
        path_8k = self.wav8k_dir / wav_name
        path_16k = self.wav16k_dir / wav_name

        waveform8k, sr = torchaudio.load(path_8k)
        waveform16k, sr = torchaudio.load(path_16k)

        if waveform8k.shape[1] > self.num_samples:
            # crop if longer
            max_offset = waveform8k.shape[1] - self.num_samples - 1
            offset = random.randint(0, max_offset)
            waveform8k = waveform8k[:, offset : offset + self.num_samples]
            waveform16k = waveform16k[:, offset : offset + self.num_samples]
        else:
            # pad if shorter
            waveform8k = torch.nn.functional.pad(
                waveform8k,
                (0, self.num_samples - waveform8k.shape[1]),
            )
            waveform16k = torch.nn.functional.pad(
                waveform16k,
                (0, self.num_samples - waveform8k.shape[1]),
            )
        return waveform8k, waveform16k

    def __getitem__(self, idx):
        wav_name = self.filenames[idx]
        x8k, x16k = self._load_and_crop(wav_name)
        return x8k, x16k


dataset = PairedAudioDataset("filelist.json")

for x8, x16 in dataset:
    break

x8.shape, x16.shape

# from matplotlib import pyplot as plt

# plt.figure(figsize=(12, 4))

# # Plot low-bandwidth audio (8k)
# plt.subplot(2, 1, 1)
# plt.plot(x8[0, :1024])
# plt.title("8kHz (Narrowband) Audio")
# plt.xlabel("Samples")
# plt.ylabel("Amplitude")

# # Plot full-bandwidth audio (16k)
# plt.subplot(2, 1, 2)
# plt.plot(x16[0, :1024])
# plt.title("16kHz (Wideband) Audio")
# plt.xlabel("Samples")
# plt.ylabel("Amplitude")

# plt.tight_layout()
# plt.show()

dataloader = DataLoader(dataset, batch_size=30, shuffle=True, num_workers=5)


class ResNetBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class AudioUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_layers=4):
        super().__init__()
        self.num_layers = num_layers

        # channel correction
        self.channel_expand = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(inplace=True),
            ResNetBlock1D(base_channels),
            ResNetBlock1D(base_channels),
            ResNetBlock1D(base_channels),
        )

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_layers):
            self.encoders.append(
                nn.Sequential(*[ResNetBlock1D(in_ch) for _ in range(3)])
            )
            out_ch = in_ch * 2
            self.downsamplers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(*[ResNetBlock1D(in_ch) for _ in range(3)])

        for i in range(100):
            print(f"{in_ch = }")
        # Decoder
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(num_layers)):
            self.upsamplers.append(
                nn.ConvTranspose1d(
                    in_ch,
                    in_ch // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            in_ch = in_ch // 2
            self.decoders.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_ch * 2,
                        in_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm1d(in_ch),
                    nn.LeakyReLU(inplace=True),
                    ResNetBlock1D(in_ch),
                    ResNetBlock1D(in_ch),
                    ResNetBlock1D(in_ch),
                )
            )

        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv1d(base_channels, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        skips = []

        # Encoder path
        x = self.channel_expand(x)
        # print(f"start {x.shape}")
        for down, enc in zip(self.downsamplers, self.encoders):
            x = enc(x)
            skips.append(x)
            # print(f"enc {x.shape}")
            x = down(x)
            # print(f"down {x.shape}")
            # print("-" * 40)

        # Bottleneck
        x = self.bottleneck(x)
        # print(f"bottleneck {x.shape}")
        # print("-" * 40)

        for skip in skips:
            # print(f"skip {skip.shape}")
            pass
        # print("-" * 40)

        # Decoder path
        for up, dec, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            x = up(x)
            # print(f"up {x.shape}")
            # In case of mismatch due to rounding in downsampling
            if x.shape[-1] != skip.shape[-1]:
                # print(f"ifskip {skip.shape = } {x.shape = }")
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            # print(f"=> skip {skip.shape}, x {x.shape}")
            # x = x + skip
            x = torch.cat([x, skip], dim=1)
            # print(f"cat {x.shape}")
            x = dec(x)
            # print(f"dec {x.shape}")
            # print("-" * 40)

        # Output
        return self.output_conv(x)


model = AudioUNet().to(device)
x = torch.randn(4, 1, 16000).to(device)
y = model(x)


x.shape, y.shape

# # Training loop


criterion = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
writer = SummaryWriter(log_dir="runs/exp01")

for epoch in range(10000):
    print(f"{ epoch = }")
    for step, (x8k, x16k) in enumerate(tqdm(dataloader, desc="Training")):
        x8k, x16k = x8k.to(device), x16k.to(device)

        optimizer.zero_grad()
        pred = model(x8k)
        loss = criterion(pred, x16k)
        loss.backward()
        optimizer.step()

        # 👇 Log to TensorBoard
        writer.add_scalar("Loss/train", loss.item(), step)

    torch.save(
        model.state_dict(),
        f"checkpoints/model_epoch{epoch:04d}.pth",
    )

    # Optional console print
    # if step % 10 == 0:
    #     print(f"[Step {step}] Loss: {loss.item():.6f}")

writer.close()
