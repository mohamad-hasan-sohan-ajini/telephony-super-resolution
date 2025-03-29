import json
import random
from pathlib import Path

import shutil
from tqdm import tqdm

base_path = Path("/data2/asr/cv-corpus-8.0-2022-01-19/en/wavclips16k/")
files = list(base_path.glob("*.wav"))
print(len(files))

agg_dataset = json.loads(
    Path("/home/aj/repo/baden_asr/train_dataset_aggregation.json").read_text()
)
mozilla_dataset = [i for i in tqdm(agg_dataset) if "common_voice_en" in i["path"]]
random.shuffle(mozilla_dataset)
mozilla_dataset[:3]

sum([i["duration"] for i in mozilla_dataset]) / 3600

save_path = Path("wav16k")
remove_files = list(save_path.glob("*.wav"))
for i in tqdm(remove_files):
    Path.unlink(i)

src_wav_base_path = Path("/data2/asr/")

selection = []
for i in tqdm(mozilla_dataset):
    if random.random() < 0.06:
        # copy file from i["path"] to save_path
        src = src_wav_base_path / i["path"]
        dst = save_path / src.name
        shutil.copy(src, dst)
        selection.append(i)


sum([i["duration"] for i in selection]) / 3600
