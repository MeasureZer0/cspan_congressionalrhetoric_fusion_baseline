import os
import warnings
from typing import Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


class MultimodalClassificationDataset(Dataset):
    """
    Supervised dataset for multimodal classification.

    Expected files:
      - text_dir/text_data_all.json
      - text_dir/train.csv / val.csv / test.csv
      - video_dir/{video_id}_faces.pt
      - audio_dir/{video_id}.wav
    """

    def __init__(
        self,
        text_dir: str,
        video_dir: str,
        audio_dir: str,
        tokenizer=None,
        split: str = "train",
        max_text_length: int = 256,
        audio_sample_rate: int = 16000,
        skip_validation: bool = False,
    ):
        self.text_dir = text_dir
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.max_length = max_text_length
        self.audio_sample_rate = audio_sample_rate
        self.split = split

        self.data = self._load_and_filter_data(skip_validation)
        print(f"[{split}] Dataset initialized: {len(self.data)} valid samples")

    def _load_and_filter_data(self, skip_validation: bool) -> pd.DataFrame:
        json_path = os.path.join(self.text_dir, "text_data_all.json")
        df = pd.read_json(json_path, orient="index")
        df = df.reset_index().rename(columns={"index": "filename"})
        split_map = {
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv",
        }
        csv_path = os.path.join(self.text_dir, split_map[self.split])
        split_df = pd.read_csv(csv_path)
        if "label" in df.columns:
            df = df.drop(columns=["label"])
        df = pd.merge(df, split_df[["filename", "label"]], on="filename", how="inner")

        label_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }
    
        if df["label"].dtype == object:
            df["label"] = df["label"].str.lower().map(label_map)
    
        df = df.dropna(subset=["label", "transcription"])
        
        df["label"] = df["label"].astype(int)
        df = df[df["label"].notna() & df["transcription"].notna()]
        df = df.set_index("filename")
        print(df)
        if skip_validation:
            return df

        valid_indices = []
        missing_stats = {"video": 0, "audio": 0, "both": 0}
        for filename in df.index:
            video_id = filename.split(".")[0] if "." in filename else filename
            video_path = os.path.join(self.video_dir, "self-supervised", f"{video_id}_faces.pt")
            pose_path = os.path.join(self.video_dir, "pose-self-supervised", f"{video_id}_pose.pt")
            audio_path = os.path.join(self.audio_dir, f"{video_id}.wav")

            has_video = os.path.exists(video_path)
            has_pose = os.path.exists(pose_path)
            has_audio = os.path.exists(audio_path)
            if has_video and has_audio and has_pose:
                valid_indices.append(filename)
            else:
                if not has_video and not has_audio:
                    missing_stats["both"] += 1
                elif not has_video:
                    missing_stats["video"] += 1
                else:
                    missing_stats["audio"] += 1

        total_missing = sum(missing_stats.values())
        if total_missing > 0:
            warnings.warn(
                f"Filtered out {total_missing}/{len(df)} samples:\n"
                f"  - Missing video only: {missing_stats['video']}\n"
                f"  - Missing audio only: {missing_stats['audio']}\n"
                f"  - Missing both: {missing_stats['both']}"
            )

        return df.loc[valid_indices]

    def __len__(self):
        return len(self.data)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.audio_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.audio_sample_rate)
            waveform = resampler(waveform)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform.float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename = self.data.index[idx]
        row = self.data.iloc[idx]
        video_id = filename.split(".")[0] if "." in filename else filename

        video_path = os.path.join(self.video_dir, f"{video_id}_faces.pt")
        audio_path = os.path.join(self.audio_dir, f"{video_id}.wav")

        video_item = torch.load(video_path, weights_only=False)
        audio_waveform = self._load_audio(audio_path)

        transcript = row["transcription"]
        text_data = {
            "video_id": video_id,
            "raw_text": transcript,
        }

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                transcript,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_data["input_ids"] = encoding["input_ids"].squeeze(0)
            text_data["attention_mask"] = encoding["attention_mask"].squeeze(0)

        label = int(row["label"])

        return {
            "text": text_data,
            "video": video_item,
            "audio": audio_waveform,
            "label": torch.tensor(label, dtype=torch.long),
            "meta": {"chunks": row.get("timestamped_chunks", [])},
        }