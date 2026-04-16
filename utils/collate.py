from typing import Any, Dict, List, Tuple
import torch

MAX_AUDIO_SAMPLES = 16_000 * 30


def _standardize_video_item(
    video_item: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expected per-sample formats:
      1) dict with keys: faces, pose, lengths
      2) tuple/list: (faces, pose, lengths)

    Returns:
      faces:   [T, ...]
      pose:    [T, ...]
      lengths: scalar tensor
    """
    if isinstance(video_item, dict):
        faces = video_item["faces"]
        pose = video_item["pose"]
        lengths = video_item["lengths"]
    elif isinstance(video_item, (tuple, list)) and len(video_item) == 3:
        faces, pose, lengths = video_item
    else:
        raise ValueError(
            "Unsupported video item format. Expected dict with faces/pose/lengths "
            "or tuple/list (faces, pose, lengths)."
        )

    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, dtype=torch.long)

    if lengths.ndim > 0:
        lengths = lengths.squeeze()

    return faces, pose, lengths


def _pad_sequence_tensors(
    seq_list: List[torch.Tensor], pad_value: float = 0.0
) -> torch.Tensor:
    """
    Pad list of [T, ...] tensors to [B, T_max, ...].
    """
    max_len = max(x.shape[0] for x in seq_list)
    out_shape = (len(seq_list), max_len) + tuple(seq_list[0].shape[1:])
    out = seq_list[0].new_full(out_shape, pad_value)

    for i, x in enumerate(seq_list):
        out[i, : x.shape[0]] = x
    return out


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = torch.stack([item["label"] for item in batch], dim=0)

    # text
    input_ids = torch.stack([item["text"]["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack(
        [item["text"]["attention_mask"] for item in batch], dim=0
    )

    # audio: each sample is [1, T]
    audio_list = [item["audio"].squeeze(0)[:MAX_AUDIO_SAMPLES] for item in batch]
    audio_lengths = torch.tensor([x.shape[-1] for x in audio_list], dtype=torch.long)
    max_audio_len = int(audio_lengths.max().item())
    padded_audio = audio_list[0].new_zeros((len(audio_list), max_audio_len))
    audio_attention_mask = torch.zeros(
        (len(audio_list), max_audio_len), dtype=torch.long
    )

    for i, x in enumerate(audio_list):
        L = x.shape[-1]
        padded_audio[i, :L] = x
        audio_attention_mask[i, :L] = 1

    # video
    faces_list, pose_list, video_lengths_list = [], [], []
    for item in batch:
        faces, pose, lengths = _standardize_video_item(item["video"])
        faces_list.append(faces)
        pose_list.append(pose)
        video_lengths_list.append(lengths)

    faces = _pad_sequence_tensors(faces_list, pad_value=0.0)
    pose = _pad_sequence_tensors(pose_list, pad_value=0.0)
    video_lengths = torch.stack(video_lengths_list, dim=0).long()

    return {
        "text": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        "audio": {
            "waveform": padded_audio,  # [B, T]
            "attention_mask": audio_attention_mask,  # [B, T]
            "lengths": audio_lengths,
        },
        "video": {
            "faces": faces,
            "pose": pose,
            "lengths": video_lengths,
        },
        "label": labels,
    }
