import torch
from torch.nn.utils.rnn import pad_sequence

def collate_spectrogram(batch):
    """
    Collate function cho spectrogram (cho các model CNN_RNN, ECAPA).
    Args:
        batch: list các tuple (features, label)
            - features: tensor [T, F]
            - label: int
    Returns:
        padded_features: tensor [B, T_max, F]
        labels: tensor [B]
    """
    try:
        features, labels = zip(*batch)
        feats_padded = pad_sequence([f for f in features], batch_first=True) # [B, T_max, F]
        labels = torch.tensor(labels, dtype=torch.long) # [B]
        return feats_padded, labels
    except Exception as e:
        print(f"[ERROR] collate_spectrogram failed: {e}")
        return None, None

def collate_waveform(batch):
    """
    Collate function cho waveform (cho model WavLM).
    Args:
        batch: list các tuple (waveform, length, label)
            - signal: tensor [T]
            - length: int
            - label: int
    Returns:
        signals: tensor [B, T]
        lengths: tensor [B]
        labels: tensor [B]
    """
    try:
        signals, lengths, labels = zip(*batch)
        signals = torch.stack(signals) # [B, T]
        labels = torch.tensor(labels, dtype=torch.long) # [B]
        lengths = torch.tensor(lengths, dtype=torch.long) # [B]
        return signals, lengths, labels
    except Exception as e:
        print(f"[ERROR] collate_waveform failed: {e}")
        return None, None, None