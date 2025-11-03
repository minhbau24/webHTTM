import torch
import torchaudio
from torch.utils.data import Dataset

# Dataset cho bài toán nhận diện người nói / phát hiện deepfake
class SpeakerDataset(Dataset):
    def __init__(self, df, model_name, sample_rate=16000, n_mels=80, max_len=3.0, augment=False):
        self.df = df
        self.model_name = model_name.lower()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_len = max_len
        self.max_samples = int(max_len * sample_rate)
        self.augment = augment

    def __len__(self):
        return len(self.df)
    
    def __load_resample(self, path):
        try:
            sig, fs = torchaudio.load(path)  # [C, T]
        except Exception as e:
            print(f"Warning: Could not load {path}. Error: {e}")
            # Trả về một tensor rỗng nếu không thể tải file
            return torch.zeros(self.max_samples)

        if sig.shape[0] > 1:
            sig = torch.mean(sig, dim=0, keepdim=True)
        if fs != self.sample_rate:
            sig = torchaudio.transforms.Resample(orig_freq=fs, new_freq=self.sample_rate)(sig)
        
        return sig.squeeze(0) # [T]

    def __getitem__(self, index):
        row = self.df[index]
        path, label = row['file_path'], row['label']
        signal = self.__load_resample(path)
        L = signal.size(0) # độ dài của audio (số mẫu)
        # crop/pad to fixed length
        if L > self.max_samples: 
            if self.augment:
                start = torch.randint(0, L - self.max_samples + 1, (1,)).item()
                seg = signal[start:start + self.max_samples]
            else:
                seg = signal[:self.max_samples]
            length = self.max_samples
        else:
            pad = self.max_samples - L
            seg = torch.nn.functional.pad(signal, (0, pad))
            length = L
        
        if self.model_name in ['cnn_rnn', 'ecapa']:
            # Mel-filterbank features (FBank) — là ma trận biểu diễn năng lượng theo thời gian và theo tần số Mel.
            feats = torchaudio.compliance.kaldi.fbank(
                waveform=seg.unsqueeze(0),      # [1, T]
                num_mel_bins=self.n_mels,      # số lượng Mel bins
                sample_frequency=self.sample_rate, # tần số mẫu
                window_type='hamming',          # loại cửa sổ
                dither=1e-5,                    # thêm dithering để giảm thiểu các artefacts
                energy_floor=0.0                # ngưỡng năng lượng
            ) # [T', n_mels]
            feats = (feats - feats.mean(dim=0)) / (feats.std(dim=0) + 1e-5) # CMVN
            return feats, label
        elif self.model_name == 'wavlm':
            # normalize to [-1, 1]
            if seg.abs().max() > 0:
                seg = seg / seg.abs().max()
            return seg, length, label
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")