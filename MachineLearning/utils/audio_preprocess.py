import torch
import torchaudio

def preprocess_audio(model_name:str, file_path: str, max_len: float=3.0, sample_rate: int=16000):
    """
    Tiền xử lý audio cho các model khác nhau.
    args: 
        model_name: tên model để xác định config
        file_path: đường dẫn tới file âm thanh
        max_len: độ dài tối đa của đoạn audio (giây)
        sample_rate: tần số mẫu mong muốn (Hz)
    Returns:
        features: list các tensor (đã được tiền xử lý)
    """
    # ==== Load audio ====
    try:
        signal, fs = torchaudio.load(file_path)
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")
    
    # stereo to mono
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    if fs != sample_rate:
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=sample_rate)(signal)

    signal = signal.squeeze(0) # [T]
    total_len = signal.size(0)                # độ dài của audio (số mẫu)
    max_samples = int(max_len * sample_rate)  # độ dài tối đa (số mẫu)

    # ==== 2. Chia audio thành các đoạn nhỏ hơn ====
    if total_len > max_samples:
        n_chunks = total_len // max_samples
        chunks = [signal[i*max_samples:(i+1)*max_samples] for i in range(n_chunks)]
        if total_len % max_samples != 0:
            tmp = signal[-max_samples:]
            tmp = torch.nn.functional.pad(tmp, (0, max_samples - tmp.size(0)))
            chunks.append(tmp)
    else:
        pad = max_samples - total_len
        signal = torch.nn.functional.pad(signal, (0, pad))
        chunks = [signal]

    # ==== 3. Xử lý theo model ====
    features = []
    for chunk in chunks:
        if model_name.lower() == "cnn_rnn" or model_name.lower() == "ecapa":
            # FBank
            feats = torchaudio.compliance.kaldi.fbank(
                waveform=chunk.unsqueeze(0),    # [1, T]
                num_mel_bins=80,                # số lượng Mel bins
                sample_frequency=sample_rate,   # tần số mẫu
                window_type='hamming',          # loại cửa sổ    
                dither=1e-5,                    # thêm dithering để giảm thiểu các artefacts
                energy_floor=0.0
            )
            # CMVN
            feats = (feats - feats.mean(dim=0)) / (feats.std(dim=0) + 1e-5)
            features.append(feats)  # [T', 80]
        elif model_name.lower() == "wavlm":
            if chunk.abs().max() > 0:
                chunk = chunk / chunk.abs().max()  # Chuẩn hóa về [-1, 1]
            features.append(chunk)  # [T]
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
    return features  # List of tensors, mỗi tensor có shape tùy theo model