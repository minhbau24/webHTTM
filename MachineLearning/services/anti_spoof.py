import torch
from core.model_loader import ModelLoader
from utils.audio_preprocess import preprocess_audio

def check_anti_spoof(input_audio_path, model_name):
    """
    Kiểm tra audio giả mạo (deepfake) hay thật.
    Args:
        input_audio_path: đường dẫn file audio đầu vào
        model_name: tên model để xác định config
    Returns:
        prediction: 0 (real) hoặc 1 (fake)
        score: điểm số xác suất (0-1)
    """
    # load model
    model, device = ModelLoader(model_name=model_name,  model_type='deepfake')

    # preprocess input audio
    features = preprocess_audio(model_name=model_name, file_path=input_audio_path)

    # predict
    model.eval()
    with torch.no_grad():
        feat = features[0].unsqueeze(0).to(device)  # [1, T, F]
        logits, _ = model(feat)
        probs = torch.softmax(logits, dim=1)
        score = probs[0, 1].item()  # xác suất là fake
        prediction = 1 if score >= 0.5 else 0

    return prediction, score