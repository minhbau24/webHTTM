import torch
from fastapi import APIRouter
from core.model_loader import ModelLoader
from utils.audio_preprocess import preprocess_audio

router = APIRouter()

@router.post("/verify_voice/")
async def verify_voice(model_name: str, ckpt_path: str, file_path: str):
    """
    Xác thực giọng nói - Kiểm tra deepfake (fake/real).
    Args:
        model_name: tên model để xác định config (cnn_rnn, ecapa, wavlm)
        ckpt_path: đường dẫn checkpoint của model
        file_path: đường dẫn file audio cần kiểm tra

    Returns:
        is_fake: True nếu là fake, False nếu là real
        confidence: độ tin cậy (0-1)
        scores: điểm số cho từng chunk
    """
    try:
        # 1. Load model deepfake
        model_loader = ModelLoader(
            model_name=model_name,
            num_classes=2,
            ckpt_path=ckpt_path,
            model_type='deepfake'
        )
        model = model_loader.get_model()
        device = model_loader.device
        
        # 2. Preprocess audio
        features = preprocess_audio(
            model_name=model_name,
            file_path=file_path,
            max_len=3.0,
            sample_rate=16000
        )
        
        # 3. Predict cho từng chunk
        model.eval()
        scores = []
        
        for feat in features:
            feat_batch = feat.unsqueeze(0).to(device)  # [1, T, F] hoặc [1, T]
            
            with torch.no_grad():
                if model_name.lower() == 'wavlm':
                    # WavLM cần lengths
                    lengths = torch.tensor([feat.size(0)], device=device)
                    logits, _ = model(feat_batch, lengths)
                else:
                    logits, _ = model(feat_batch)
            
            # Binary classification với BCEWithLogitsLoss
            prob = torch.sigmoid(logits).item()
            scores.append(prob)
        
        # 4. Kết quả tổng hợp
        avg_score = sum(scores) / len(scores)
        is_fake = avg_score >= 0.5
        
        return {
            "is_fake": is_fake,
            "confidence": avg_score,
            "prediction": "FAKE" if is_fake else "REAL",
            "num_chunks": len(features),
            "chunk_scores": scores
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "is_fake": None,
            "confidence": 0.0
        }
