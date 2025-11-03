import torch
from fastapi import APIRouter
from pydantic import BaseModel
from core.model_loader import ModelLoader
from utils.audio_preprocess import preprocess_audio

router = APIRouter()

# Bước 1: Định nghĩa schema cho JSON body
class VoiceVerifyRequest(BaseModel):
    model_name: str
    ckpt_path: str
    file_path: str  # hoặc đổi thành audio_base64 nếu bạn muốn gửi file base64 sau này

# Bước 2: Sửa endpoint nhận vào JSON thay vì query params
@router.post("/verify_voice/")
async def verify_voice(request: VoiceVerifyRequest):
    """
    Xác thực giọng nói - Kiểm tra deepfake (fake/real).
    Nhận dữ liệu dưới dạng JSON body.
    """
    try:
        model_name = request.model_name
        ckpt_path = request.ckpt_path
        file_path = request.file_path

        #Load model deepfake
        model_loader = ModelLoader(
            model_name=model_name,
            num_classes=2,
            ckpt_path=ckpt_path,
            model_type='deepfake'
        )
        model = model_loader.get_model()
        device = model_loader.device
        
        # Preprocess audio
        features = preprocess_audio(
            model_name=model_name,
            file_path=file_path,
            max_len=3.0,
            sample_rate=16000
        )
        
        # Predict cho từng chunk
        model.eval()
        scores = []
        for feat in features:
            feat_batch = feat.unsqueeze(0).to(device)
            with torch.no_grad():
                if model_name.lower() == 'wavlm':
                    lengths = torch.tensor([feat.size(0)], device=device)
                    logits, _ = model(feat_batch, lengths)
                else:
                    logits, _ = model(feat_batch)
            prob = torch.sigmoid(logits).item()
            scores.append(prob)
        
        # Tổng hợp kết quả
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
