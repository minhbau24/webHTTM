import torch
from fastapi import APIRouter
from pydantic import BaseModel
from core.model_loader import ModelLoader
from utils.audio_preprocess import preprocess_audio
from utils.feature_extraction import extract_embedding

router = APIRouter()


# 🧩 Định nghĩa model cho JSON input
class RegisterVoiceRequest(BaseModel):
    model_name: str
    ckpt_path: str
    file_path: str
    num_classes: int | None = None


@router.post("/register_voice/")
async def register_voice(request: RegisterVoiceRequest):
    """
    Đăng ký giọng nói mới bằng cách trích xuất embedding từ file audio.
    Input: JSON có các trường model_name, ckpt_path, file_path, num_classes
    """
    try:
        # 1️⃣ Load model
        model_loader = ModelLoader(
            model_name=request.model_name,
            num_classes=request.num_classes,
            ckpt_path=request.ckpt_path
        )
        model = model_loader.get_model()
        device = model_loader.device

        # 2️⃣ Xử lý audio
        features = preprocess_audio(
            model_name=request.model_name,
            file_path=request.file_path
        )

        # 3️⃣ Chuẩn bị lengths (nếu là WavLM)
        lengths = torch.tensor([1.0], device=device) if request.model_name.lower() == "wavlm" else None

        # 4️⃣ Trích xuất embedding
        embs = extract_embedding(
            model=model,
            features=features,
            lengths=lengths,
            device=device,
            mean_embedding=False
        )

        # 5️⃣ Chuyển tensor → list
        embs_serializable = [
            e.squeeze().tolist() if isinstance(e, torch.Tensor) else e
            for e in embs
        ]

        return {"embeddings": embs_serializable}

    except Exception as e:
        return {"error": str(e), "embeddings": []}
