import torch
from fastapi import APIRouter, UploadFile
from core.model_loader import ModelLoader
from utils.audio_preprocess import preprocess_audio
from utils.feature_extraction import extract_embedding

router = APIRouter()

@router.post("/register_voice/")
async def register_voice(model_name: str, ckpt_path: str, file_path: str, num_classes: int = 100):
    """
    Đăng ký giọng nói mới bằng cách trích xuất embedding từ file audio.
    Args:
        model_name: tên model để xác định config
        ckpt_path: đường dẫn checkpoint của model
        file_path: đường dẫn file audio

    Returns:
        embs: list các tensor embedding (nếu audio dài thì có thể có nhiều hơn 1 embedding)
    """
    # load model
    model_loader  = ModelLoader(model_name=model_name, num_classes=num_classes, ckpt_path=ckpt_path)
    model = model_loader.get_model()
    device = model_loader.device

    # preprocess audio
    features = preprocess_audio(model_name=model_name, file_path=file_path)
    
    if model_name.lower() == "wavlm":
        lengths = torch.tensor([1.0], device=device)  # vì chỉ 1 file, normalized = 1.0
    else:
        lengths = None
    # extract embeddings
    embs = extract_embedding(model=model, features=features, lengths=lengths, device=device, mean_embedding=False)

    embs_serializable = []
    for e in embs:
        if isinstance(e, torch.Tensor):
            embs_serializable.append(e.squeeze().tolist())  # chuyển tensor thành list Python
        else:
            embs_serializable.append(e)

    return {"embeddings": embs_serializable}