from fastapi import APIRouter, UploadFile
from core.model_loader import ModelLoader
from utils.audio_preprocess import preprocess_audio
from utils.feature_extraction import extract_embedding

router = APIRouter()

@router.post("/register_voice/")
async def register_voice(model_name: str, ckpt_path: str, file_path: str):
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
    model, device = ModelLoader(model_name=model_name, num_classes=100, ckpt_path=ckpt_path)
    
    # preprocess audio
    features = preprocess_audio(model_name=model_name, file_path=file_path)

    # extract embeddings
    embs = extract_embedding(model=model, features=features, device=device, mean_embedding=False)

    return {"embeddings": embs}