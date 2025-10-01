from fastapi import APIRouter, UploadFile
from core.model_loader import ModelLoader
from utils.audio_preprocess import preprocess_audio
from utils.feature_extraction import extract_embedding
import torch

router = APIRouter()

@router.post("/register_voice/")
async def register_voice(model_name: str, ckpt_path: str, file: UploadFile):
    # load model
    if model_name.lower() not in ["cnn_rnn", "ecapa", "wavlm"]:
        return {"error": "Model not supported. Choose from ['cnn_rnn', 'ecapa', 'wavlm']"}
    try:
        model = ModelLoader(model_name=model_name, num_classes=300, ckpt_path=ckpt_path)
    except Exception as e:
        return {"error": f"Error loading model: {e}"}

    features = preprocess_audio(model_name=model_name, file_path=file.file)

    embs = [extract_embedding(model, f.unsqueeze(0), lengths=torch.tensor([len(f)])) for f in features]

    return {"embeddings": embs}