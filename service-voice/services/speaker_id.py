import torch
import torch.nn.functional as F
from core.model_loader import ModelLoader
from utils.audio_preprocess import preprocess_audio
from utils.feature_extraction import extract_embedding

def get_n_similar_speakers(input_audio_path, all_speaker_embeddings, model_name, top_k=5):
    """
    So sánh input audio với tất cả speaker embeddings để tìm top-k speakers giống nhất.
    Args:
        input_audio_path: đường dẫn file audio đầu vào
        all_speaker_embeddings: list các tensor embedding của tất cả speakers trong database
        model_name: tên model để xác định config
        top_k: số lượng speakers giống nhất cần trả về
    Returns:
        top_k_indices: list các chỉ số (index) của top-k speakers giống nhất
        top_k_scores: list các điểm số (similarity) tương ứng
    """
    # load model
    model, device = ModelLoader(model_name=model_name, num_classes=100)

    # preprocess input audio
    features = preprocess_audio(model_name=model_name, file_path=input_audio_path)

    # extract embedding
    input_emb = extract_embedding(model=model, features=features, device=device, mean_embedding=True)

    # normalize embedding
    input_emb = F.normalize(input_emb, p=2, dim=0)  # [1, emb_dim]
    if isinstance(all_speaker_embeddings, list):
        all_speaker_embeddings = torch.stack(all_speaker_embeddings, dim=0)  # [N, emb_dim]
    all_speaker_embeddings = F.normalize(all_speaker_embeddings, p=2, dim=1)  # [N, emb_dim]

    # cosine similarity
    similarities = torch.matmul(all_speaker_embeddings, input_emb) # [N]

    # top-k
    top_k_scores, top_k_indices = torch.topk(similarities, k=top_k)

    return top_k_indices.tolist(), top_k_scores.tolist()