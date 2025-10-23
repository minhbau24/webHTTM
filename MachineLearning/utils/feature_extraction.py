import torch

def extract_embedding(model, features, lengths=None, device="cpu", mean_embedding=False):
    """
    Trích xuất embedding từ features (đã preprocess).
    Args:
        model: model đã có method get_embedding
        features: list các tensor (từ preprocess_audio)
        lengths: list/tensor chiều dài (chỉ dùng cho WavLM)
        mean_embedding: nếu True thì trả về embedding trung bình
    Returns:
        embs: list các tensor embedding hoặc 1 tensor (nếu mean_embedding=True)
    """
    model.eval()
    embs = []
    for feature in features:
        feat = feature.unsqueeze(0).to(device)  # [1, T, F]
        if lengths is not None:
            length = torch.tensor([feat.size(1)], dtype=torch.long).to(device) \
                         if isinstance(lengths, (list, torch.Tensor)) else lengths.to(device)
            emb = model.get_embedding(feat, length)
        else:
            emb = model.get_embedding(feat)
        embs.append(emb.cpu())

    if mean_embedding:
        emb = torch.stack(embs, dim=0).mean(dim=0)
        return emb
    
    return embs