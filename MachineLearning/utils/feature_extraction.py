import torch

def extract_embedding(model, features, lengths=None, device="cpu", mean_embedding=False, return_logits=False):
    """
    Trích xuất embedding từ features (đã preprocess).
    Args:
        model: model đã có method get_embedding hoặc forward
        features: list các tensor (từ preprocess_audio)
        lengths: list/tensor chiều dài (chỉ dùng cho WavLM)
        mean_embedding: nếu True thì trả về embedding trung bình
        return_logits: nếu True thì trả về (logits, embeddings) thay vì chỉ embeddings
    Returns:
        nếu return_logits=False: embs (list hoặc tensor)
        nếu return_logits=True: (logits_list, embs_list) hoặc (logits_tensor, emb_tensor)
    """
    model.eval()
    embs = []
    logits_list = [] if return_logits else None
    
    for feature in features:
        feat = feature.unsqueeze(0).to(device)  # [1, T, F]
        
        # Kiểm tra model có method get_embedding hay không
        if hasattr(model, 'get_embedding'):
            # Model có get_embedding (ResNet18_GRU_AAM, ECAPA_Finetune_AAM, WavLM_Finetune)
            if lengths is not None:
                length = torch.tensor([feat.size(1)], dtype=torch.long).to(device) \
                             if isinstance(lengths, (list, torch.Tensor)) else lengths.to(device)
                emb = model.get_embedding(feat, length)
            else:
                emb = model.get_embedding(feat)
            embs.append(emb.cpu())
            
            # Nếu cần logits, gọi forward để lấy
            if return_logits:
                with torch.no_grad():
                    if lengths is not None:
                        length = torch.tensor([feat.size(1)], dtype=torch.long).to(device) \
                                     if isinstance(lengths, (list, torch.Tensor)) else lengths.to(device)
                        logits, _ = model(feat, length)
                    else:
                        logits, _ = model(feat)
                    logits_list.append(logits.cpu())
        else:
            # Model không có get_embedding, gọi forward trực tiếp
            # (VD: WavLM_Finetune_Classification)
            with torch.no_grad():
                if lengths is not None:
                    length = torch.tensor([feat.size(1)], dtype=torch.long).to(device) \
                                 if isinstance(lengths, (list, torch.Tensor)) else lengths.to(device)
                    out, emb = model(feat, length)
                else:
                    out, emb = model(feat)
                
                embs.append(emb.cpu())
                if return_logits:
                    logits_list.append(out.cpu())

    if mean_embedding:
        emb = torch.stack(embs, dim=0).mean(dim=0)
        if return_logits:
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
            return logits, emb
        return emb
    
    if return_logits:
        return logits_list, embs
    return embs