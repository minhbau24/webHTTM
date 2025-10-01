import torch

def extract_embedding(model, inputs, lengths=None, device="cpu"):
    model.eval()
    with torch.no_grad():
        if lengths is not None:
            _, emb = model(inputs.to(device), lengths.to(device))
        else:
            _, emb = model(inputs.to(device))
    return emb.cpu()