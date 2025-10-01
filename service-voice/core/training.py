import torch

def train_one_epoch(model, dataloader, optimizer, loss_fn, model_name, device="cpu"):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        if model_name == "CNN_RNN":
            Wavs, labels = batch
            Wavs, labels = Wavs.to(device), labels.to(device)
            logits, emb = model(Wavs)

        elif model_name == "ecapa":
            Wavs, labels, _ = batch
            Wavs, labels = Wavs.to(device), labels.to(device)
            logits, emb = model(Wavs)

        elif model_name == "wavlm":
            Wavs, labels, lengths = batch
            Wavs, labels, lengths = Wavs.to(device), labels.to(device), lengths.to(device)
            logits, emb = model(Wavs, lengths)
        
        else:
            raise ValueError("model_name must be 'ecapa' or 'wavlm' or 'cnn_rnn'")
        
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device, model_name):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if model_name == "cnn_rnn":
                wavs, labels = batch
                wavs, labels = wavs.to(device), labels.to(device)
                logits, emb = model(wavs)

            elif model_name == "ecapa":
                wavs, labels, _ = batch
                wavs, labels = wavs.to(device), labels.to(device)
                logits, emb = model(wavs)

            elif model_name == "wavlm":
                wavs, labels, lengths = batch
                wavs, labels, lengths = wavs.to(device), labels.to(device), lengths.to(device)
                logits, emb = model(wavs, lengths)

            # loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # acc
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc

