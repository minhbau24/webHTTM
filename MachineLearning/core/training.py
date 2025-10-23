import torch
from sklearn.metrics import accuracy_score

class Trainer:
    def __init__(self, model, model_name, optimizer, loss_fn, scheduler, device, save_dict='./checkpoint', on_epoch_end=None):
        self.model = model
        self.model_name = model_name.lower()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.save_dict = save_dict
        self.on_epoch_end = on_epoch_end

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        y_true, y_pred = [], []
        for batch in dataloader:
            if self.model_name in ['cnn_rnn', 'ecapa']:
                feats, labels = batch
                feats, labels = feats.to(self.device), labels.to(self.device)
                logits, _ = self.model(feats)

            elif self.model_name == 'wavlm':
                signals, lengths, labels = batch
                signals, lengths, labels = signals.to(self.device), lengths.to(self.device), labels.to(self.device)
                logits, _ = self.model(signals, lengths)
            
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

            loss = self.loss_fn(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
        
        acc = accuracy_score(y_true, y_pred)
        avg_loss = total_loss / len(dataloader)
        return acc, avg_loss
    
    def evaluate(self, dataloader):
        self.model.eval()
        y_true, y_pred = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                if self.model_name in ['cnn_rnn', 'ecapa']:
                    feats, labels = batch
                    feats, labels = feats.to(self.device), labels.to(self.device)
                    logits, _ = self.model(feats)
                elif self.model_name == 'wavlm':
                    signals, lengths, labels = batch
                    signals, lengths, labels = signals.to(self.device), lengths.to(self.device), labels.to(self.device)
                    logits, _ = self.model(signals, lengths)
                else:
                    raise ValueError(f"Unsupported model name: {self.model_name}")
            
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        acc = accuracy_score(y_true, y_pred)
        return acc, total_loss / len(dataloader)
    
    def fit(self, train_loader, val_loader, num_epochs=50, patience=7, ckpt_name="best_model.pth"):
        best_val = float('inf')
        counter = 0
        for epoch in range(1, num_epochs+1):
            train_acc, train_loss = self.train_one_epoch(train_loader)
            val_acc, val_loss = self.evaluate(val_loader)

            if self.scheduler is not None:
                try:
                    self.scheduler.step(val_loss)
                except:
                    pass
            if val_loss < best_val:
                best_val = val_loss
                counter = 0
                path = f"{self.save_dict}/{ckpt_name}"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': getattr(self.optimizer, 'state_dict', lambda: None)(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, path)
            else:
                counter += 1
                if counter >= patience:
                    break

            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, train_acc, train_loss, val_acc, val_loss)
