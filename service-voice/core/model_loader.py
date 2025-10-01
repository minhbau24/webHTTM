import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from speechbrain.pretrained import SpeakerRecognition
from transformers import WavLMModel
import os


# =====================================
# 1. Định nghĩa các cấu hình model
# =====================================
class AAMSoftmax(nn.Module):
    def __init__(self, in_dim, num_classes, margin=0.2, scale=30):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings, labels=None):
        # normalize weight & feature
        W = F.normalize(self.weight, dim=1)
        x = F.normalize(embeddings, dim=1)
        cosine = F.linear(x, W)  # [B, num_classes]

        if labels is None:
            # Inference mode (không margin)
            return cosine * self.scale

        # one-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # áp dụng margin vào class đúng
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.margin)
        logits = cosine * (1 - one_hot) + target_logits * one_hot

        return logits * self.scale
    

class ResNet18_GRU_AAM(nn.Module):
    def __init__(self, num_classes, embedding_dim=256, rnn_hidden=256, rnn_layers=2, dropout=0.3):
        super().__init__()

        # ResNet18 backbone (dùng cho input 1 kênh: spectrogram/mel)
        base_model = models.resnet18(pretrained=False)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # bỏ avgpool & fc

        feat_dim = 512  # ResNet18 cuối cùng ra [B, 512, T', F']

        # GRU
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0
        )

        # projection to embedding
        self.embedding = nn.Sequential(
            nn.Linear(rnn_hidden * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # AAM softmax
        self.aam = AAMSoftmax(embedding_dim, num_classes)

    def forward(self, feats, labels=None):
        # feats: [B, T, F] mel-spectrogram
        x = feats.unsqueeze(1)  # [B,1,T,F]
        x = self.feature_extractor(x)  # [B,512,T',F']

        # gộp tần số (frequency) → chuỗi thời gian
        x = x.mean(dim=-1)  # [B,512,T']
        x = x.transpose(1, 2)  # [B,T',512]

        out, _ = self.gru(x)  # [B,T',2*H]
        mean = out.mean(dim=1)  # [B,2H]

        emb = self.embedding(mean)  # [B,emb_dim]

        logits = self.aam(emb, labels) if labels is not None else self.aam(emb)

        return logits, emb
    

class ECAPA_Finetune_AAM(nn.Module):
    def __init__(self, backbone, num_classes, embedding_dim=192, freeze_backbone=False, margin=0.2, scale=30):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.aamsoftmax = AAMSoftmax(embedding_dim, num_classes, margin, scale)

    def forward(self, x, labels=None):
        emb = self.backbone(x)       # [B, 1, 192]
        emb = emb.squeeze(1)         # [B, 192]
        logits = self.aamsoftmax(emb, labels)
        return logits, emb
    
    def get_embedding(self, x):
        """Chỉ lấy embedding, không qua classifier"""
        with torch.no_grad():
            emb = self.backbone(x)
            return emb.squeeze(1)

# =====================================
# 2. class ModelLoader
# =====================================

class ModelLoader:
    def __init__(self, model_name: str, num_classes: int, ckpt_path: str = None, device: str = None):
        self.model_name = model_name.lower()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(num_classes).to(self.device)

        if ckpt_path and os.path.isfile(ckpt_path):
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.eval()

    def _build_model(self, num_classes: int):
        if self.model_name == "cnn_rnn":
            return ResNet18_GRU_AAM(num_classes=num_classes)
        elif self.model_name == "ecapa":
            pretrained = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                save_dir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            backbone = pretrained.modules.embedding_model
            return ECAPA_Finetune_AAM(backbone, num_classes=num_classes, freeze_backbone=False)
        elif self.model_name == "wavlm":
            return ECAPA_Finetune_AAM(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
    def get_model(self):
        return self.model
