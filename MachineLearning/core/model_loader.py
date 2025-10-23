import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from speechbrain.inference import SpeakerRecognition
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
    
    def get_embedding(self, feats):
        """Chỉ lấy embedding, không qua classifier
        feats: [B, T, F] mel-spectrogram
        """
        self.eval()
        with torch.no_grad():
            x = feats.unsqueeze(1)  # [B,1,T,F]
            x = self.feature_extractor(x)  # [B,512,T',F']

            # gộp tần số (frequency) → chuỗi thời gian
            x = x.mean(dim=-1)  # [B,512,T']
            x = x.transpose(1, 2)  # [B,T',512]

            out, _ = self.gru(x)  # [B,T',2*H]
            mean = out.mean(dim=1)  # [B,2H]

            emb = self.embedding(mean)  # [B,emb_dim]
            return emb
    
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
    
class WavLM_Finetune(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False, margin=0.2, scale=30):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.embedding_dim = self.backbone.config.hidden_size
        self.aam = AAMSoftmax(self.embedding_dim, num_classes, margin=margin, scale=scale)

    def forward(self, x, lengths, labels=None):
        # x: [B, T]
        attn_mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]

        outputs = self.backbone(x, attention_mask=attn_mask)
        hidden = outputs.last_hidden_state  # [B, T', H]

        # mean pooling (nếu muốn chuẩn hơn có thể mask theo attn_mask)
        emb = hidden.mean(dim=1)

        logits = self.aam(emb, labels)
        return logits, emb
    
    def get_embedding(self, x, lengths):
        """Chỉ lấy embedding, không qua classifier"""
        with torch.no_grad():
            attn_mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]

            outputs = self.backbone(x, attention_mask=attn_mask)
            hidden = outputs.last_hidden_state  # [B, T', H]

            emb = hidden.mean(dim=1)
            return emb

class ECAPA_Deepfake(nn.Module):
    def __init__(self, backbone, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Linear giảm dần
        self.fc_layers = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Classifier cuối cho binary
        self.classifier = nn.Linear(64, 1)  # 1 output, dùng BCEWithLogitsLoss

    def forward(self, x):
        emb = self.backbone(x)      # [B, 1, 192]
        emb = emb.squeeze(1)        # [B, 192]

        reduced = self.fc_layers(emb)  # [B, 64]
        out = self.classifier(reduced) # [B, 1], chưa sigmoid

        return out, reduced

class WavLM_Deepfake(nn.Module):
    def __init__(self, backbone, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        self.embedding_dim = self.backbone.config.hidden_size
        self.fc_liner = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x, lengths=None):
        # x: [B, T] waveform float32 [-1,1]
        # lengths: [B] số frame gốc (trước pad)

        attn_mask = None
        if lengths is not None:
            # tạo mask đúng: 1 cho frame hợp lệ, 0 cho pad
            attn_mask = (torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]).bool()

        outputs = self.backbone(x, attention_mask=attn_mask)
        hidden = outputs.last_hidden_state  # [B, T', H]

        if attn_mask is not None:
            mask = attn_mask[:, :hidden.size(1)].unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # mean pooling có mask
        else:
            emb = hidden.mean(dim=1)

        reduced = self.fc_liner(emb) # [B, 128]
        out = self.classifier(reduced)
        return out, emb

class CNN_RNN_Deepfake(nn.Module):
    def __init__(self, embedding_dim=256, rnn_hidden=256, rnn_layers=2, n_mels=80, dropout=0.3):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),   # [B,32,T/2,F/2]
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((4,2)),   # mạnh hơn: [B,64,T/8,F/4]
        )
        
        # Projection (giảm số chiều feature trước khi đưa vào LSTM)
        cnn_out_dim = 64 * (n_mels // 4)   # sau pooling (F/4)
        self.proj = nn.Linear(cnn_out_dim, 256)
        
        # RNN
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        
        # Projection → embedding
        self.embedding = nn.Sequential(
            nn.Linear(rnn_hidden * 2 * 2, embedding_dim),  # *2 vì bi-LSTM, *2 vì mean+std
            nn.ReLU(),
            nn.Dropout(dropout)
        )  
        
        self.fc_linear = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, feats):
        x = feats.unsqueeze(1)           # [B, 1, T, F]
        x = self.cnn(x)                  # [B, C, T', F']
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)

        # Projection trước khi vào LSTM
        x = self.proj(x)                 # [B, T, 256]

        out, _ = self.rnn(x)             # [B, T, 2*H]
        mean = out.mean(dim=1)
        std = out.std(dim=1)
        stats = torch.cat([mean, std], dim=1)

        emb = self.embedding(stats)      # [B, emb_dim]

        reduced = self.fc_linear(emb)
        out = self.classifier(reduced)
        return out, emb

# =====================================
# 2. class ModelLoader
# =====================================

class ModelLoader:
    def __init__(self, model_name: str, num_classes: int, ckpt_path: str = None, model_type: str = None, device: str = None):
        self.model_name = model_name.lower()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type if model_type else 'embedding'
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
        if self.model_type == 'embedding':
            if self.model_name == "cnn_rnn":
                return ResNet18_GRU_AAM(num_classes=num_classes)
            elif self.model_name == "ecapa":
                pretrained = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    # save_dir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                backbone = pretrained.mods.embedding_model
                return ECAPA_Finetune_AAM(backbone, num_classes=num_classes, freeze_backbone=False)
            elif self.model_name == "wavlm":
                return WavLM_Finetune(num_classes=num_classes)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")
        elif self.model_type == 'deepfake':
            if self.model_name == "cnn_rnn":
                return CNN_RNN_Deepfake()
            elif self.model_name == "ecapa":
                pretrained = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    # save_dir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                backbone = pretrained.mods.embedding_model
                return ECAPA_Deepfake(backbone, freeze_backbone=False)
            elif self.model_name == "wavlm":
                backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
                return WavLM_Deepfake(backbone, freeze_backbone=False)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    def get_model(self):
        return self.model
