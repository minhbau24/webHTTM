import torch
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from torch.utils.data import DataLoader
from typing import Optional
import os
import traceback

from core.model_loader import ModelLoader
from core.training import Trainer
from datasets.speaker_dataset import SpeakerDataset
from datasets.collate_fns import collate_spectrogram, collate_waveform

router = APIRouter()


async def train_with_websocket(
    websocket: WebSocket,
    model_name: str,
    model_type: str,
    train_data: list,
    val_data: list,
    num_classes: int,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 7,
    save_dir: str = "./checkpoints",
    ckpt_name: str = "best_model.pt",
    ckpt_path: Optional[str] = None,
    num_workers: int = 0
):
    """
    Helper function: Training với WebSocket streaming.
    Nhận parameters đã parse sẵn từ endpoint.
    """
    try:
        # Initial status
        await websocket.send_json({
            "type": "status",
            "message": "Initializing training...",
            "stage": "init"
        })
        await asyncio.sleep(0.1)  # Delay để client nhận riêng message

        # --- Validate data ---
        await websocket.send_json({
            "type": "status",
            "message": "Validating datasets...",
            "stage": "loading_data"
        })
        await asyncio.sleep(0.1)

        if not train_data or not val_data:
            raise ValueError("train_data and val_data cannot be empty!")

        # Data already contains "label" field, pass directly to SpeakerDataset
        train_records = [{**x, "label": int(x["label"])} for x in train_data]
        val_records = [{**x, "label": int(x["label"])} for x in val_data]


        await websocket.send_json({
            "type": "info",
            "message": f"Loaded datasets: Train={len(train_records)} samples, Val={len(val_records)} samples"
        })
        await asyncio.sleep(0.1)

        train_dataset = SpeakerDataset(train_records, model_name=model_name, augment=True)
        val_dataset = SpeakerDataset(val_records, model_name=model_name, augment=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_waveform if model_name == "wavlm" else collate_spectrogram,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_waveform if model_name == "wavlm" else collate_spectrogram,
            num_workers=num_workers
        )

        await websocket.send_json({
            "type": "info",
            "message": f"Dataloaders ready - {len(train_loader)} train batches, {len(val_loader)} val batches"
        })
        await asyncio.sleep(0.1)

        # --- Load model ---
        await websocket.send_json({
            "type": "status",
            "message": "Loading model...",
            "stage": "loading_model"
        })
        await asyncio.sleep(0.1)

        loader = ModelLoader(
            model_name=model_name,
            num_classes=num_classes,
            ckpt_path=ckpt_path,
            model_type=model_type
        )
        model = loader.get_model()
        device = loader.device

        await websocket.send_json({
            "type": "info",
            "message": f"Model loaded on device: {device}"
        })
        await asyncio.sleep(0.1)

        # --- Setup optimizer, loss, scheduler ---
        if model_type == 'embedding':
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        os.makedirs(save_dir, exist_ok=True)

        # --- Async callback để stream tiến trình ---
        async def on_epoch_end(epoch, train_acc, train_loss, val_acc, val_loss):
            try:
                await websocket.send_json({
                    "type": "epoch",
                    "epoch": epoch,
                    "total_epochs": num_epochs,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc)
                })
                await asyncio.sleep(0.1)  # Delay sau mỗi epoch update
            except WebSocketDisconnect:
                raise  # propagate để dừng training nếu client ngắt

        # --- Training ---
        await websocket.send_json({
            "type": "status",
            "message": f"Starting training for {num_epochs} epochs...",
            "stage": "training"
        })

        trainer = Trainer(
            model=model,
            model_name=model_name,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            device=device,
            model_type=model_type,
            save_dict=save_dir,
            on_epoch_end=on_epoch_end
        )

        await trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            patience=patience,
            ckpt_name=ckpt_name
        )

        # --- Load checkpoint info ---
        checkpoint_path = os.path.join(save_dir, ckpt_name)
        final_epoch, best_val_loss = num_epochs, None
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                final_epoch = checkpoint.get("epoch", num_epochs)
                best_val_loss = checkpoint.get("val_loss", None)
                
                # Load lại model tốt nhất trước khi test
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)
            except Exception:
                pass

        # --- Evaluate on validation set ---
        await websocket.send_json({
            "type": "status",
            "message": "Evaluating final model...",
            "stage": "testing"
        })
        await asyncio.sleep(0.1)

        test_metrics = trainer.evaluate(val_loader, test=True)

        # --- Send test results ---
        await websocket.send_json({
            "type": "completed",
            "message": "Training completed successfully!",
            "best_val_loss": float(best_val_loss) if best_val_loss else None,
            "final_epoch": final_epoch,
            "checkpoint_path": checkpoint_path,
            "test_results": {
                "accuracy": float(test_metrics["accuracy"]),
                "precision": float(test_metrics["precision"]),
                "recall": float(test_metrics["recall"]),
                "f1": float(test_metrics["f1"]),
                "loss": float(test_metrics["loss"])
            }
        })
        await asyncio.sleep(0.1)  # Delay cuối cùng trước khi đóng

    except WebSocketDisconnect:
        print("❌ Client disconnected during training.")
    except Exception as e:
        tb = traceback.format_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "traceback": tb,
                "error_type": type(e).__name__
            })
            await asyncio.sleep(0.1)  # Delay sau error message
        except Exception:
            pass


@router.websocket("/train/ws")
async def websocket_train_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint huấn luyện mô hình (real-time progress).
    
    Optional fields:
    - num_epochs: default 50
    - batch_size: default 32
    - learning_rate: default 0.001
    - patience: default 7
    - save_dir: default "./checkpoints"
    - ckpt_name: default "best_model.pt"
    - num_workers: default 0
    """
    await websocket.accept()

    try:
        data = await websocket.receive_json()

        required_fields = ["model_name", "model_type", "train_data", "val_data"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            await websocket.send_json({
                "type": "error",
                "message": f"Missing required fields: {', '.join(missing)}"
            })
            await asyncio.sleep(0.1)
            return

        # --- Xử lý data từ Java format ---
        def normalize_data(records):
            """
            Convert Java format to Python format:
            - user_id (string) → label (int)
            - Remove escaped quotes from file_path
            """
            normalized = []
            for item in records:
                file_path = item.get("file_path", "")
                # Remove escaped quotes: "\"path\"" → "path"
                file_path = file_path.strip('"').strip("'")
                
                # Get user_id, convert to int for label
                user_id = item.get("user_id") or item.get("label")
                if isinstance(user_id, str):
                    user_id = int(user_id)
                
                normalized.append({
                    "file_path": file_path,
                    "label": user_id
                })
            return normalized

        train_data = normalize_data(data["train_data"])
        val_data = normalize_data(data["val_data"])

        # --- Auto-calculate num_classes if not provided or is 0 ---
        num_classes = data.get("num_classes", 0)
        if num_classes == 0:
            # Tính từ unique labels trong train_data và val_data
            all_labels = set()
            for item in train_data + val_data:
                all_labels.add(item["label"])
            num_classes = len(all_labels)
            
            await websocket.send_json({
                "type": "info",
                "message": f"Auto-detected num_classes={num_classes} from data"
            })

        # Config với giá trị mặc định
        config = {
            "model_name": data["model_name"],
            "model_type": data["model_type"],
            "train_data": train_data,
            "val_data": val_data,
            "num_classes": num_classes,
            "num_epochs": data.get("num_epochs", 50),
            "batch_size": data.get("batch_size", 32),
            "learning_rate": data.get("learning_rate", 0.001),
            "patience": data.get("patience", 7),
            "save_dir": data.get("save_dir", "./checkpoints"),
            "ckpt_name": data.get("ckpt_name", "best_model.pt"),
            "ckpt_path": data.get("ckpt_path"),
            "num_workers": data.get("num_workers", 0)
        }

        await train_with_websocket(websocket, **config)

    except WebSocketDisconnect:
        print("❌ Client disconnected.")
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Unexpected error: {str(e)}"
            })
            await asyncio.sleep(0.1)
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
