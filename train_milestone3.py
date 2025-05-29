import csv
import os
import pickle
import random
import re

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pickle
from tqdm import tqdm
import cv2

import csv
import random


class DrivingDataset(Dataset):
    def __init__(self, file_list, test=False):
        self.samples = file_list
        self.test = test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        camera = torch.FloatTensor(data['camera']).permute(2, 0, 1) / 255.0
        history = torch.FloatTensor(data['sdc_history_feature'])

        semantic = torch.LongTensor(data['semantic_label'])
        if semantic.shape[-2:] != (200, 300):
            semantic = TF.resize(semantic.unsqueeze(0).float(), size=(200, 300), interpolation=TF.InterpolationMode.NEAREST).long().squeeze(0)

        if not self.test:
            future = torch.FloatTensor(data['sdc_future_feature'])
            return {
                'camera': camera,
                'history': history,
                'future': future,
                'semantic': semantic
            }
        else:
            return {
                'camera': camera,
                'history': history,
                'semantic': semantic
            }


def compute_ade_fde(pred_trajectories, gt_trajectory, include_heading=False, confidences=None):
    """
    Compute Average Displacement Error and Final Displacement Error

    Args:
        pred_trajectories: Tensor of shape [B, M, T, 2] where M is number of modes
        gt_trajectory: Tensor of shape [B, T, 2+] (the + indicates there might be more features)

    Returns:
        ade: Average Displacement Error across all timesteps
        fde: Final Displacement Error (last timestep only)
    """
    # Extract only x, y coordinates from ground truth if needed
    if include_heading:
        gt_xy = gt_trajectory[..., :3]
    else:
        gt_xy = gt_trajectory[..., :2]

    # Calculate per-mode errors
    error_per_mode = torch.norm(pred_trajectories - gt_xy.unsqueeze(1), dim=-1)  # [B, M, T]

    # If confidences are provided, use them to select best mode
    if confidences is not None:
        best_mode_idx = confidences.argmax(dim=1)  # [B] - Use highest confidence
    else:
        # Fall back to minimum ADE if no confidences provided
        mode_ade = error_per_mode.mean(dim=2)  # [B, M]
        best_mode_idx = mode_ade.argmin(dim=1)  # [B]

    # Get errors for best mode per batch element
    batch_indices = torch.arange(pred_trajectories.size(0), device=pred_trajectories.device)
    best_mode_error = error_per_mode[batch_indices, best_mode_idx]  # [B, T]

    # Compute metrics
    ade = best_mode_error.mean().item()
    fde = best_mode_error[:, -1].mean().item()

    return ade, fde

class EnhancedDrivingPlanner(nn.Module):
    def __init__(self, num_modes=4, lr=1e-4, future_steps=60, include_heading=False,
                 semantic_output_size=(200, 300), num_semantic_classes=15):
        super().__init__()

        self.num_modes = num_modes
        self.future_steps = future_steps
        self.include_heading = include_heading
        self.semantic_output_size = semantic_output_size
        self.num_semantic_classes = num_semantic_classes

        # Visual encoder: ResNet18 pretrained
        resnet = models.resnet18(pretrained=True)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc layers
        self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pool to (batch, channels, 1, 1)
        self.visual_fc = nn.Linear(512, 256)  # ResNet18 last conv outputs 512 channels

        # History encoder
        self.history_encoder = nn.Sequential(
            nn.Linear(21 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # Semantic Decoder - Fixed to work with feature maps
        # We need to work with the feature maps before pooling for semantic segmentation
        self.semantic_decoder = nn.Sequential(
            # Start from ResNet feature maps (B, 512, H/32, W/32)
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Upsample by 2x
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Upsample by 2x
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Upsample by 2x
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Final upsampling and classification
            nn.ConvTranspose2d(32, self.num_semantic_classes, kernel_size=4, stride=2, padding=1),
        )

        # Trajectory decoder
        if include_heading:
            self.trajectory_head = nn.Linear(512, num_modes * future_steps * 3)
        else:
            self.trajectory_head = nn.Linear(512, num_modes * future_steps * 2)  # (x, y) only, no heading
        self.confidence_head = nn.Linear(512, num_modes)  # One confidence score per mode

    def forward(self, camera, history):
        batch_size = camera.size(0)

        # Encode image - keep feature maps for semantic segmentation
        visual_feature_maps = self.visual_encoder(camera)  # (B, 512, H', W')

        # For trajectory prediction, pool the features
        visual_features_pooled = self.visual_pool(visual_feature_maps).view(batch_size, -1)  # (B, 512)
        visual_features = self.visual_fc(visual_features_pooled)  # (B, 256)

        # Encode history
        history_flat = history.reshape(batch_size, -1)
        history_features = self.history_encoder(history_flat)  # (B, 128)

        # Fuse for trajectory prediction
        fused = self.fusion(torch.cat([visual_features, history_features], dim=1))  # (B, 512)

        # Predict Trajectory
        traj = self.trajectory_head(fused)  # (B, num_modes * future_steps * 2)
        if self.include_heading:
            traj = traj.view(batch_size, self.num_modes, self.future_steps, 3)
        else:
            traj = traj.view(batch_size, self.num_modes, self.future_steps, 2)

        conf = self.confidence_head(fused)  # (B, num_modes)
        conf = torch.softmax(conf, dim=-1)  # Confidence scores across modes

        # Predict semantic segmentation using feature maps
        semantic_pred_logits = self.semantic_decoder(visual_feature_maps)  # (B, num_classes, H, W)

        # Resize to target output size if needed
        if semantic_pred_logits.shape[-2:] != self.semantic_output_size:
            semantic_pred_logits = F.interpolate(
                semantic_pred_logits,
                size=self.semantic_output_size,
                mode="bilinear",
                align_corners=False
            )

        return traj, conf, semantic_pred_logits

    def compute_loss(self, traj_pred, conf_pred, future, semantic_pred_logits, semantic_gt,
                     semantic_weight=0.5):
        # Trajectory loss
        if self.include_heading:
            future_expand = future[:, None, :, :3].expand_as(traj_pred)
        else:
            future_expand = future[:, None, :, :2].expand_as(traj_pred)

        # Compute loss for each mode
        loss_per_mode = nn.SmoothL1Loss(reduction='none')(traj_pred, future_expand).mean(dim=[2,3])  # (B, num_modes)
        # Confidence-weighted loss
        trajectory_loss = (loss_per_mode * conf_pred).sum(dim=1).mean()

        # Semantic loss
        semantic_loss = nn.CrossEntropyLoss()(semantic_pred_logits, semantic_gt)

        # Combined loss
        combined_loss = trajectory_loss + semantic_weight * semantic_loss

        return combined_loss, trajectory_loss, semantic_loss


class LightningSimplifiedDrivingPlanner(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4, epochs=50, num_modes=4,
                 scheduler_factor=0.95, scheduler_patience=5, include_heading=False,
                 semantic_output_size=(200, 300), num_semantic_classes=15, semantic_weight=0.5):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.semantic_weight = semantic_weight

        self.planner = EnhancedDrivingPlanner(
            num_modes=num_modes,
            include_heading=include_heading,
            semantic_output_size=semantic_output_size,
            num_semantic_classes=num_semantic_classes
        )

        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.include_heading = include_heading
        self.save_hyperparameters()

        print("\n====================================\nModel initialized with parameters:")
        print(f"Learning Rate: {self.lr}")
        print(f"Weight Decay: {self.weight_decay}")
        print(f"Number of Epochs: {self.epochs}")
        print(f"Number of Modes: {num_modes}")
        print(f"Scheduler Factor: {self.scheduler_factor}")
        print(f"Scheduler Patience: {self.scheduler_patience}")
        print(f"Include Heading: {include_heading}")
        print(f"Semantic Classes: {num_semantic_classes}")
        print(f"Semantic Weight: {semantic_weight}")

    def forward(self, camera, history):
        return self.planner(camera, history)

    def training_step(self, batch, batch_idx):
        camera = batch['camera']           # [B,3,H,W]
        history = batch['history']         # [B, T, 3]
        future = batch['future']           # [B, T, 3]
        semantic_labels = batch['semantic'] # [B, H, W]

        # Forward pass
        pred_trajs, confidences, semantic_pred_logits = self(camera, history)

        # Compute loss
        combined_loss, traj_loss, semantic_loss = self.planner.compute_loss(
            pred_trajs, confidences, future, semantic_pred_logits, semantic_labels,
            semantic_weight=self.semantic_weight
        )

        # Log losses
        self.log('train_loss', combined_loss, prog_bar=True)
        self.log('train_traj_loss', traj_loss, prog_bar=False)
        self.log('train_semantic_loss', semantic_loss, prog_bar=False)

        return combined_loss

    def validation_step(self, batch, batch_idx):
        camera = batch['camera']
        history = batch['history']
        future = batch['future']
        semantic_labels = batch['semantic']

        # Forward pass
        pred_trajs, confidences, semantic_pred_logits = self(camera, history)

        # Compute losses
        combined_loss, traj_loss, semantic_loss = self.planner.compute_loss(
            pred_trajs, confidences, future, semantic_pred_logits, semantic_labels,
            semantic_weight=self.semantic_weight
        )

        # Compute ADE and FDE
        ade, fde = compute_ade_fde(pred_trajs, future, self.include_heading, confidences)

        # Compute semantic accuracy
        semantic_pred = torch.argmax(semantic_pred_logits, dim=1)
        semantic_acc = (semantic_pred == semantic_labels).float().mean()

        # Log validation metrics
        self.log('val_loss', combined_loss, prog_bar=True, sync_dist=True)
        self.log('val_traj_loss', traj_loss, prog_bar=False, sync_dist=True)
        self.log('val_semantic_loss', semantic_loss, prog_bar=False, sync_dist=True)
        self.log('val_ade', ade, prog_bar=True, sync_dist=True)
        self.log('val_fde', fde, prog_bar=True, sync_dist=True)
        self.log('val_semantic_acc', semantic_acc, prog_bar=True, sync_dist=True)

        return {
            'val_loss': combined_loss,
            'val_ade': ade,
            'val_fde': fde,
            'val_semantic_acc': semantic_acc
        }

    def test_step(self, batch, batch_idx):
        # Don't do anything because we don't have the ground truth
        return {}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_ade',
                'interval': 'epoch',
                'frequency': 1,
                'reduce_on_plateau': True
            }
        }

def adjust_brightness(image, bright_thresh=110, dark_thresh=90):
    """
    Adjust brightness adaptively and safely.
    - Input: image in uint8 format
    - Output: image in uint8 format
    """
    if image.dtype != np.uint8:
        raise ValueError("Expected input image in uint8 format")

    # Compute mean brightness from grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()
    # Choose factor based on brightness level
    if mean_brightness > bright_thresh:
        factor = np.random.uniform(0.5, 0.8)  # darken
    elif mean_brightness < dark_thresh:
        factor = np.random.uniform(1.2, 1.5)  # brighten
    else:
        factor = np.random.uniform(0.8, 1.2)  # mild change

    # Convert to float for safe scaling
    image = image.astype(np.float32)
    image = image * factor
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def apply_blur(image):
    k = random.choice([1, 3, 5])
    return cv2.GaussianBlur(image, (k, k), 0)

def add_noise(image, noise_level=15):
    noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image

def apply_gamma(image, gamma_range=(0.5, 1.5)):
    gamma = np.random.uniform(*gamma_range)
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- Helper to apply and save ---

def apply_augmentation_to_subset(file_list, save_dir, augmentation_fn, suffix, ratio):
    os.makedirs(save_dir, exist_ok=True)
    new_files = []

    selected_files = random.sample(file_list, int(len(file_list) * ratio))

    for file_path in tqdm(selected_files, desc=f"Applying {suffix}"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        image = data['camera']
        aug_image = augmentation_fn(image)
        data['camera'] = aug_image
        base = os.path.basename(file_path)
        new_filename = f"{suffix}_{base}"
        new_path = os.path.join(save_dir, new_filename)

        with open(new_path, 'wb') as f:
            pickle.dump(data, f)

        new_files.append(new_path)

    return new_files

# --- Main driver ---

def incremental_augmentation(file_list, save_dir, ratios):
    os.makedirs(save_dir, exist_ok=True)
    all_files = list(file_list)  # Start with original + flipped

    stage1 = apply_augmentation_to_subset(
        all_files, save_dir, apply_gamma, 'gamma', ratios.get('gamma', 0.5)
    )
    all_files += stage1

    stage2 = apply_augmentation_to_subset(
        all_files, save_dir, add_noise, 'noise', ratios.get('noise', 0.5)
    )
    all_files += stage2

    stage3 = apply_augmentation_to_subset(
        all_files, save_dir, adjust_brightness, 'bright', ratios.get('brightness', 0.5)
    )
    all_files += stage3

    stage4 = apply_augmentation_to_subset(
        all_files, save_dir, apply_blur, 'blur', ratios.get('blur', 0.5)
    )
    all_files += stage4

    return all_files

from collections import defaultdict

def count_augmentation_variants(file_paths):
    """
    Given a list of file paths, returns a dictionary counting how many files
    belong to each augmentation suffix (e.g., 'flipped', 'gamma', 'bright', etc.)
    """

    counts = defaultdict(int)

    for path in file_paths:
        filename = os.path.basename(path)
        name_parts = filename.split('_')

        # Collect all known prefixes (excluding index and extension)
        suffixes = name_parts[:-1] if len(name_parts) > 1 else ['original']
        key = '+'.join(sorted(suffixes)) if suffixes else 'original'

        counts[key] += 1

    return dict(counts)

# Updated loading functions to handle the new architecture
def load_pretrained_encoders_multitask(model, checkpoint_path, load_visual=True, load_history=True,
                                      strict=False, skip_semantic=F):
    """
    Load pretrained encoders for the multi-task model, optionally skipping semantic decoder.

    Args:
        model: Your current multi-task model instance
        checkpoint_path: Path to the checkpoint file
        load_visual: Whether to load the visual encoder weights
        load_history: Whether to load the history encoder weights
        strict: Whether to strictly match all keys
        skip_semantic: Whether to skip loading semantic decoder (useful when loading from single-task model)
    """

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Get the current model's state dict
    model_state_dict = model.state_dict()

    # Create a new state dict with only the components we want to load
    filtered_state_dict = {}

    if load_visual:
        # Load visual encoder components (but not semantic decoder)
        visual_keys = [
            'planner.visual_encoder',
            'planner.visual_pool',
            'planner.visual_fc'
        ]

        for key in state_dict:
            for visual_key in visual_keys:
                if key.startswith(visual_key):
                    # Handle prefix differences
                    model_key = key
                    if key.startswith('planner.') and not any(mk.startswith('planner.') for mk in model_state_dict.keys()):
                        model_key = key.replace('planner.', '')
                    elif not key.startswith('planner.') and any(mk.startswith('planner.') for mk in model_state_dict.keys()):
                        model_key = 'planner.' + key

                    if model_key in model_state_dict:
                        filtered_state_dict[model_key] = state_dict[key]
                        #print(f"Loading visual encoder key: {key} -> {model_key}")

    if load_history:
        # Load history encoder components
        for key in state_dict:
            if 'history_encoder' in key:
                # Handle prefix differences
                model_key = key
                if key.startswith('planner.') and not any(mk.startswith('planner.') for mk in model_state_dict.keys()):
                    model_key = key.replace('planner.', '')
                elif not key.startswith('planner.') and any(mk.startswith('planner.') for mk in model_state_dict.keys()):
                    model_key = 'planner.' + key

                if model_key in model_state_dict:
                    filtered_state_dict[model_key] = state_dict[key]
                    #print(f"Loading history encoder key: {key} -> {model_key}")

    # Also load fusion and trajectory/confidence heads if they exist and match
    other_keys = ['planner.fusion', 'planner.trajectory_head', 'planner.confidence_head']
    for key in state_dict:
        for other_key in other_keys:
            if key.startswith(other_key):
                model_key = key
                if key.startswith('planner.') and not any(mk.startswith('planner.') for mk in model_state_dict.keys()):
                    model_key = key.replace('planner.', '')
                elif not key.startswith('planner.') and any(mk.startswith('planner.') for mk in model_state_dict.keys()):
                    model_key = 'planner.' + key

                if model_key in model_state_dict:
                    # Check if dimensions match
                    if state_dict[key].shape == model_state_dict[model_key].shape:
                        filtered_state_dict[model_key] = state_dict[key]
                        #print(f"Loading {other_key} key: {key} -> {model_key}")
                    else:
                        print(f"Skipping {key} due to shape mismatch: {state_dict[key].shape} vs {model_state_dict[model_key].shape}")

    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=strict)

    print(f"\nSuccessfully loaded {len(filtered_state_dict)} parameters")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    return model, missing_keys, unexpected_keys

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)



# Modified training function to use the pretrained model weights
def run_experiment_with_pretrained(
    train_loader,
    val_loader,
    pretrained_model_path=None,
    max_epochs=70,
    lr=1e-5,
    weight_decay=2.6551e-6,
    beta=1,
    dynamic_weighting=True,
    logger_name="milestone3",
    semantic_k=0.3,
    freeze_encoders=False,
):
    # Set random seed for reproducibility
    pl.seed_everything(13)
    logger_path = os.path.join("lightning_logs", "phase3")
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    # Initialize model with Lightning wrapper that loads pretrained weights
    model = LightningSimplifiedDrivingPlanner()
    pretrained_model_path= "checkpoints/phase1/driving_planner_version_68_epoch=107_val_ade=1.55.ckpt"

    model, missing, unexpected = load_pretrained_encoders_multitask(
        model,
        pretrained_model_path,
        load_visual=True,
        load_history=True,
        strict=False
    )

    # Initialize logger
    logger = TensorBoardLogger(save_dir=logger_path, name=logger_name)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    filename = f"version_{logger.version}"
    print(f"Model version: {logger.version}\n====================================\n ")

    checkpoint_path = os.path.join("checkpoints", "phase3")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_dir = os.path.join(checkpoint_path, logger_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename + "_{epoch:02d}_{val_traj_loss:.4f}_{val_ade:.4f}",
        monitor="val_ade",
        mode="min",
        save_top_k=2,
        save_last=False,
        verbose=False,
    )
    # Create trainer and fit model
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision=16,  # Use mixed precision
        log_every_n_steps=10,  # Logging freq
        gradient_clip_val=5.0,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer.checkpoint_callback.best_model_path


def flip_and_save(file_list, save_dir, flip_ratio=0.5):
    """Horizontal flip augmentation for both camera and semantic data with randomness"""
    os.makedirs(save_dir, exist_ok=True)
    new_files = []

    # Randomly select files to flip based on flip_ratio
    selected_files = random.sample(file_list, int(len(file_list) * flip_ratio))

    for file_path in tqdm(selected_files, desc="Horizontal flip augmentation"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Flip camera image horizontally
        camera = data['camera'][:, ::-1, :]  # shape HxWxC, flip width

        # Flip semantic labels horizontally
        semantic = data['semantic_label'][:, ::-1]  # shape HxW, flip width

        # Update data
        data['camera'] = camera
        data['semantic_label'] = semantic
        data['sdc_history_feature'][:, 1:] *= -1

        if 'sdc_future_feature' in data:
            data['sdc_future_feature'][:, 1:] *= -1

        # Save to new file
        base = os.path.basename(file_path)
        new_path = os.path.join(save_dir, f"flipped_{base}")
        with open(new_path, 'wb') as f:
            pickle.dump(data, f)

        new_files.append(new_path)

    return new_files

if __name__ == '__main__':
    seed = 13
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #train_files = [os.path.join("data", "phase2", "augmented_train", f) for f in os.listdir("data/phase2/augmented_train") if f.endswith('.pkl')]
    #train_files1= [os.path.join("data", "phase2", "augmented_val", f) for f in os.listdir("data/phase2/augmented_val") if f.endswith('.pkl')]
    train_files = [os.path.join("data", "phase2", "train", f) for f in os.listdir("data/phase2/train") if f.endswith('.pkl')]
    val_files_real = [os.path.join("data", "phase3", "val_real_semantics", f) for f in os.listdir("data/phase3/val_real_semantics") if f.endswith('.pkl')]
    val_files_synth = [os.path.join("data", "phase2", "val", f) for f in os.listdir("data/phase2/val") if f.endswith('.pkl')]

    if not os.path.exists("data/phase2/flipped_train"):
        flipped_train_files = flip_and_save(train_files, "data/phase2/flipped_train", flip_ratio=1)
    else:
        flipped_train_files = [os.path.join("data", "phase2" , "flipped_train", f) for f in os.listdir("data/phase2/flipped_train") if f.endswith('.pkl')]
    if not os.path.exists("data/phase2/flipped_val"):
        flipped_val_files_synth = flip_and_save(val_files_synth, "data/phase2/flipped_val", flip_ratio=1)
    else:
        flipped_val_files_synth = [os.path.join("data", "phase2", "flipped_val", f) for f in os.listdir("data/phase2/flipped_val") if f.endswith('.pkl')]
    if not os.path.exists("data/phase3/flipped_val_real_semantics"):
        flipped_val_files_real = flip_and_save(val_files_real, "data/phase3/flipped_val_real_semantics", flip_ratio=1)
    else:
        flipped_val_files_real = [os.path.join("data", "phase3", "flipped_val_real_semantics", f) for f in os.listdir("data/phase3/flipped_val_real_semantics") if f.endswith('.pkl')]



    random.shuffle(val_files_real) 
    random.shuffle(val_files_synth)
    random.shuffle(train_files)

    random.shuffle(flipped_val_files_real)
    random.shuffle(flipped_val_files_synth)
    random.shuffle(flipped_train_files)

    num_samples = int(0.4 * len(flipped_train_files))
    flipped_train_files = random.sample(flipped_train_files, num_samples)

    num_samples = int(0.4 * len(flipped_val_files_synth))
    flipped_val_files_synth = random.sample(flipped_val_files_synth, num_samples)

    train_files = train_files + flipped_train_files
    val_files_synth = val_files_synth + flipped_val_files_synth 
    real_val_files_used = val_files_real[:len(val_files_real)//2] + flipped_val_files_real[:len(flipped_val_files_real)//2]
    real_val_files_test = val_files_real[len(val_files_real)//2:] + flipped_val_files_real[len(flipped_val_files_real)//2:]

    ratios = {
        'gamma': 0.0,       # Apply gamma correction to 30%
        'noise': 0.3,       # Apply noise to 40% of the current data
        'brightness': 0.3,
        'blur': 0.3,
    }

    augment_train_data = True  # Set to True to augment training data

    if augment_train_data:
        # Augment the training syntetic training set
        if not os.path.exists("data/phase3/augmented_train"):
            augmented_train = incremental_augmentation(train_files, "data/phase3/augmented_train", ratios)
        else:
            augmented_train = [os.path.join("data/phase3/augmented_train", f) for f in os.listdir("data/phase3/augmented_train") if f.endswith('.pkl')]
            augmented_train += train_files

        # Augment the syntetic validation set
        if not os.path.exists("data/phase3/augmented_val"):
            augmented_val = incremental_augmentation(val_files_synth, "data/phase3/augmented_val", ratios)
        else:
            augmented_val = [os.path.join("data/phase3/augmented_val", f) for f in os.listdir("data/phase3/augmented_val") if f.endswith('.pkl')]
            augmented_val += val_files_synth
    else:
        # If not augmenting, just use the original files
        augmented_train = train_files
        augmented_val = val_files_synth


    # Augment the real validation set for training
    if not os.path.exists("data/phase3/augmented_val_real_semantics"):
        augmented_real_val_used = incremental_augmentation(real_val_files_used, "data/phase3/augmented_val_real_semantics", ratios)
    else:
        augmented_real_val_used = [os.path.join("data/phase3/augmented_val_real_semantics", f) for f in os.listdir("data/phase3/augmented_val_real_semantics") if f.endswith('.pkl')]
        augmented_real_val_used += real_val_files_used

    # Augment the real validation set for testing
    if not os.path.exists("data/phase3/augmented_val_real_semantics_test"):
        augmented_real_val_test = incremental_augmentation(real_val_files_test, "data/phase3/augmented_val_real_semantics_test", ratios)
    else:
        augmented_real_val_test = [os.path.join("data/phase3/augmented_val_real_semantics_test", f) for f in os.listdir("data/phase3/augmented_val_real_semantics_test") if f.endswith('.pkl')]
        augmented_real_val_test += real_val_files_test

    all_train_files = augmented_train + augmented_real_val_used
    all_val_files = augmented_real_val_test + augmented_val
    

    print("Using ", len(all_train_files), " training files and ", len(all_val_files), " validation files\n")

    variant_counts = count_augmentation_variants(all_val_files)
    print("Variant counts in validation set:\n")
    for variant, count in variant_counts.items():
        print(f"{variant}: {count} files")

    variant_counts = count_augmentation_variants(all_train_files)
    print("\nVariant counts in training set:\n")
    for variant, count in variant_counts.items():
        print(f"{variant}: {count} files")



    random.shuffle(all_val_files)
    train_dataset = DrivingDataset(all_train_files)
    val_dataset = DrivingDataset(all_val_files)

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True, 
        prefetch_factor=4, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True,
        prefetch_factor=4, persistent_workers=True, worker_init_fn=worker_init_fn)

    p  = run_experiment_with_pretrained(train_loader, val_loader, max_epochs=100)
    print("Best model path: ", p)
    print("Done!")
    
# Submission
import pandas as pd
test_data_dir = "data/phase3/test_public_real"
test_files = [os.path.join(test_data_dir, fn) for fn in sorted([f for f in os.listdir(test_data_dir) if f.endswith(".pkl")], key=lambda fn: int(os.path.splitext(fn)[0]))]

test_dataset = DrivingDataset(test_files, test=True, augment=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
checkpoint = "checkpoints/phase3/milestone3/version_15_epoch=13_val_traj_loss=0.6528_val_ade=1.5072.ckpt"

model = LightningSimplifiedDrivingPlanner.load_from_checkpoint(checkpoint)
model.eval()
all_plans = []


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
with torch.no_grad():
    for batch in test_loader:
        camera = batch['camera'].to(device)
        history = batch['history'].to(device)

        trajectories, confidences, semantic = model(camera, history)

         # Select the most confident trajectory for each sample
        best_idx = confidences.argmax(dim=1)  # (B,)
        batch_size = trajectories.size(0)

        # Use advanced indexing to select the best trajectory for each sample
        best_trajectories = trajectories[torch.arange(batch_size), best_idx]  # (B, T, 2)

        all_plans.append(best_trajectories.cpu().numpy())

all_plans = np.concatenate(all_plans, axis=0)

# Now save the plans as a csv file
pred_xy = all_plans[..., :2]  # shape: (total_samples, T, 2)

# Flatten to (total_samples, T*2)
total_samples, T, D = pred_xy.shape
pred_xy_flat = pred_xy.reshape(total_samples, T * D)

# Build a DataFrame with an ID column
ids = np.arange(total_samples)
df_xy = pd.DataFrame(pred_xy_flat)
df_xy.insert(0, "id", ids)

# Column names: id, x_1, y_1, x_2, y_2, ..., x_T, y_T
new_col_names = ["id"]
for t in range(1, T + 1):
    new_col_names.append(f"x_{t}")
    new_col_names.append(f"y_{t}")
df_xy.columns = new_col_names

# Save to CSV
df_xy.to_csv("aug_val_synt_g0_n3_br3_bl3_val_loss_0.6528_val_ade_1.5072.csv", index=False)

print(f"Shape of df_xy: {df_xy.shape}")