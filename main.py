import argparse
import os
import time

import torch
import torch.nn as nn
import cv2
import multiprocessing
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam, AdamW
from torch.utils.data import random_split
from torchmetrics.classification import Accuracy
from tqdm import tqdm
from customDataset import CustomKineticsDataset
import matplotlib.pyplot as plt
import numpy as np

from R2plus1 import R2plus1_with_attention_v4
from util import *
from process_video import *


def parse_args():
    parser = argparse.ArgumentParser(description='Run the model for training or inference.')
    parser.add_argument('--train_mode', type=bool, default=False, help='Set the model to training mode. It will train using the provided dataset')
    parser.add_argument('--train_name',type=str, help="Name given to the checkpoint folder in training")
    parser.add_argument('--checkpoint', type=str, help="Model checkpoint to do the inference.")
    parser.add_argument('--input_video',type=str, help='Input video for the inference')

    return parser.parse_args()

def main():

    args = parse_args()
    print(time.time())
    if not os.path.exists('output'):
        os.makedirs('output')

    if args.train_mode == True:
        train_mode(args.train_name)
    else:
        inference_mode(args.checkpoint, args.input_video)


def inference_mode(checkpoint_path, video_path):
    start_time = time.time()

    model = R2plus1_with_attention_v4(nr_classes=400).cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = load_video_frames(video_path)
    if len(frames) < 16:
        print("Warning: Not enough frames. Consider padding or skipping the video.")
    inputs = preprocess_frames(frames).to(device) 

    model.eval()
    with torch.no_grad():
        logits, spatial_attention_map, temporal_attention_map = model(inputs)

    np.savez_compressed("output/model_outputs.npz",
                    logits=logits.detach().cpu().numpy(),
                    spatial_attention_map=spatial_attention_map.detach().cpu().numpy(),
                    temporal_attention_map=temporal_attention_map.detach().cpu().numpy())
    
    visualize_attention(spatial_attention_map)
    visualize_temporal_attention(temporal_attention_map)
    overlay_high_attention(inputs,attention_maps=spatial_attention_map)
    end_time = time.time()
    print(f"Done - check output folder! Execution time: {end_time - start_time:.4f} seconds")
    

def train_mode(train_name):

    if not os.path.exists(f'output/{train_name}'):
        os.makedirs(f'output/{train_name}')

    transform = transforms.Compose([
    transforms.Resize((128, 171)),  # Resize frames
    transforms.CenterCrop(112),     # Crop to 112x112
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained model
    ])

    dataset = CustomKineticsDataset(root_dir='kinetics400_5per/train',frames_per_clip=16,transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    drop_last=True, 
    num_workers=multiprocessing.cpu_count() // 2, 
    pin_memory=True,
    persistent_workers=True)

    val_loader = DataLoader(val_dataset, 
        batch_size=32, 
        shuffle=False, 
        drop_last=True, 
        num_workers=multiprocessing.cpu_count() // 2, 
        pin_memory=True,
        persistent_workers=True)
    
    model = R2plus1_with_attention_v4(nr_classes=400).cuda()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy(task="multiclass",num_classes=400).cuda()

    train_losses, train_accuracies = train_model(model, optimizer, train_loader, criterion, accuracy_metric)
    plot_metrics(train_losses, train_accuracies)

    val_model(model,val_loader, criterion, accuracy_metric)
    print("Training done! Please check the output folder!")
    

def train_model(model, optimizer, train_loader, criterion, accuracy_metric, num_epochs = 50):

    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            videos, labels = videos.cuda(), labels.cuda()
            optimizer.zero_grad()

            logits, _, _= model(videos) #

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += accuracy_metric(logits, labels)

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f}", end='\r')
        #torch.save(model, f'checkpoints/v4_checkpoints_2/R2plus1D_checkpoint_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, f"output/checkpoints/checkpoint_{epoch + 1}.pt")
    
    return train_losses, train_accuracies

def val_model(model, val_loader, criterion, accuracy_metric, num_epochs = 15):
    
    for epoch in range(num_epochs):
    # Validation loop
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                videos, labels = videos.cuda(), labels.cuda()

                # Forward pass
                logits, _, _ = model(videos)

                # Compute the loss
                loss = criterion(logits, labels)

                # Track loss and accuracy
                val_loss += loss.item()
                val_accuracy += accuracy_metric(logits, labels)

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        print(f"Validation - Loss: {val_loss:.4f} - Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()