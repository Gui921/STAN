import matplotlib.pyplot as plt
import cv2 
import numpy as np


# CHANGE THIS SO THAT IT SAVES THE IMAGE
def plot_metrics(train_losses, train_accuracies):

    train_losses = [t.cpu().item() if hasattr(t, 'device') else t for t in train_losses]
    train_accuracies = [t.cpu().item() if hasattr(t, 'device') else t for t in train_accuracies]

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='b', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='g', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('output/training_metrics.png')
    plt.close()

def visualize_attention(attention_map):

    attention_map = attention_map.detach().cpu().numpy()
    attention_map = attention_map[0, 0]  # Shape: [16, 56, 56]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i in range(16):
        row, col = divmod(i, 4)
        attn = attention_map[i]

        attn = (attn - np.min(attn)) / (np.max(attn) - np.min(attn))

        axes[row, col].imshow(attn, cmap='jet')
        axes[row, col].set_title(f'Frame {i + 1}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('output/spatial_attention.png')
    plt.close()

def visualize_temporal_attention(attention_map):

    attention_map = attention_map.detach().cpu().numpy()

    # Remove batch and channel dimensions, leaving shape [16]
    attention_map = attention_map[0, 0]  

    # Normalize values between 0 and 1
    attention_map = attention_map.squeeze()
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 17), attention_map, color='royalblue')

    plt.xlabel("Frame Index")
    plt.ylabel("Attention Weight")
    plt.title("Temporal Attention Across Frames")
    plt.xticks(range(1, 17))
    plt.ylim(0, 1)

    plt.savefig('output/temporal_attention.png')
    plt.close()

def overlay_high_attention(input_frames, attention_maps, alpha=0.3, threshold=0.8):

    frames = input_frames.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    attention_maps = attention_maps.squeeze(0).squeeze(0).cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frames = frames * std + mean
    frames = np.clip(frames, 0, 1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(16):
        frame = frames[i]
        attn = attention_maps[i]

        attn_resized = cv2.resize(attn, (112, 112), interpolation=cv2.INTER_LINEAR)

        if attn_resized.max() > attn_resized.min():
            attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())
        else:
            attn_resized = np.zeros_like(attn_resized)

        mask = attn_resized > threshold

        attn_colored = plt.cm.jet(attn_resized)[:, :, :3]

        overlay = frame.copy()
        overlay[mask] = (1 - alpha) * frame[mask] + alpha * attn_colored[mask]

        overlay = np.clip(overlay, 0, 1)

        axes[i].imshow(overlay)
        axes[i].set_title(f"Frame {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig('output/high_attention_overlay.png')
    plt.close()