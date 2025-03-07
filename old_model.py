import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class R2plus1_with_attention(nn.Module):
    def __init__(self, nr_classes):
        super(R2plus1_with_attention, self).__init__()

        self.r2plus1 = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        self.r2plus1.fc = nn.Identity()  # Remove the classification head

        for param in self.r2plus1.parameters():
            param.requires_grad = False

        # Spatial Attention: Use Conv2d for each frame (so that H and W are preserved)
        self.spatial_attention = nn.Conv2d(512, 1, kernel_size=3, padding=1)  # Apply 2D convolution for spatial attention
        self.temporal_attention = nn.Conv1d(512, 1, kernel_size=1)
        
        # Classifier
        self.classifier = nn.Linear(512, nr_classes)

    def forward(self, x):
        # Extract features (B, 512, T, H, W)
        features = self.r2plus1.stem(x)
        features = self.r2plus1.layer1(features)
        #Esta é a ultima camada com o numero de frames == 16, a attention tem que entrar aqui
        features = self.r2plus1.layer2(features)
        features = self.r2plus1.layer3(features)
        features = self.r2plus1.layer4(features)
  

        # Now the features have shape (B, 512, T, H, W)

        # Spatial Attention: Apply attention to each frame (dim=H and W)
        # Use Conv2d to operate on each frame's HxW spatial resolution
        spatial_attention_map = torch.sigmoid(self.spatial_attention(features.view(-1, 512, features.shape[3], features.shape[4])))
        spatial_attention_map = spatial_attention_map.view(features.shape[0], 1, features.shape[2], features.shape[3], features.shape[4])

        weighted_features_spatial = features * spatial_attention_map
        spatial_pooled = weighted_features_spatial.mean(dim=[3, 4])  # Mean across spatial dimensions (H, W)

        temporal_attention_map = torch.sigmoid(self.temporal_attention(spatial_pooled))  # Shape: (B, 1, T)
        weighted_features_temporal = spatial_pooled * temporal_attention_map

        logits = self.classifier(weighted_features_temporal.mean(dim=2))

        return logits, spatial_attention_map, temporal_attention_map
