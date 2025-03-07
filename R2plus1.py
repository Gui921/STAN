import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class R2plus1_with_attention(nn.Module):
    def __init__(self, nr_classes):
        super(R2plus1_with_attention, self).__init__()

        self.r2plus1 = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        self.r2plus1.fc = nn.Identity()  # Remove the classification head

        for param in self.r2plus1.parameters():
            param.requires_grad = False  # Freeze backbone

        self.spatial_attention = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Match layer1 output channels
        self.temporal_attention = nn.Conv1d(64, 1, kernel_size=1)

        self.classifier = nn.Linear(512, nr_classes)

    def forward(self, x):

        features = self.r2plus1.stem(x)   
        features = self.r2plus1.layer1(features) 

        spatial_attention_map = torch.sigmoid(self.spatial_attention(features.view(-1, 64, features.shape[3], features.shape[4])))
        spatial_attention_map = spatial_attention_map.view(features.shape[0], 1, features.shape[2], features.shape[3], features.shape[4])

        weighted_features_spatial = features * spatial_attention_map
        spatial_pooled = weighted_features_spatial.mean(dim=[3, 4])

        # Apply Temporal Attention
        temporal_attention_map = torch.sigmoid(self.temporal_attention(spatial_pooled))  # (B, 1, T)
        #weighted_features_temporal = spatial_pooled * temporal_attention_map


        features = self.r2plus1.layer2(features)  # (B, 128, 8, 28, 28)
        features = self.r2plus1.layer3(features)  # (B, 256, 4, 14, 14)
        features = self.r2plus1.layer4(features)  # (B, 512, 2, 7, 7)
        pooled_features = features.mean(dim=[2, 3, 4])  # Global Avg Pool

        logits = self.classifier(pooled_features)

        return logits, spatial_attention_map, temporal_attention_map
    

class R2plus1_with_attention_v3(nn.Module):
    def __init__(self, nr_classes):
        super(R2plus1_with_attention_v3, self).__init__()

        self.r2plus1 = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        self.r2plus1.fc = nn.Identity()  # Remove the classification head

        for param in self.r2plus1.parameters():
            param.requires_grad = False  # Freeze backbone layers (optional)

        self.spatial_attention = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Apply spatial attention on layer1 output
        self.temporal_attention = nn.Conv1d(64, 1, kernel_size=1)  # Temporal attention

        self.spatial_attention_conv3d = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.temporal_feature_reduction = nn.Linear(16, 32)

        self.classifier = nn.Linear(512 + 64 + 32, nr_classes)  # Concatenating pooled features and attention features

    def forward(self, x):
        # Get features from the R2Plus1D backbone
        features = self.r2plus1.stem(x)  # Shape: (B, 64, T, H, W)
        features = self.r2plus1.layer1(features)  # Shape: (B, 64, T, H, W)

        # SPATIAL
        spatial_attention_map = torch.sigmoid(self.spatial_attention(features.view(-1, 64, features.shape[3], features.shape[4])))  # Shape: (B, 1, T, H, W)
        spatial_attention_map = spatial_attention_map.view(features.shape[0], 1, features.shape[2], features.shape[3], features.shape[4])#([64, 1, 16, 56, 56])

        weighted_features_spatial = features * spatial_attention_map
        spatial_pooled = weighted_features_spatial.mean(dim=[3, 4])  # (B, 64, T)

        # TEMPORAL
        temporal_attention_map = torch.sigmoid(self.temporal_attention(spatial_pooled))
        temporal_attention_map = temporal_attention_map.squeeze(1) # [64,16])
        reduced_temporal = F.relu(self.temporal_feature_reduction(temporal_attention_map))
        
        # REDUCE SPATIAL
        spatial_attention_conv_output = self.spatial_attention_conv3d(spatial_attention_map)  # (B, 64, 16, 56, 56)
        spatial_reduced = F.adaptive_avg_pool3d(spatial_attention_conv_output, (1, 1, 1)).view(x.size(0), -1)

        # Global Average Pooling on R2Plus1D output
        features = self.r2plus1.layer2(features)  
        features = self.r2plus1.layer3(features)  
        features = self.r2plus1.layer4(features)  
        pooled_features = features.mean(dim=[2, 3, 4])  

        # Concatenate spatial and temporal attention features with pooled features
        final_features = torch.cat([pooled_features, spatial_reduced, reduced_temporal], dim=1)

        # Classifier to make final prediction
        logits = self.classifier(final_features)

        return logits, spatial_attention_map, temporal_attention_map

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 1x1 convolution to generate attention maps
        self.attn_conv = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Apply convolution to get attention maps
        attn_map = torch.sigmoid(self.attn_conv(x))  # Shape: (B, 1, H, W)
        return attn_map

class TemporalAttention(nn.Module):
    def __init__(self, num_frames=16):
        super(TemporalAttention, self).__init__()
        self.num_frames = num_frames
        self.attn_weights = nn.Parameter(torch.randn(num_frames))

    def forward(self, x):
        # x: (B, C, T, H, W) where T is the number of frames (16)
        B, C, T, H, W = x.shape
        
        # Reshape x to (B, T, C*H*W) to flatten spatial dimensions
        x_flat = x.view(B, T, C * H * W)  # Shape: (B, T, C*H*W)
        
        # Apply attention weights to frames (normalize with softmax)
        attn_scores = F.softmax(self.attn_weights, dim=-1)  # Shape: (T,)
        attn_scores = attn_scores.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T)
        
        # Now we need to expand attn_scores to match the batch size
        attn_scores = attn_scores.permute(0, 2, 1)
        attn_scores = attn_scores.expand(B, T, 1)  # Shape: (B, T, 1)

        # Reshape attn_scores to (B, T, C*H*W) so it can be applied to each frame's feature vector
        attn_scores_expanded = attn_scores.expand(-1, -1, C * H * W)  # Shape: (B, T, C*H*W)

        # Apply the attention scores to the flattened features
        weighted_x = x_flat * attn_scores_expanded  # Shape: (B, T, C*H*W)
        
        # Reshape back to (B, C, T, H, W)
        weighted_x = weighted_x.view(B, C, T, H, W)

        return weighted_x


class R2plus1_with_attention_v4(nn.Module):
    def __init__(self, nr_classes):
        super(R2plus1_with_attention_v4, self).__init__()

        self.r2plus1 = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        self.r2plus1.fc = nn.Identity()  # Remove the classification head

        for param in self.r2plus1.parameters():
            param.requires_grad = False  # Freeze backbone layers (optional)

        self.spatial_attention = SpatialAttention()
        self.temporal_attention = TemporalAttention()


        self.classifier = nn.Linear(512, nr_classes)  # Concatenating pooled features and attention features

    def forward(self, x):
        # MODEL INPUT: torch.Size([64, 3, 16, 112, 112])
        B, C, T, H, W = x.shape

        spatial_attention_map = []

        # Applying spatial attention to each frame 
        for t in range(T):
            frame = x[:, :, t, :, :]  # Shape: (B, C, H, W)
            attn_map = self.spatial_attention(frame)  # Shape: (B, 1, H, W)
            spatial_attention_map.append(attn_map)
            x = x.clone()
            x[:, :, t, :, :] = frame * attn_map
        
        spatial_attention_map = torch.stack(spatial_attention_map, dim=2)
        # Apply temporal attention
        temporal_features = self.temporal_attention(x)

        attn_scores = torch.mean(temporal_features, dim=(1, 3, 4))
        attn_scores = F.softmax(attn_scores, dim=-1)
        temporal_attention_map = attn_scores.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Shape: (B, 1, T, 1, 1)
        x = x * temporal_attention_map

        # Pass through R2Plus1D backbone
        features = self.r2plus1.stem(x)  # Shape: (B, 64, T, H, W)
        features = self.r2plus1.layer1(features)  # Shape: (B, 64, T, H, W)
        features = self.r2plus1.layer2(features)  
        features = self.r2plus1.layer3(features)  
        features = self.r2plus1.layer4(features)  
        pooled_features = features.mean(dim=[2, 3, 4])  # Global Average Pooling


        # Classifier to make final prediction
        logits = self.classifier(pooled_features)

        return logits, spatial_attention_map, temporal_attention_map