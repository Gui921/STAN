import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_projection(x) 
        x = self.transformer_encoder(x) 
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, output_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        memory = x
        x = self.transformer_decoder(x, memory)
        x = self.output_projection(x)
        return x

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=768, model_dim=96, num_heads=8, num_layers=6):
        super(TransformerAutoencoder, self).__init__()

        self.encoder = TransformerEncoder(input_dim, model_dim, num_heads, num_layers)
        self.decoder = TransformerDecoder(model_dim, input_dim, num_heads, num_layers)

    def forward(self, x):
        encoded = self.encoder(x)  
        reconstructed = self.decoder(encoded)  
        return reconstructed

class StudentMLP(nn.Module):
    def __init__(self, input_dim=768, model_dim=96, hidden_dim=256):
        super(StudentMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, model_dim)
        )

    def forward(self, x):
        return self.model(x)
    
class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        def linear_block(in_features, out_features, dropout_prob = 0.5):
            return nn.Sequential(
                nn.Linear(in_features,out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            )
        
        self.encoder = nn.Sequential(
            linear_block(768,384),
            linear_block(384,96, dropout_prob=0),
            #linear_block(128,64),
            #linear_block(64,32),
            #linear_block(32, 16, dropout_prob=0)
        )
        self.decoder = nn.Sequential(
            #linear_block(128,32),
            linear_block(96,384),
            #linear_block(64,128),
            #linear_block(128,256),
            nn.Linear(384,768)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

