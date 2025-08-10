import torch.nn as nn

class GlobalDecoder(nn.Module):
    def __init__(self, text_size, vae_size, vision_hub_hidden_dim, vision_hub_dropout):
        super().__init__()
        self.project_net = nn.Sequential(
            nn.Linear(text_size, vision_hub_hidden_dim),
            nn.ReLU(),
            nn.Dropout(vision_hub_dropout),
            nn.Linear(vision_hub_hidden_dim, vision_hub_hidden_dim*3),
            nn.ReLU(),
            nn.Dropout(vision_hub_dropout),
            nn.Linear(vision_hub_hidden_dim*3, vision_hub_hidden_dim),
            nn.ReLU(),
            nn.Dropout(vision_hub_dropout),
            nn.Linear(vision_hub_hidden_dim, vae_size),
        )
        

    def forward(self, text_token_embedding):        
        projected_feature = self.project_net(text_token_embedding)
        
        return projected_feature
