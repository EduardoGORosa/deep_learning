import torch
import torch.nn as nn

class ChessModel(nn.Module):
    def __init__(self, action_size=64*64):
        super(ChessModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_common = nn.Sequential(
            nn.Linear(64*8*8, 256),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(256, action_size)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc_common(x)
        p = self.policy_head(x)
        v = torch.tanh(self.value_head(x))
        return p, v

def load_model(checkpoint_path: str):
    model = ChessModel()
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model checkpoint not found. Using untrained model.")
    return model
