
from torch import nn

class Net_mask(nn.Module):
    """The network."""
    def __init__(self, in_feat=20, out_feat=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=220), #1
            #nn.Sigmoid(),
            #nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=220, out_features=220), #2
            nn.ReLU(),
            nn.Linear(in_features=220, out_features=220), #3
            nn.ReLU(),
            nn.Linear(in_features=220, out_features=220), #4
            nn.ReLU(),
            nn.Linear(in_features=220, out_features=220), #5
            nn.ReLU(),
            nn.Linear(in_features=220, out_features=220), #6
            nn.ReLU(),
            nn.Linear(in_features=220, out_features=220), #7
            nn.ReLU(),
            nn.Linear(in_features=220, out_features=220), #8
            nn.ReLU(),
            nn.Linear(in_features=220, out_features=out_feat)
        )

    def forward(self, x):
        x = self.layers(x)
        return x