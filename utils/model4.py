
from torch import nn

class Net_mask(nn.Module):
    """The network."""
    def __init__(self, in_feat=20, out_feat=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=20), #1
            #nn.Sigmoid(),
            #nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20), #2
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20), #3
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20), #4
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20), #5
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20), #6
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20), #7
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20), #8
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20), #9
            nn.ReLU(),
            #nn.Linear(in_features=20, out_features=20), #10
            #nn.ReLU(),
            #nn.Linear(in_features=20, out_features=20), #11
            #nn.ReLU(),
            #nn.Linear(in_features=20, out_features=20), #12
            #nn.ReLU(),
            #nn.Linear(in_features=20, out_features=20), #13
            #nn.ReLU(),
            #nn.Linear(in_features=20, out_features=20), #14
            #nn.ReLU(),
            nn.Linear(in_features=20, out_features=10), #15
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10)
        )

    def forward(self, x):
        x = self.layers(x)
        return x