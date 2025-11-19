from torch import nn


# Definiere ein einfaches neuronales Netz für Regression
class InsuranceRegressor(nn.Module):
    def __init__(self, input_dim):
        super(InsuranceRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x)