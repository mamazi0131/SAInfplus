from components import *
import numpy as np
from scipy.interpolate import interp1d

class MLP(BaseModel):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.whether_stay_head = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 1)) #

    def forward(self, pair_feature):
        logit = self.whether_stay_head(pair_feature)
        return logit