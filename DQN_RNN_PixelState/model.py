import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv_layer1 = nn.Conv3d(3, 128, kernel_size = (1,3,3), stride=(1,3,3))
        self.btm_layer1 = nn.BatchNorm3d(128)
        self.conv_layer2 = nn.Conv3d(128, 256, kernel_size = (1,3,3), stride=(1,3,3))
        self.btm_layer2 = nn.BatchNorm3d(256)
        self.conv_layer3 = nn.Conv3d(256, 512, kernel_size = (4,3,3), stride=(1,3,3))
        self.btm_layer3 = nn.BatchNorm3d(512)

        out_size = self._calculate_conv_out_size(state_size)

        fc = [out_size, 128, 64, 32]
        self.fc1 = nn.Linear(fc[0],fc[1])
        self.fc2 = nn.Linear(fc[1],fc[2])
        self.fc3 = nn.Linear(fc[2],fc[3])

    def forward(self, state):
        #print(state.shape)
        x = self._cnn(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def _calculate_conv_out_size(self, shape):
        x = torch.rand(shape)
        #print(x.shape)
        x = self._cnn(x)
        n_size = x.data.view(1, -1).size(1)
        return n_size

    def _cnn(self, x):
        x = F.relu(self.btm_layer1(self.conv_layer1(x)))
        x = F.relu(self.btm_layer2(self.conv_layer2(x)))
        x = F.relu(self.btm_layer3(self.conv_layer3(x)))
        x = x.view(x.size(0), -1)
        return x
