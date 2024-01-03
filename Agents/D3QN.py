from torch import nn


class D3QN(nn.Module):
    def __init__(self, observation_shape, action_size, use_conv=True):
        super().__init__()

        self.action_size = action_size

        # Because frames (4) frames are stacked we use them as if they were channels
        self.rolling_frames = observation_shape[0] if use_conv else observation_shape

        self.conv = nn.Sequential(
            nn.Conv2d(self.rolling_frames, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=4, stride=2), nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten()
        ) if use_conv else nn.Sequential()

        self.adv = nn.Sequential(
            nn.LazyLinear(512), nn.ReLU(),
            nn.LazyLinear(self.action_size)
        )
        self.val = nn.Sequential(
            nn.LazyLinear(512), nn.ReLU(),
            nn.LazyLinear(1)
        )
        self.net = nn.Sequential(self.conv, self.fc)

    def forward(self, state):
        return self.net(state)

        self.adv1 = nn.Linear(self.feature_size(), 512)
        self.adv2 = nn.Linear(512, self.num_actions)

        self.val1 = nn.Linear(self.feature_size(), 512)
        self.val2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()
