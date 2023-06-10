import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        
        # ----------TODO------------
        # define a network 
        # ----------TODO------------
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.AvgPool2d(1,1),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=109 * 109 * 96, out_features=2 ** 12),
            nn.ReLU(),
            nn.Linear(in_features=2 ** 12, out_features = num_class)
        )

    def forward(self, x):

        # ----------TODO------------
        # network forwarding 
        # ----------TODO------------
        features = self.feature_conv(x)
        x = self.mlp(features)
        return x


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break 
