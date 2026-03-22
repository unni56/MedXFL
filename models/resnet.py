class TBResNet(nn.Module):
    def __init__(self, num_classes=14):  
        super(TBResNet, self).__init__()

        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)