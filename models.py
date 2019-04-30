import torch.nn as nn

class ConvNetCam(nn.Module):

    def __init__(self, batch_size, nclass=7, nfeatures=20, length=120):
        super(ConvNetCam, self).__init__()

        self.batch_size = batch_size
        self.nclass = nclass
        self.nfeatures = nfeatures
        self.length = length

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=60, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(60),
            nn.ReLU(True),
            nn.Conv1d(in_channels=60, out_channels=50, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(50),
            nn.ReLU(True),
            nn.Conv1d(in_channels=50, out_channels=40, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(40),
            nn.ReLU(True),
            nn.Conv1d(in_channels=40, out_channels=30, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(30),
            nn.ReLU(True),
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(30),
            nn.ReLU(True),
            nn.Conv1d(in_channels=30, out_channels=nfeatures, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(nfeatures),
            nn.ReLU(True)
        )
        self.pool = nn.AvgPool1d(kernel_size=self.length)
        self.classifier = nn.Sequential(
            nn.Linear(1*nfeatures, nclass),  # 1 because global pooling reduce length of features to 1
            #nn.Softmax(1)  # Already included in nn.CrossEntropy
        )

    def forward(self, x):
        # (batch_size x length TS)
        x = self.features(x)

        # (batch_size x nfeatures x length_TS)
        # Average pooling for CAM: global pooling so set kernel size to all data
        x = self.pool(x)

        # (batch_size x nfeatures x length_pool; length_pool=1 if global pooling)
        # Flatten features (size batch, lengthpool * nchannels)
        x = x.view(self.batch_size, x.size(2)*self.nfeatures)
        x = self.classifier(x)
        return x


class ConvNetCamBi(nn.Module):
    # Consider the bivariate series as a 2D image

    def __init__(self, batch_size, nclass=7, nfeatures=20, length=120):
        super(ConvNetCamBi, self).__init__()

        self.batch_size = batch_size
        self.nclass = nclass
        self.nfeatures = nfeatures
        self.length = length

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=nfeatures, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(nfeatures),
            nn.ReLU(True)
        )
        self.pool = nn.AvgPool2d(kernel_size=(2, self.length))
        self.classifier = nn.Sequential(
            nn.Linear(1*nfeatures, nclass),  # 1 because global pooling reduce length of features to 1
            #nn.Softmax(1)  # Already included in nn.CrossEntropy
        )

    def forward(self, x):
        # (batch_size x number_pixel_row x length_TS)
        x = self.features(x)

        # (batch_size x nfeatures x number_pixel_row x length_TS)
        # Average pooling for CAM: global pooling so set kernel size to all data
        x = self.pool(x)

        # (batch_size x nfeatures x length_pool; length_pool=1 if global)
        # Flatten features (batch_size, length_pool * nchannels)
        x = x.view(self.batch_size, x.size(2)*self.nfeatures)
        x = self.classifier(x)
        return x
