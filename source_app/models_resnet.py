# Adapted from https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278 and https://www.bigrabbitdata.com/pytorch-17-residual-network-resnet-explained-in-detail-with-implementation-cifar10/
import torch
import torch.nn as nn
from functools import partial

# -----------------------------------------------------------------------------
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1, bias=False)
def conv3x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3,5), stride=stride, padding=1, bias=False)

# -----------------------------------------------------------------------------
class ResidualBlockSkip2Bi(nn.Module):
    '''
        Conv--> Batchnorm-->ReLu-->Conv-->Batchnorm--> 
        Only downsample if needed
    '''
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlockSkip2Bi, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# -----------------------------------------------------------------------------
class ResNetBi(nn.Module):
    def __init__(self, block, batch_size, nclass=7, nfeatures=20, length=120, nResblocks_layer1=2, nResblocks_layer2=2, nResblocks_layer3=2):
        super(ResNetBi, self).__init__()
        self.batch_size = batch_size
        self.nclass = nclass
        self.length = length
        self.nfeatures = nfeatures

        self.in_channels = 20
        self.layers = [nResblocks_layer1, nResblocks_layer2, nResblocks_layer3]
        # For the initial convolution layer, without residuals. Consider one channel with 2 rows of pixels
        self.conv = conv3x3(in_channels=1, out_channels=20)
        self.bn = nn.BatchNorm2d(20)
        self.relu = nn.ReLU(inplace=True)
        # Build the residual layers, each layer comprises several residual blocks
        self.layer1 = self.make_layer(block, out_channels=20, nResblocks=self.layers[0], stride=1)
        self.layer2 = self.make_layer(block, out_channels=20, nResblocks=self.layers[1], stride=1)
        self.layer3 = self.make_layer(block, out_channels=nfeatures, nResblocks=self.layers[2], stride=1)

        # Put all residual blocks into one container
        self.features = nn.Sequential(
            self.conv,
            self.bn,
            self.relu,
            self.layer1,
            self.layer2,
            self.layer3
        )

        # Flatten the output of the residuals blocks into a 1d vector 
        self.pool = nn.AvgPool2d(kernel_size=(2, self.length))
        self.classifier = nn.Sequential(
            nn.Linear(1*nfeatures, nclass),  # 1 because global pooling reduce length of features to 1
            #nn.Softmax(1)  # Already included in nn.CrossEntropy
        )
        
    def make_layer(self, blocktype, out_channels, nResblocks, stride=1):
        downsample = None
        # Only downsample when stride is not 2 
        # or when input channel doesn't match output channel
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(self.in_channels, out_channels, stride=stride),
                                       nn.BatchNorm2d(out_channels))
        layers = []
        # append the first residual block for each layer
        # downsample the image if needed
        layers.append(blocktype(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        # append the suceeding residual block in each layer
        for ii in range(1, nResblocks):
            layers.append(blocktype(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.features(x)
        out = self.pool(out)
        out = out.view(self.batch_size, out.size(2)*self.nfeatures)
        out = self.classifier(out)
        return out

#model = ResNetBi(ResidualBlockSkip2Bi, batch_size=1, nResblocks_layer1=2, nResblocks_layer2=2, nResblocks_layer3=2).to('cpu')
