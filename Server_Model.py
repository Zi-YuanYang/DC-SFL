import torch
import torch.nn as nn
from torchsummary import summary
from torchstat import stat
from thop import profile

class Client_Model1(nn.Module):
    def __init__(self, out_ch=96):
        super(Client_Model1, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        return out

class Client_Model2(nn.Module):
    def __init__(self, out_ch=96):
        super(Client_Model2, self).__init__()
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x, inp):

        out = self.tconv5(self.relu(x))
        out = out + inp
        out = self.relu(out)
        return out

class Client_Model2_Test(nn.Module):
    def __init__(self, out_ch=96):
        super(Client_Model2_Test, self).__init__()
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.tconv5(self.relu(x))
        out = out + out
        out = self.relu(out)
        return out

class Server_Model(nn.Module):
    def __init__(self, out_ch=96):
        super(Server_Model, self).__init__()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.relu(self.conv2(x))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        return out


if __name__ == '__main__':
    # model = Client_Model1()
    model = Server_Model().cuda()
    summary(model,(96,252,252))
    stat(model.cpu(),(96,252,252))

    input = torch.randn(1, 96, 252, 252)
    flops, params = profile(model.cpu(), (input,))
    print(flops/1e6)
#    red = RED_CNN()
#    red.cuda()
#    summary(red, input_size=(1, 128, 128))