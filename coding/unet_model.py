from common import *
from dataset import *

BatchNorm2d = nn.BatchNorm2d

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD = [0.229, 0.224, 0.225]

def upsize(x, scale_factor=2):
    # x = F.interpolate(x, size=e.shape[2:], mode='nearest')
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x

# def criterion(logit, truth, weight=[5,5,2,5]):
def criterion(logit, truth, weight=None):
    logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, len(classes)+1) #[N, C, H, W] -> [N, H, W, C]
    truth = truth.contiguous().view(-1) #[N, H, W]

    if weight is not None: weight = torch.FloatTensor([1] + weight).cuda()
    loss = F.cross_entropy(logit, truth, weight=weight, reduction='none')

    loss = loss.mean()
    return loss

# ----

### loss ###################################################################

def one_hot_encode_truth(truth, num_class=4):
    one_hot = truth.repeat(1, num_class, 1, 1)
    arange = torch.arange(1, num_class + 1).view(1, num_class, 1, 1).to(truth.device) #with values from the interval [start, end)
    one_hot = (one_hot == arange).float()
    return one_hot


def one_hot_encode_predict(predict, num_class=4):
    value, index = torch.max(predict, 1, keepdim=True)

    value = value.repeat(1, num_class, 1, 1)
    index = index.repeat(1, num_class, 1, 1)
    arange = torch.arange(1, num_class + 1).view(1, num_class, 1, 1).to(predict.device)

    one_hot = (index == arange).float()
    value = value * one_hot
    return value



def metric_hit(logit, truth, threshold=0.5):
    batch_size, num_class, H, W = logit.shape

    with torch.no_grad():
        logit = logit.view(batch_size, num_class, -1)
        truth = truth.view(batch_size, -1)

        probability = torch.softmax(logit, 1)
        p = torch.max(probability, 1)[1]
        t = truth
        correct = (p == t)

        index0 = t == 0
        index1 = t == 1
        index2 = t == 2
        index3 = t == 3
        index4 = t == 4

        num_neg = index0.sum().item()
        num_pos1 = index1.sum().item()
        num_pos2 = index2.sum().item()
        num_pos3 = index3.sum().item()
        num_pos4 = index4.sum().item()

        neg = correct[index0].sum().item() / (num_neg + 1e-12)
        pos1 = correct[index1].sum().item() / (num_pos1 + 1e-12)
        pos2 = correct[index2].sum().item() / (num_pos2 + 1e-12)
        pos3 = correct[index3].sum().item() / (num_pos3 + 1e-12)
        pos4 = correct[index4].sum().item() / (num_pos4 + 1e-12)

        num_pos = [num_pos1, num_pos2, num_pos3, num_pos4, ]
        tn = neg
        tp = [pos1, pos2, pos3, pos4, ]

    return tn, tp, num_neg, num_pos


def metric_dice(logit, truth, threshold=0.1, sum_threshold=1):
    with torch.no_grad():
        probability = torch.softmax(logit, 1)
        probability = one_hot_encode_predict(probability)
        truth = one_hot_encode_truth(truth, num_class=len(classes))

        batch_size, num_class, H, W = truth.shape
        probability = probability.view(batch_size, num_class, -1)
        truth = truth.view(batch_size, num_class, -1)
        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)

        d_neg = (p_sum < sum_threshold).float()
        d_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1) + 1e-12)

        neg_index = (t_sum == 0).float()
        pos_index = 1 - neg_index

        num_neg = neg_index.sum()
        num_pos = pos_index.sum(0)
        dn = (neg_index * d_neg).sum() / (num_neg + 1e-12)
        dp = (pos_index * d_pos).sum(0) / (num_pos + 1e-12)

        # ----

        dn = dn.item()
        dp = list(dp.data.cpu().numpy())
        num_neg = num_neg.item()
        num_pos = list(num_pos.data.cpu().numpy())

    return dn, dp, num_neg, num_pos


class FPAv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 100
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.upsample(x_glob, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=True)  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 4, 25
        d3 = self.down3_1(d2)  # 512, 2, 13

        d2 = self.down2_2(d2)  # 256, 4, 25
        d3 = self.down3_2(d3)  # 256, 2, 13

        d3 = F.upsample(d3, size=[d2.shape[2], d2.shape[3]], mode='bilinear', align_corners=True)  # 256, 4, 25
        d2 = d2 + d3

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 50
        x = self.conv1(x)  # 256, 8, 50
        x = x * d2

        x = x + x_glob

        return x

def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, dilation=rate, padding=rate, bias=False),
                         nn.BatchNorm2d(output_dim),
                         nn.ELU(True))

class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)
        self.s_att = SpatialAttention2d(out_channels)
        self.c_att = GAB(out_channels, 16)

    def forward(self, x, e=None):
        x = F.upsample(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        s = self.s_att(x)
        c = self.c_att(x)
        output = s + c
        return output

class Decoderv2(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super(Decoderv2, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(True)
        self.s_att = SpatialAttention2d(n_out)
        self.c_att = GAB(n_out, 16)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)

        cat_p = torch.cat([up_p, x_p], 1)
        cat_p = self.relu(self.bn(cat_p))
        s = self.s_att(cat_p)
        c = self.c_att(cat_p)
        return s + c

class Decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decode, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),

            nn.Conv2d(out_channel // 2, out_channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),

            nn.Conv2d(out_channel // 2, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),  # Swish(), #
        )

    def forward(self, x):
        x = self.top(torch.cat(x, 1))
        return x

# stage1 model
class Res18Unetv4(nn.Module):
    def __init__(self):
        super(Res18Unetv4, self).__init__()
        self.resnet = torchvision.models.resnet18(True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)

        self.encode2 = self.resnet.layer1
        self.encode3 = self.resnet.layer2
        self.encode4 = self.resnet.layer3
        self.encode5 = self.resnet.layer4

        self.center = nn.Sequential(FPAv2(512, 256),
                                    nn.MaxPool2d(2, 2))

        self.decode5 = Decoderv2(256, 512, 64)
        self.decode4 = Decoderv2(64, 256, 64)
        self.decode3 = Decoderv2(64, 128, 64)
        self.decode2 = Decoderv2(64, 64, 64)
        self.decode1 = Decoder(64, 32, 64)

        self.logit = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(64, 1, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (batch_size, 3, 256, 256)

        x = self.conv1(x)  # 64, 128, 128
        e2 = self.encode2(x)  # 64, 128, 128
        e3 = self.encode3(e2)  # 128, 64, 64
        e4 = self.encode4(e3)  # 256, 32, 32
        e5 = self.encode5(e4)  # 512, 16, 16

        f = self.center(e5)  # 256, 8, 8

        d5 = self.decode5(f, e5)  # 64, 16, 16
        d4 = self.decode4(d5, e4)  # 64, 32, 32
        d3 = self.decode3(d4, e3)  # 64, 64, 64
        d2 = self.decode2(d3, e2)  # 64, 128, 128
        d1 = self.decode1(d2)  # 64, 256, 256

        f = torch.cat((d1,
                       F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 320, 256, 256

        logit = self.logit(f)  # 1, 256, 256

        return logit


class Resnet18_supercolumn_channel128(nn.Module):
    '''
    NO FPA
    '''
    def __init__(self, num_class=5, drop_connect_rate=0.2):
        super(Resnet18_supercolumn_channel64, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)

        self.encode2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            self.resnet.layer1)

        self.encode3 = self.resnet.layer2
        self.encode4 = self.resnet.layer3
        self.encode5 = self.resnet.layer4

        self.decode5 = Decode(512, 128)  # 64 or 128?
        self.decode4 = Decode(128 + 256, 128)
        self.decode3 = Decode(128 + 128, 128)
        self.decode2 = Decode(128 + 64, 128)
        self.decode1 = Decode(128, 128)

        self.logit = nn.Sequential(nn.Conv2d(128 * 5, 64, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(64, num_class, kernel_size=1, bias=False))

    def forward(self, x):
        batch_size, C, H, W = x.shape
        # x: (batch_size, 3, 1600, 256)

        # ----------------------------------
        # contraction
        x = self.conv1(x)  # 64, 128, 800
        e2 = self.encode2(x)  # 64, 64, 400
        e3 = self.encode3(e2)  # 128, 32, 200
        e4 = self.encode4(e3)  # 256, 16, 100
        e5 = self.encode5(e4)  # 512, 8, 50

        # ----------------------------------
        x = None
        d5 = self.decode5([e5, ])  # 128, 8, 50
        e5 = None
        d4 = self.decode4([e4, upsize(d5)])  # 128, 16, 100
        e4 = None
        d3 = self.decode3([e3, upsize(d4)])  # 128, 32, 200
        e3 = None
        d2 = self.decode2([e2, upsize(d3)])  # 128, 64, 400
        e2 = None
        d1 = self.decode1([upsize(d2), ])  # 128, 256, 1600

        f = torch.cat((d1,
                       F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 640, 128, 800

        logit = self.logit(f)  # 5, 128, 800
        f = None
        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)

        return logit

class Resnet18_supercolumn_channel64(nn.Module):
    '''
    NO FPA
    '''
    def __init__(self, num_class=5, drop_connect_rate=0.2):
        super(Resnet18_supercolumn_channel64, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)

        self.encode2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            self.resnet.layer1)

        self.encode3 = self.resnet.layer2
        self.encode4 = self.resnet.layer3
        self.encode5 = self.resnet.layer4

        # self.decode5 = Decoderv2(256, 512, 64)
        # self.decode4 = Decoderv2(64, 256, 64)
        # self.decode3 = Decoderv2(64, 128, 64)
        # self.decode2 = Decoderv2(64, 64, 64)
        # self.decode1 = Decoder(64, 32, 64)
        self.decode5 = Decode(512, 64)  # 64 or 128?
        self.decode4 = Decode(64 + 256, 64)
        self.decode3 = Decode(64 + 128, 64)
        self.decode2 = Decode(64 + 64, 64)
        self.decode1 = Decode(64, 64)

        self.logit = nn.Sequential(nn.Conv2d(128 * 5, 64, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(64, num_class, kernel_size=1, bias=False))

    def forward(self, x):
        batch_size, C, H, W = x.shape
        # x: (batch_size, 3, 1600, 256)

        # ----------------------------------
        # contraction
        x = self.conv1(x)  # 64, 128, 800
        e2 = self.encode2(x)  # 64, 64, 400
        e3 = self.encode3(e2)  # 128, 32, 200
        e4 = self.encode4(e3)  # 256, 16, 100
        e5 = self.encode5(e4)  # 512, 8, 50

        # ----------------------------------
        x = None
        d5 = self.decode5([e5, ])  # 128, 8, 50
        e5 = None
        d4 = self.decode4([e4, upsize(d5)])  # 128, 16, 100
        e4 = None
        d3 = self.decode3([e3, upsize(d4)])  # 128, 32, 200
        e3 = None
        d2 = self.decode2([e2, upsize(d3)])  # 128, 64, 400
        e2 = None
        d1 = self.decode1([upsize(d2), ])  # 128, 256, 1600

        f = torch.cat((d1,
                       F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 640, 128, 800

        logit = self.logit(f)  # 5, 128, 800
        f = None
        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)

        return logit
class Resnet18_supercolumn_FPA_channel64(nn.Module):
    def __init__(self, num_class=5, drop_connect_rate=0.2):
        super(Resnet18_supercolumn_channel64, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)

        self.encode2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            self.resnet.layer1)
        
        self.encode3 = self.resnet.layer2
        self.encode4 = self.resnet.layer3
        self.encode5 = self.resnet.layer4

        self.center = nn.Sequential(FPAv2(512, 256),
                                    nn.MaxPool2d(2, 2))

        self.decode5 = Decode(256 + 512, 64) # 64 or 128?
        self.decode4 = Decode(64 + 256, 64)
        self.decode3 = Decode(64 + 128, 64)
        self.decode2 = Decode(64 + 64, 64)
        self.decode1 = Decode(64, 64)

        self.logit = nn.Sequential(nn.Conv2d(64 * 5, 64, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(64, num_class, kernel_size=1, bias=False))


    def forward(self, x):
        batch_size, C, H, W = x.shape
        # x: (batch_size, 3, 1600, 256)

        # ----------------------------------
        # contraction
        x = self.conv1(x)  # 64, 128, 800
        e2 = self.encode2(x)  # 64, 64, 400
        e3 = self.encode3(e2)  # 128, 32, 200
        e4 = self.encode4(e3)  # 256, 16, 100
        e5 = self.encode5(e4)  # 512, 8, 50

        f = self.center(e5)  # 256, 4, 25

        # ----------------------------------
        x = None
        d5 = self.decode5([e5, upsize(f)])  # 128, 8, 50
        e5 = None
        d4 = self.decode4([e4, upsize(d5)])  # 128, 16, 100
        e4 = None
        d3 = self.decode3([e3, upsize(d4)])  # 128, 32, 200
        e3 = None
        d2 = self.decode2([e2, upsize(d3)])  # 128, 64, 400
        e2 = None
        d1 = self.decode1([upsize(d2), ])  # 128, 256, 1600

        f = torch.cat((d1,
                       F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 640, 128, 800

        logit = self.logit(f)  # 5, 128, 800
        f = None
        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)

        return logit

class Resnet18_supercolumn_FPA_channel128(nn.Module):
    def __init__(self, num_class=5, drop_connect_rate=0.2):
        super(Resnet18_supercolumn_channel64, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)

        self.encode2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            self.resnet.layer1)

        self.encode3 = self.resnet.layer2
        self.encode4 = self.resnet.layer3
        self.encode5 = self.resnet.layer4

        self.center = nn.Sequential(FPAv2(512, 256),
                                    nn.MaxPool2d(2, 2))

        self.decode5 = Decode(256 + 512, 128)  # 64 or 128?
        self.decode4 = Decode(128 + 256, 128)
        self.decode3 = Decode(128 + 128, 128)
        self.decode2 = Decode(128 + 64, 128)
        self.decode1 = Decode(128, 128)

        self.logit = nn.Sequential(nn.Conv2d(128 * 5, 64, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(64, num_class, kernel_size=1, bias=False))

    def forward(self, x):
        batch_size, C, H, W = x.shape
        # x: (batch_size, 3, 1600, 256)

        # ----------------------------------
        # contraction
        x = self.conv1(x)  # 64, 128, 800
        e2 = self.encode2(x)  # 64, 64, 400
        e3 = self.encode3(e2)  # 128, 32, 200
        e4 = self.encode4(e3)  # 256, 16, 100
        e5 = self.encode5(e4)  # 512, 8, 50

        f = self.center(e5)  # 256, 4, 25

        # ----------------------------------
        x = None
        d5 = self.decode5([e5, upsize(f)])  # 128, 8, 50
        e5 = None
        d4 = self.decode4([e4, upsize(d5)])  # 128, 16, 100
        e4 = None
        d3 = self.decode3([e3, upsize(d4)])  # 128, 32, 200
        e3 = None
        d2 = self.decode2([e2, upsize(d3)])  # 128, 64, 400
        e2 = None
        d1 = self.decode1([upsize(d2), ])  # 128, 256, 1600

        f = torch.cat((d1,
                       F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 640, 128, 800

        logit = self.logit(f)  # 5, 128, 800
        f = None
        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)

        return logit


class Resnet18_supercolumn_FPA_SCse_channel64(nn.Module):
    def __init__(self, num_class=5, drop_connect_rate=0.2):
        super(Resnet18_supercolumn_FPA_SCse_channel64, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)

        self.encode2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            self.resnet.layer1,
            SCse(64))

        self.encode3 = nn.Sequential(self.resnet.layer2,
                                     SCse(128))
        self.encode4 = nn.Sequential(self.resnet.layer3,
                                     SCse(256))
        self.encode5 = nn.Sequential(self.resnet.layer4,
                                     SCse(512))

        self.center = nn.Sequential(FPAv2(512, 256),
                                    nn.MaxPool2d(2, 2))

        self.decode5 = Decoderv2(256, 512, 64)
        self.decode4 = Decoderv2(64, 256, 64)
        self.decode3 = Decoderv2(64, 128, 64)
        self.decode2 = Decoderv2(64, 64, 64)
        self.decode1 = Decoder(64, 32, 64)

        self.logit = nn.Sequential(nn.Conv2d(64 * 5, 64, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(64, num_class, kernel_size=1, bias=False))

    def forward(self, x):
        batch_size, C, H, W = x.shape
        # x: (batch_size, 3, 1600, 256)

        # ----------------------------------
        # contraction
        x = self.conv1(x)  # 64, 128, 800
        e2 = self.encode2(x)  # 64, 64, 400
        e3 = self.encode3(e2)  # 128, 32, 200
        e4 = self.encode4(e3)  # 256, 16, 100
        e5 = self.encode5(e4)  # 512, 8, 50

        f = self.center(e5)  # 256, 4, 25

        # ----------------------------------
        x = None
        d5 = self.decode5(f, e5)  # 64, 8, 50
        e5 = None
        d4 = self.decode4(d5, e4)  # 64, 16, 100
        e4 = None
        d3 = self.decode3(d4, e3)  # 64, 32, 200
        e3 = None
        d2 = self.decode2(d3, e2)  # 64,  64, 400
        e2 = None
        d1 = self.decode1(d2)  # 64, 128, 800

        f = torch.cat((d1,
                       F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 64*5, 128, 800

        logit = self.logit(f)  # 5, 128, 800
        f = None
        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False) # 5, 256, 1200

        return logit

##############################################################################################
def make_dummy_data(folder='256x256', batch_size=8):
    image_file = glob.glob(os.path.join(config.data_dir, 'dump/%s/image/*.png' % folder))  # 32
    image_file = sorted(image_file)

    input = []
    truth_mask = []
    truth_label = []
    for b in range(0, batch_size):
        i = b % len(image_file)
        image = cv2.imread(image_file[i], cv2.IMREAD_COLOR)
        mask = np.load(image_file[i].replace('/image/', '/mask/').replace('.png', '.npy'))
        label = (mask.reshape(4, -1).sum(1) > 0).astype(np.int32)

        num_class, H, W = mask.shape
        mask = mask.transpose(1, 2, 0) * [1, 2, 3, 4]
        mask = mask.reshape(-1, 4)
        mask = mask.max(-1).reshape(1, H, W)

        input.append(image)
        truth_mask.append(mask)
        truth_label.append(label)

    input = np.array(input)
    input = image_to_input(input, IMAGE_RGB_MEAN, IMAGE_RGB_STD)

    truth_mask = np.array(truth_mask)
    truth_label = np.array(truth_label)

    infor = None

    return input, truth_mask, truth_label, infor

def run_check_net():
    batch_size = 1
    C, H, W = 6, 336, 336

    input = np.random.uniform(-1, 1, (batch_size, C, H, W))
    input = torch.from_numpy(input).float().cuda()

    net = UNet(in_channels=6, n_classes=len(classes)+1, wf=5, depth=4, padding=True, up_mode='upsample').cuda()
    net.eval()

    with torch.no_grad():
        logit = net(input)

    print('')
    print('input: ', input.shape)
    print('logit: ', logit.shape)
    # print(net)

def run_check_train():
    loss_weight = [0.2] + [1.0] * (len(classes)-1)
    if 1:
        input, truth_mask, truth_label, infor = make_dummy_data(folder='256x256', batch_size=2)
        batch_size, C, H, W = input.shape

        print('input shape:{}'.format(input.shape))
        print("truth label shape: {}".format(truth_label.shape))
        print("truth mask shape: {}".format(truth_mask.shape))
        print("truth label.sum :{}".format(truth_label.sum(0)))

    # ---
    truth_mask = torch.from_numpy(truth_mask).long().cuda()
    truth_label = torch.from_numpy(truth_label).float().cuda()
    input = torch.from_numpy(input).float().cuda()

    net = Resnet18_supercolumn_channel64().cuda()
    net = net.eval()

    with torch.no_grad():
        logit = net(input)
        loss = criterion(logit, truth_mask)
        tn, tp, num_neg, num_pos = metric_hit(logit, truth_mask)
        dn, dp, num_neg, num_pos = metric_dice(logit, truth_mask)

        print('loss = %0.5f' % loss.item())
        print('tn,tp = %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f] ' % (tn, tp[0], tp[1], tp[2], tp[3]))
        print('dn,dp = %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f] ' % (dn, dp[0], dp[1], dp[2], dp[3]))
        print('num_pos,num_neg = %d, [%d,%d,%d,%d] ' % (num_neg, num_pos[0], num_pos[1], num_pos[2], num_pos[3]))
        print('')

    # exit(0)
    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0001)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.001)

    print('batch_size =', batch_size)
    print('----------------------------------------------------------------------')
    print('[iter ]  loss     |  tn, [tp1,tp2,tp3,tp4]  |  dn, [dp1,dp2,dp3,dp4]  ')
    print('----------------------------------------------------------------------')
    # [00000]  0.70383  | 0.00000, 0.46449

    i = 0
    optimizer.zero_grad()
    while i <= 200:

        net.train()
        optimizer.zero_grad()

        logit = net(input)
        loss = criterion(logit, truth_mask, loss_weight)
        tn, tp, num_neg, num_pos = metric_hit(logit, truth_mask)
        dn, dp, num_neg, num_pos = metric_dice(logit, truth_mask)

        (loss).backward()
        optimizer.step()

        if i % 10 == 0:
            print('[%05d] %8.5f  | %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f]  | %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f] ' % (
                i,
                loss.item(),
                tn, tp[0], tp[1], tp[2], tp[3],
                dn, dp[0], dp[1], dp[2], dp[3],
            ))
        i = i + 1
    print('')

    if 1:
        # net.eval()
        logit = net(input)
        probability = torch.softmax(logit, 1)
        probability = one_hot_encode_predict(probability)
        truth_mask = one_hot_encode_truth(truth_mask)

        probability_mask = probability.data.cpu().numpy()
        truth_label = truth_label.data.cpu().numpy()
        truth_mask = truth_mask.data.cpu().numpy()
        image = input_to_image(input, IMAGE_RGB_MEAN, IMAGE_RGB_STD)

        for b in range(batch_size):
            print('%2d ------ ' % (b))
            result = draw_predict_result(image[b], truth_mask[b], truth_label[b], probability_mask[b])
            image_show('result', result, resize=0.5)
            cv2.waitKey(0)

# This implementation was copied from https://github.com/jvanvugt/pytorch-unet, it is MIT licensed.

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # run_check_basenet()
    # run_check_net()
    run_check_train()

    print('\nsucess!')
