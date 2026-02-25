import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from models.modules.feat_select.multi_scale_task_selector import MultiScaleTaskSelector


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(UNetEncoder, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        self.last_features = None

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        self.last_features = [x1, x2, x3, x4, x5]
        return x1, x2, x3, x4, x5

    def get_feature_stats(self):
        if self.last_features is None:
            return {}
        stats = {}
        for idx, feat in enumerate(self.last_features, start=1):
            mean_c = feat.mean(dim=(0, 2, 3))
            std_c = feat.std(dim=(0, 2, 3), unbiased=False)
            stats[f"enc_l{idx}_mean"] = mean_c.mean().item()
            stats[f"enc_l{idx}_std"] = std_c.mean().item()
        return stats


class UNetDecoder(nn.Module):
    def __init__(self, num_classes=1):
        super(UNetDecoder, self).__init__()
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2, x3, x4, x5):
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d2, d1


# NOTE:
# U_Net is single-task version.
# Do NOT integrate MultiScaleTaskSelector here.
# Feature selection is handled only in UNetMultiTask.
class U_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, cfg=None):
        super(U_Net, self).__init__()
        self.encoder = UNetEncoder(in_channels=in_channels)
        self.decoder = UNetDecoder(num_classes=num_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        d2, d1 = self.decoder(x1, x2, x3, x4, x5)
        return d2, d1
        # return d1


# Multi-task version with optional feature selector integration.
class UNetMultiTask(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, cfg=None):
        super(UNetMultiTask, self).__init__()
        self.encoder = UNetEncoder(in_channels=in_channels)
        self.decoder_seg = UNetDecoder(num_classes=num_classes)
        self.decoder_sdf = UNetDecoder(num_classes=1)
        self.decoder_bnd = UNetDecoder(num_classes=num_classes)
        self.num_tasks = 3
        self.selector = None
        fs_cfg = cfg if cfg is not None else None
        if fs_cfg is not None:
            self.use_feat_selector = fs_cfg["ENABLE"]
            assert fs_cfg["TYPE"] in ["task_dw", "hybrid", "expert"], "FEATURE_SELECT.TYPE must be one of [task_dw, hybrid, expert]"
            if fs_cfg["TYPE"] == "hybrid" and len(fs_cfg["HYBRID_SCALES"]) == 0:
                print("Warning: FEATURE_SELECT.TYPE is 'hybrid' but HYBRID_SCALES is empty.")
            if self.use_feat_selector:
                self.selector = MultiScaleTaskSelector(
                    in_channels_list=[64, 128, 256, 512, 1024],
                    num_tasks=self.num_tasks,
                    mode=fs_cfg["TYPE"],
                    hybrid_scales=fs_cfg["HYBRID_SCALES"],
                )
                print(
                    "Selector params:",
                    sum(p.numel() for p in self.selector.parameters()) / 1e6,
                    "M",
                )
            if self.use_feat_selector and self.selector is None:
                raise RuntimeError("FEATURE_SELECT.ENABLE is True but selector was not created.")
        else:
            self.use_feat_selector = False

    def set_selector_alpha(self, alpha: float):
        if not self.use_feat_selector or self.selector is None:
            return
        if hasattr(self.selector, "selectors"):
            for selector in self.selector.selectors:
                if hasattr(selector, "set_alpha"):
                    selector.set_alpha(alpha)
        elif hasattr(self.selector, "set_alpha"):
            self.selector.set_alpha(alpha)

    def get_selector_weight_stats(self):
        if self.selector is None:
            return {}
        if hasattr(self.selector, "get_all_weight_stats"):
            return self.selector.get_all_weight_stats()
        if hasattr(self.selector, "get_weight_stats"):
            return self.selector.get_weight_stats()
        return {}

    def forward(self, x):
        features = list(self.encoder(x))
        assert isinstance(features, list), "features must be a list of feature maps"
        assert len(features) == 5, "features must contain 5 scales (f1..f5)"

        # ---- Task-aware feature selection (template integration) ----
        if self.use_feat_selector:
            assert self.selector is not None, "selector must be initialized when USE_FEAT_SELECTOR is True"
            task_features = self.selector(features)
        else:
            # replicate original features for each task
            task_features = [
                [f for f in features]
                for _ in range(self.num_tasks)
            ]
        assert isinstance(task_features, list), "task_features must be a list"
        assert len(task_features) == self.num_tasks, "task_features length must match num_tasks"

        out_seg = self.decoder_seg(*task_features[0])
        if self.num_tasks == 1:
            return out_seg

        out_sdf = self.decoder_sdf(*task_features[1])
        out_bnd = self.decoder_bnd(*task_features[2])

        return {
            "seg": out_seg,
            "sdf": out_sdf,
            "bnd": out_bnd,
        }


class R2U_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=in_channels, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        # outputs = []
        # outputs.append(d1)
        # return outputs
        return d1

class AttU_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        # outputs = []
        # outputs.append(d1)
        # return outputs
        return d1


class R2AttU_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, t=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=in_channels, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        #输出维度不同，输出头不同
        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        # outputs = []
        # outputs.append(d1)
        # return outputs
        return d1


def unet(in_channels, num_classes):
    model = U_Net(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model


def create_unet_encoder(in_channels):
    model = UNetEncoder(in_channels=in_channels)
    init_weights(model, 'kaiming')
    return model


def create_unet_decoder(num_classes):
    model = UNetDecoder(num_classes=num_classes)
    init_weights(model, 'kaiming')
    return model


def r2_unet(in_channels, num_classes):
    model = R2U_Net(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

def attention_unet(in_channels, num_classes):
    model = AttU_Net(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

def r2_attention_unet(in_channels, num_classes):
    model = R2AttU_Net(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model


# if __name__ == '__main__':
#     model = U_Net(1,10)
#     model.eval()
#     input = torch.rand(2,1,128,128)
#     output = model(input)
#     output = output[0].data.cpu().numpy()
#     # print(output)
#     print(output.shape)
