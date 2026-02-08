import sys
from models import *

def get_network(network, in_channels, num_classes, **kwargs):

    if network == "multi_gating_attention":
        net = multi_gating_attention(in_channels, num_classes)
    elif network == 'unet':
        net = unet(in_channels, num_classes)
    elif network == 'unet_plusplus' or network == 'unet++':
        net = unet_plusplus(in_channels, num_classes)
    elif network == 'r2unet':
        net = r2_unet(in_channels, num_classes)
    elif network == 'attunet':
        net = attention_unet(in_channels, num_classes)
    elif network == 'hrnet18':
        net = hrnet18(in_channels, num_classes)
    elif network == 'hrnet48':
        net = hrnet48(in_channels, num_classes)
    elif network == 'resunet':
        net = res_unet(in_channels, num_classes)
    elif network == 'resunet++':
        net = res_unet_plusplus(in_channels, num_classes)
    elif network == 'u2net':
        net = u2net(in_channels, num_classes)
    elif network == 'u2net_s':
        net = u2net_small(in_channels, num_classes)
    elif network == 'unet3+':
        net = unet_3plus(in_channels, num_classes)
    elif network == 'unet3+_ds':
        net = unet_3plus_ds(in_channels, num_classes)
    elif network == 'unet3+_ds_cgm':
        net = unet_3plus_ds_cgm(in_channels, num_classes)
    elif network == 'swinunet':
        net = swinunet(num_classes, 224)  # img_size = 224
    elif network == 'unet_urpc':
        net = unet_urpc(in_channels, num_classes)
    elif network == 'unet_cct':
        net = unet_cct(in_channels, num_classes)
    elif network == 'wavesnet':
        net = wsegnet_vgg16_bn(in_channels, num_classes)
    elif network == 'mwcnn':
        net = mwcnn(in_channels, num_classes)
    elif network == 'alnet':
        net = Aerial_LaneNet(in_channels, num_classes)
    elif network == 'wds':
        net = WDS(in_channels, num_classes)

    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
