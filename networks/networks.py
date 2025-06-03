import torch
import torch.nn as nn
import torch.nn.functional as F
# networks.
# from networks.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
# from networks.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# from networks.decoders import EMCAD
from pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from decoders import MCAFTM

class MCAFTM(nn.Module):
    def __init__(self, num_classes=9, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu', encoder='pvt_v2_b2', pretrain=True, image_size=224):
        super(MCAFTM, self).__init__()
        self.image_size = image_size
        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        if encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0()
            path = './pretrained_pth/pvt/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            self.backbone = pvt_v2_b1()
            path = './pretrained_pth/pvt/pvt_v2_b1.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            path = '/root/autodl-tmp/networks/pretrained_pth/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            self.backbone = pvt_v2_b3()
            path = './pretrained_pth/pvt/pvt_v2_b3.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            self.backbone = pvt_v2_b4()
            path = './pretrained_pth/pvt/pvt_v2_b4.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            self.backbone = pvt_v2_b5() 
            path = './pretrained_pth/pvt/pvt_v2_b5.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain)
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            self.backbone = pvt_v2_b2()  
            path = './pretrained_pth/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
            
        if pretrain==True and 'pvt_v2' in encoder:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        
        # print('Model %s created, param count: %d' %
        #              (encoder+' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        
        #   decoder initialization
        self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation, image_size=image_size)
        
        # print('Model %s created, param count: %d' %
        #              ('EMCAD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        self.out_head_dw = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head4 = nn.Conv2d(channels[3], num_classes, 1)
        
    def forward(self, x,rate=None):
        if rate != None :
            H = int(round(self.image_size  * rate / 32) * 32)
            W = int(round(self.image_size  * rate / 32) * 32)

            # if grayscale input, convert to 3 channels
            if x.size()[1] == 1:
                x = self.conv(x)

            # encoder
            x1, x2, x3, x4 = self.backbone(x)
            # print(x1.shape, x2.shape, x3.shape, x4.shape)
            # torch.Size([4, 64, 88, 88]) torch.Size([4, 128, 44, 44]) torch.Size([4, 320, 22, 22]) torch.Size([4, 512, 11, 11])

            # decoder
            dec_outs = self.decoder(x4, x3, x2, x1, H ,W)

            # prediction heads  
            # p_dw = self.out_head_dw(dec_outs[0])
            p1 = self.out_head1(dec_outs[0])
            p2 = self.out_head2(dec_outs[1])
            p3 = self.out_head3(dec_outs[2])
            p4 = self.out_head4(dec_outs[3])
            
            # p_dw = F.interpolate(p_dw, scale_factor=8, mode='bilinear')
            p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
            p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
            p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
            p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')


        else:
            # if grayscale input, convert to 3 channels
            H = self.image_size
            W = self.image_size
            if x.size()[1] == 1:
                x = self.conv(x)
            
            # encoder
            x1, x2, x3, x4 = self.backbone(x)
            # print(x1.shape, x2.shape, x3.shape, x4.shape)
            # torch.Size([4, 64, 88, 88]) torch.Size([4, 128, 44, 44]) torch.Size([4, 320, 22, 22]) torch.Size([4, 512, 11, 11])

            # decoder
            dec_outs = self.decoder(x4, x3, x2, x1, H, W)

            # prediction heads  
            # p_dw = self.out_head_dw(dec_outs[0])
            p1 = self.out_head1(dec_outs[0])
            p2 = self.out_head2(dec_outs[1])
            p3 = self.out_head3(dec_outs[2])
            p4 = self.out_head4(dec_outs[3])
            
            # p_dw = F.interpolate(p_dw, scale_factor=8, mode='bilinear')
            p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
            p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
            p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
            p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')

        
        return  p1, p2, p3, p4
               

        
if __name__ == '__main__':
    model = MCAFTM().cuda()
    x = torch.randn(4, 3, 224, 224).cuda()
    # x3 = torch.randn(4, 320, 22, 22).cuda()
    # x2 = torch.randn(4, 128, 44, 44).cuda()
    # x1 = torch.randn(4, 64, 88, 88).cuda()
    # predict1, predict2, predict3, predict4 = model(x)
    # print(predict1.shape)  #  deep_supervision true   predict[0] [2, 1, 256, 256] , predict[1] [2, 1, 128, 128] 这两项用于监督
    # print(predict2.shape)
    # print(predict3.shape)
    # print(predict4.shape)
    predict1, predict2, predict3, predict4 = model(x,None)
    print(predict1.shape)
    print(predict2.shape)
    print(predict3.shape)
    print(predict4.shape)