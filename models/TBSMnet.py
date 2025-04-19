import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *
from models.modules import *
from models.shuffle_attention import *
from vit_pytorch.deepvit import DeepViT
import math

class TBSMnet(nn.Module):
    def __init__(self):
        super(TBSMnet, self).__init__()
        ###############################################################
        ## scale     #  1  #  1/2  #  1/4  #  1/8  #  1/16  #  1/32  ##
        ## channels  #  16 #  32   #  64   #  96   #  128   #  160   ##
        ###############################################################

        # Feature Extraction
        self.hourglass = Hourglass([32, 64, 96, 128, 160])

        # Cascaded Parallax-Attention Module
        self.cas_pam = CascadedPAM([128, 96, 64])

        # Output Module
        self.output = Output()
        self.output_c3 = Output_cost3()

        # Disparity Refinement
        self.refine = Refinement([64, 96, 128, 160, 160, 128, 96, 64, 32, 16])   
        self.refine_fusion = RefinementFusion()

    def forward(self, x_left, x_right, max_disp=0):
        b, _, h, w = x_left.shape

        # Feature Extraction
        (fea_left_s1, fea_left_s2, fea_left_s3), fea_refine, refine_vit = self.hourglass(x_left)
        (fea_right_s1, fea_right_s2, fea_right_s3), _, _       = self.hourglass(x_right)

        # Cascaded Parallax-Attention Module
        cost_s1, cost_s2, cost_s3, cost_s3_fea = self.cas_pam([fea_left_s1, fea_left_s2, fea_left_s3],
                                                 [fea_right_s1, fea_right_s2, fea_right_s3])

        # Output Module
        if self.training:
            disp_s1, att_s1, att_cycle_s1, valid_mask_s1 = self.output(cost_s1, max_disp // 16)
            disp_s2, att_s2, att_cycle_s2, valid_mask_s2 = self.output(cost_s2, max_disp // 8)
            disp_s3, att_s3, att_cycle_s3, valid_mask_s3 = self.output_c3(cost_s3, max_disp // 4, cost_s3_fea)
        else:
            #disp_s3 = self.output_c3(cost_s3, max_disp // 4, cost_s3_fea)
            disp_s3 = self.output_c3(cost_s3, max_disp // 4, cost_s3_fea) 
            
        #fea_refine_last = 0.5 * fea_refine + (1 - 0.5) * refine_vit
        # fea_refine_last = self.refine_fusion(torch.cat([fea_refine, refine_vit], dim=1))
        # Disparity Refinement
        disp = self.refine(fea_refine, disp_s3)

        if self.training:
            return disp, \
                   [att_s1, att_s2, att_s3], \
                   [att_cycle_s1, att_cycle_s2, att_cycle_s3], \
                   [valid_mask_s1, valid_mask_s2, valid_mask_s3]
        else:
            return disp


# Hourglass Module for Feature Extraction
class Hourglass(nn.Module):
    def __init__(self, channels):
        super(Hourglass, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.E0 = EncoderB(1,           3, channels[0], downsample=True)               # scale: 1/2
        self.E1 = EncoderB(1, channels[0], channels[1], downsample=True)               # scale: 1/4
        self.E2 = EncoderB(1, channels[1], channels[2], downsample=True)               # scale: 1/8
        self.E3 = EncoderB(1, channels[2], channels[3], downsample=True)               # scale: 1/16
        self.E4 = EncoderB(1, channels[3], channels[4], downsample=True)               # scale: 1/32

        self.D0 = EncoderB(1, channels[4], channels[4], downsample=False)              # scale: 1/32
        self.D1 = DecoderB(1, channels[4] + channels[3], channels[3])                  # scale: 1/16
        self.D2 = DecoderB(1, channels[3] + channels[2], channels[2])                  # scale: 1/8
        self.D3 = DecoderB(1, channels[2] + channels[1], channels[1])                  # scale: 1/4
        
        self.att_E3 = ShuffleAttention(channels[3],G=8)
        self.att_E4 = ShuffleAttention(channels[4],G=16)
        self.att_D1 = ShuffleAttention(channels[3],G=8)
        self.att_D2 = ShuffleAttention(channels[2],G=8)
        
        self.vit = DeepViTWithClass(image_size = 512, patch_size = 32, num_classes = 1000, dim = 512, depth = 6
                         , heads = 16, mlp_dim = 1024, dropout = 0.1, emb_dropout = 0.1)
        
        self.vit_channel_r = ChannelReductionNet()
        self.fea_vit = ExtendedConv(channels[4])
        self.fusion_fc = nn.Conv2d(channels[4] * 2, channels[4], kernel_size=1)
        
        self.fusion_refine = FeatureFusion()
        self.vitcnn_channel_reduce = HourglassFusion()

    def forward(self, x):
        fea_E0 = self.E0(x)                                                            # scale: 1/2
        fea_E1 = self.E1(fea_E0)                                                       # scale: 1/4
        fea_E2 = self.E2(fea_E1)                                                       # scale: 1/8
        fea_E3 = self.E3(fea_E2)                                                       # scale: 1/16
        fea_E3 = self.att_E3(fea_E3)                                                   # scale: 1/16
        fea_E4 = self.E4(fea_E3)                                                       # scale: 1/16
        fea_E4 = self.att_E4(fea_E4)                                                   # scale: 1/16
        
        #vit_feature
        x_vit = nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        fea_vit = self.vit(x_vit)
        fea_vit = self.vit_channel_r(fea_vit)
        fea_vit = self.fea_vit(fea_vit)  
        fea_vit = nn.functional.interpolate(fea_vit, size=(x.shape[2]//32, x.shape[3]//32), mode='bilinear', align_corners=False)
        
        refine_vit = self.fusion_refine(fea_vit)
        
        fea_D0 = self.D0(fea_E4)                                                       # scale: 1/32
        
        '''
        fusion_weight = torch.sigmoid(self.fusion_fc(torch.cat([fea_D0.mean(dim=(2,3), keepdim=True), 
                                                         fea_vit.mean(dim=(2,3), keepdim=True)], dim=1)))
        fea_D0 = fusion_weight * 0.5 * fea_D0 + (1 - 0.5* fusion_weight) * fea_vit 
        ''' 
        
        vitcnn_fusion = torch.cat([fea_D0, fea_vit], dim=1)  
        fea_D0 = self.vitcnn_channel_reduce(vitcnn_fusion)
        
        # fea_D0 = 0.1 * fea_D0 + 0.9 * fea_vit 
        fea_D1 = self.D1(torch.cat((self.upsample(fea_D0), fea_E3), 1))                # scale: 1/16
        fea_D1 = self.att_D1(fea_D1)                                                   # scale: 1/16
        fea_D2 = self.D2(torch.cat((self.upsample(fea_D1), fea_E2), 1))                # scale: 1/8
        fea_D2 = self.att_D2(fea_D2)                                                   # scale: 1/8
        fea_D3 = self.D3(torch.cat((self.upsample(fea_D2), fea_E1), 1))                # scale: 1/4
        


        return (fea_D1, fea_D2, fea_D3), fea_E1, refine_vit


# Cascaded Parallax-Attention Module
class CascadedPAM(nn.Module):
    def __init__(self, channels):
        super(CascadedPAM, self).__init__()
        self.stage1 = PAM_stage(channels[0])
        self.stage2 = PAM_stage(channels[1])
        self.stage3 = PAM_stage(channels[2])

        # bottleneck in stage 2
        self.b2 = nn.Sequential(
            nn.Conv2d(128 + 96, 96, 1, 1, 0, bias=True),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # bottleneck in stage 3
        self.b3 = nn.Sequential(
            nn.Conv2d(96 + 64, 64, 1, 1, 0, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # bilateral_filter
        # self.bilateral_filter = BilateralFilter()
        self.bilfea = BilateralNet()

    def forward(self, fea_left, fea_right):
        '''
        :param fea_left:    feature list [fea_left_s1, fea_left_s2, fea_left_s3]
        :param fea_right:   feature list [fea_right_s1, fea_right_s2, fea_right_s3]
        '''
        fea_left_s1, fea_left_s2, fea_left_s3 = fea_left
        fea_right_s1, fea_right_s2, fea_right_s3 = fea_right

        b, _, h_s1, w_s1 = fea_left_s1.shape
        b, _, h_s2, w_s2 = fea_left_s2.shape

        # stage 1: 1/16
        cost_s0 = [
            torch.zeros(b, h_s1, w_s1, w_s1).to(fea_right_s1.device),
            torch.zeros(b, h_s1, w_s1, w_s1).to(fea_right_s1.device)
        ]

        fea_left, fea_right, cost_s1 = self.stage1(fea_left_s1, fea_right_s1, cost_s0)       

        # stage 2: 1/8
        fea_left = F.interpolate(fea_left, scale_factor=2, mode='bilinear')
        fea_right = F.interpolate(fea_right, scale_factor=2, mode='bilinear')
        fea_left = self.b2(torch.cat((fea_left, fea_left_s2), 1))
        fea_right = self.b2(torch.cat((fea_right, fea_right_s2), 1))

        cost_s1_up = [
            F.interpolate(cost_s1[0].view(b, 1, h_s1, w_s1, w_s1), scale_factor=2, mode='trilinear').squeeze(1),
            F.interpolate(cost_s1[1].view(b, 1, h_s1, w_s1, w_s1), scale_factor=2, mode='trilinear').squeeze(1)
        ]

        fea_left, fea_right, cost_s2 = self.stage2(fea_left, fea_right, cost_s1_up)

        # stage 3: 1/4
        fea_left = F.interpolate(fea_left, scale_factor=2, mode='bilinear')
        fea_right = F.interpolate(fea_right, scale_factor=2, mode='bilinear')
        fea_left = self.b3(torch.cat((fea_left, fea_left_s3), 1))
        fea_right = self.b3(torch.cat((fea_right, fea_right_s3), 1))
        
        cost_s2_up = [
            F.interpolate(cost_s2[0].view(b, 1, h_s2, w_s2, w_s2), scale_factor=2, mode='trilinear').squeeze(1),
            F.interpolate(cost_s2[1].view(b, 1, h_s2, w_s2, w_s2), scale_factor=2, mode='trilinear').squeeze(1)
        ]

        fea_left, fea_right, cost_s3 = self.stage3(fea_left, fea_right, cost_s2_up)
        channel_bil = cost_s3[0].shape[1]
        cost3_bil_fea = self.bilfea(cost_s3[0], channel_bil)
        # print(channel_bil)
        # cost_s3_bil = [self.bilateral_filter(cost_s3[0], cost3_bil_fea), self.bilateral_filter(cost_s3[1], cost3_bil_fea)]
        #torch.Size([20, 64, 128, 128])torch.Size([20, 16, 32, 32])torch.Size([20, 32, 64, 64])

        return [cost_s1, cost_s2, cost_s3, cost3_bil_fea]


class PAM_stage(nn.Module):
    def __init__(self, channels):
        super(PAM_stage, self).__init__()
        self.pab1 = PAB(channels)
        self.pab2 = PAB(channels)
        self.pab3 = PAB(channels)
        self.pab4 = PAB(channels)

    def forward(self, fea_left, fea_right, cost):
        fea_left, fea_right, cost = self.pab1(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab2(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab3(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab4(fea_left, fea_right, cost)

        return fea_left, fea_right, cost


# Disparity Refinement Module
class Refinement(nn.Module):
    def __init__(self, channels):
        super(Refinement, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.AvgPool2d(2)

        self.E0 = EncoderB(1, channels[0] + 1, channels[0], downsample=False)   # scale: 1/4
        self.E1 = EncoderB(1, channels[0],     channels[1], downsample=True)    # scale: 1/8
        self.E2 = EncoderB(1, channels[1],     channels[2], downsample=True)    # scale: 1/16
        self.E3 = EncoderB(1, channels[2],     channels[3], downsample=True)    # scale: 1/32

        self.D0 = EncoderB(1, channels[4],     channels[4], downsample=False)   # scale: 1/32
        self.D1 = DecoderB(1, channels[4] + channels[5], channels[5])           # scale: 1/16
        self.D2 = DecoderB(1, channels[5] + channels[6], channels[6])           # scale: 1/8
        self.D3 = DecoderB(1, channels[6] + channels[7], channels[7])           # scale: 1/4
        self.D4 = DecoderB(1, channels[7],               channels[8])           # scale: 1/2
        self.D5 = DecoderB(1, channels[8],               channels[9])           # scale: 1

        # regression
        self.confidence = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.disp = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False)
        )

    def forward(self, fea, disp):
        # scale the input disparity
        disp = disp / (2 ** 5)
        fea_E0 = self.E0(torch.cat((disp, fea), 1))                         # scale: 1/4
        fea_E1 = self.E1(fea_E0)                                            # scale: 1/8
        fea_E2 = self.E2(fea_E1)                                            # scale: 1/16
        fea_E3 = self.E3(fea_E2)                                            # scale: 1/32

        fea_D0 = self.D0(fea_E3)                                            # scale: 1/32
        fea_D1 = self.D1(torch.cat((self.upsample(fea_D0), fea_E2), 1))     # scale: 1/16
        fea_D2 = self.D2(torch.cat((self.upsample(fea_D1), fea_E1), 1))     # scale: 1/8
        fea_D3 = self.D3(torch.cat((self.upsample(fea_D2), fea_E0), 1))     # scale: 1/4
        fea_D4 = self.D4(self.upsample(fea_D3))                             # scale: 1/2
        fea_D5 = self.D5(self.upsample(fea_D4))                             # scale: 1

        # regression
        confidence = self.confidence(fea_D5)
        disp_res = self.disp(fea_D5)
        disp_res = torch.clamp(disp_res, 0)

        disp = F.interpolate(disp, scale_factor=4, mode='bilinear') * (1-confidence) + disp_res * confidence

        # scale the output disparity
        # note that, the size of output disparity is 4 times larger than the input disparity
        return disp * 2 ** 7

    
class DeepViTWithClass(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout, emb_dropout):
        super(DeepViTWithClass, self).__init__()
        # 使用 DeepViT 的构造方式
        self.transformer = DeepViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        
        self.H_new = image_size // patch_size
        self.W_new = image_size // patch_size
        self.channel = dim
        
    def forward(self, x):
        features = self.transformer.to_patch_embedding(x)  # (B, N, C)
        features_tr = self.transformer.transformer(features) 
        feature_map = features_tr.view(-1, self.channel, self.H_new, self.W_new)  # (B, C, H', W')
        return feature_map

    
class ChannelReductionNet(nn.Module):
    def __init__(self, num_layers=3):
        super(ChannelReductionNet, self).__init__()
        
        # 设置通道数的每层降维比例
        channels = [512, 320, 224, 160]
        
        # 定义每层的卷积、激活函数、批归一化
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1),  # 3x3卷积
                    nn.BatchNorm2d(channels[i + 1]),  # 批归一化
                    nn.ReLU(inplace=True)  # ReLU激活函数
                )
            )
            
    def forward(self, x):
        # 顺序通过每一层
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class ExtendedConv(nn.Module):
    def __init__(self, channels, num_layers=3):
        super(ExtendedConv, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
        self.fea_vit = nn.Sequential(*layers)

    def forward(self, x):
        return self.fea_vit(x)
    
    
class BilateralFilter(nn.Module):
    def __init__(self):
        super(BilateralFilter, self).__init__()


    def _gaussian_kernel(self, size, sigma):
        """ 计算空间高斯核 """
        coords = torch.arange(size) - size // 2
        grid = coords.repeat(size, 1) ** 2 + coords.repeat(size, 1).T ** 2
        kernel = torch.exp(-grid / (2 * sigma ** 2))
        return kernel / kernel.sum()

    def forward(self, x, params):
        """
        x: (B, C, H, W) - 输入张量
        params: (B, 2) - 第一列是kernel_size，第二列是sigma_space
        输出: 经过双边滤波后的张量
        """
        B, C, H, W = x.shape
        kernel_size = params[:, 0].long()  # 获取每个样本的 kernel_size
        sigma_space = params[:, 1]  # 获取每个样本的 sigma_space

        # 准备一个输出列表
        filtered_list = []

        for i in range(B):
            k_size = kernel_size[i].item()  # 当前样本的 kernel_size
            sigma = sigma_space[i].item()  # 当前样本的 sigma_space
            k_size = 5 + 5 * (1 / (1 + math.exp(-k_size)))  # kernel_size 缩放到 [5, 10]
            k_size = int(k_size)
            sigma = 0.5 + 4.5 * (1 / (1 + math.exp(-sigma)))  # sigma_space 缩放到 [0.5, 5.0]

            # 计算高斯核
            kernel = self._gaussian_kernel(k_size, sigma)

            # 计算高斯核，并扩展成每个通道共享的权重
            kernel = kernel.expand(C, 1, k_size, k_size)  # [C, 1, k_size, k_size]

            # 使用groups=C表示每个通道共享同一个卷积核
            filtered = F.conv2d(x[i:i+1], kernel.to(x.device), padding=k_size // 2, groups=C)
            filtered = F.interpolate(filtered, size=(H, W), mode='bilinear', align_corners=False)
            
            filtered_list.append(filtered)

        # 拼接所有批次的输出
        filtered_output = torch.cat(filtered_list, dim=0)

        return filtered_output
    
class BilateralNet(nn.Module):
    def __init__(self):
        super(BilateralNet, self).__init__()
        
        # 卷积层，逐步减少通道数和空间尺寸
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)  # 降低空间尺寸
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)   # 降低空间尺寸
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)   # 降低空间尺寸
        self.conv5 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)    # 降低空间尺寸

        # 批处理归一化层
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(8)

        # 全连接层，用于输出 kernel_size 和 sigma
        self.fc2 = nn.Linear(1024, 3)  # 输出两个值：kernel_size 和 sigma1 sigma2

    def forward(self, x, channel):
        # 卷积层依次进行特征提取
        conv1 = nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1).to('cuda:0')
        x = F.relu(self.bn1(conv1(x)))  # [B, 128, 128, 128]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 64, 64, 64]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 32, 32, 32]
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 16, 16, 16]
        x = F.relu(self.bn5(self.conv5(x)))  # [B, 8, 8, 8]
        h, w = x.shape[2], x.shape[3]
        
        # 将输出拉平成一个一维张量
        x = x.view(x.size(0), -1)  # 拉平成 (B, C*H*W)

        # 通过全连接层
        fc1 = nn.Linear(h * w * 8, 1024).to('cuda:0')  # 计算输出特征的维度，假设空间尺寸为 4x4
        x = F.relu(fc1(x))
        out = self.fc2(x)

        # 输出的形状为 (B, 2)，对应每个样本的 kernel_size 和 sigma
        # 需要进行调整：

        return out 
    
    
class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()

        # 使用 ConvTranspose2d（转置卷积）进行上采样
        self.up_block1 = nn.Sequential(
            nn.ConvTranspose2d(160, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.up_block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True)
        )

        self.up_block3 = nn.Sequential(
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )


        # 调整最后的高度，使得尺寸为 [B, 64, 128, 256]
        self.adjust_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # 最终的输出卷积层
        self.conv_1_0 = nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0)
        self.conv_1_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 输入形状 [B, 160, 16, 16]

        # 第一个上采样块
        x = self.up_block1(x)  # [B, 128, 32, 32]

        # 第二个上采样块
        x = self.up_block2(x)  # [B, 96, 64, 64]      
        x = self.conv_1_0(x)

        # 第三个上采样块
        x = self.up_block3(x)  # [B, 64, 128, 128]

        # 调整高度到 128
        x = self.adjust_conv(x)  # [B, 64, 128, 256]

        # 最终卷积
        x = self.conv_1_1(x)  # [B, 64, 128, 256]

        return x
    
class HourglassFusion(nn.Module):
    def __init__(self):
        super(HourglassFusion, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(320, 288, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(288),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(288, 224, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(224, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(192, 160, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True)
        )

        # 可选的卷积融合层
        self.mid_conv = nn.Conv2d(288, 288, kernel_size=1, stride=1, padding=0)
        self.final_conv = nn.Conv2d(160, 160, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 输入 x: [B, 160, H, W] （如 H=W=16）
        x = self.conv_block1(x)  
        x = self.mid_conv(x)
        x = self.conv_block2(x)  
        x = self.conv_block3(x)  
        x = self.conv_block4(x)  
        x = self.final_conv(x)   

        return x 
class RefinementFusion(nn.Module):
    def __init__(self):
        super(RefinementFusion, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(96, 80, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(80, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )


        # 可选的卷积融合层
        self.final_conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 输入 x: [B, 160, H, W] （如 H=W=16）
        x = self.conv_block1(x)  
        x = self.conv_block2(x)  
        x = self.conv_block3(x)  
        x = self.final_conv(x)   

        return x 
    