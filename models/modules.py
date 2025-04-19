import torch
import math
import torch.nn as nn
from skimage import morphology
from utils import *
# Residual Block
class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self,x):
        out = self.body(x)
        return self.lrelu(out + x)


# Encoder Block
class EncoderB(nn.Module):
    def __init__(self, n_blocks, channels_in, channels_out, downsample=False):
        super(EncoderB, self).__init__()
        body = []
        if downsample:
            body.append(nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, 2, 1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.LeakyReLU(0.1, inplace=True),
            ))
        if not downsample:
            body.append(nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.LeakyReLU(0.1, inplace=True),
            ))
        for i in range(n_blocks):
            body.append(
                ResB(channels_out)
            )
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


# Decoder Block
class DecoderB(nn.Module):
    def __init__(self, n_blocks, channels_in, channels_out):
        super(DecoderB, self).__init__()
        body = []
        body.append(nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.LeakyReLU(0.1, inplace=True),
            ))
        for i in range(n_blocks):
            body.append(
                ResB(channels_out)
            )
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


# Parallax-Attention Block
class PAB(nn.Module):
    def __init__(self, channels):
        super(PAB, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.query = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels),
        )
        self.key = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x_left, x_right, cost):
        '''
        :param x_left:      features from the left image  (B * C * H * W)
        :param x_right:     features from the right image (B * C * H * W)
        :param cost:        input matching cost           (B * H * W * W)
        '''
        b, c, h, w = x_left.shape
        fea_left = self.head(x_left)
        fea_right = self.head(x_right)

        # C_right2left
        Q = self.query(fea_left).permute(0, 2, 3, 1).contiguous()                     # B * H * W * C
        K = self.key(fea_right).permute(0, 2, 1, 3) .contiguous()                     # B * H * C * W
        cost_right2left = torch.matmul(Q, K) / c                                      # scale the matching cost
        cost_right2left = cost_right2left + cost[0]

        # C_left2right
        Q = self.query(fea_right).permute(0, 2, 3, 1).contiguous()                    # B * H * W * C
        K = self.key(fea_left).permute(0, 2, 1, 3).contiguous()                       # B * H * C * W
        cost_left2right = torch.matmul(Q, K) / c                                      # scale the matching cost
        cost_left2right = cost_left2right + cost[1]

        return x_left + fea_left, \
               x_right + fea_right, \
               (cost_right2left, cost_left2right)


# Output Module
class Output(nn.Module):
    def __init__(self):
        super(Output, self).__init__()

    def forward(self, cost, max_disp):
        cost_right2left, cost_left2right = cost
        b, h, w, _ = cost_right2left.shape

        # M_right2left
        ## exclude negative disparities & disparities larger than max_disp (if available)
        cost_right2left = torch.tril(cost_right2left)
        if max_disp > 0:
            cost_right2left = cost_right2left - torch.tril(cost_right2left, -max_disp)
        cost_right2left = torch.exp(cost_right2left - cost_right2left.max(-1)[0].unsqueeze(-1))
        cost_right2left = torch.tril(cost_right2left)
        if max_disp > 0:
            cost_right2left = cost_right2left - torch.tril(cost_right2left, -max_disp)
        att_right2left = cost_right2left / (cost_right2left.sum(-1, keepdim=True) + 1e-8)

        # M_left2right
        ## exclude negative disparities & disparities larger than max_disp (if available)
        cost_left2right = torch.triu(cost_left2right)
        if max_disp > 0:
            cost_left2right = cost_left2right - torch.triu(cost_left2right, max_disp)
        cost_left2right = torch.exp(cost_left2right - cost_left2right.max(-1)[0].unsqueeze(-1))
        cost_left2right = torch.triu(cost_left2right)
        if max_disp > 0:
            cost_left2right = cost_left2right - torch.triu(cost_left2right, max_disp)
        att_left2right = cost_left2right / (cost_left2right.sum(-1, keepdim=True) + 1e-8)

        # valid mask (left image)
        valid_mask_left = torch.sum(att_left2right.detach(), -2) > 0.1
        valid_mask_left = valid_mask_left.view(b, 1, h, w)
        valid_mask_left = morphologic_process(valid_mask_left)

        # disparity
        disp = regress_disp(att_right2left, valid_mask_left)

        if self.training:
            # valid mask (right image)
            valid_mask_right = torch.sum(att_right2left.detach(), -2) > 0.1
            valid_mask_right = valid_mask_right.view(b, 1, h, w)
            valid_mask_right = morphologic_process(valid_mask_right)

            # cycle-attention maps
            att_left2right2left = torch.matmul(att_right2left, att_left2right).view(b, h, w, w)
            att_right2left2right = torch.matmul(att_left2right, att_right2left).view(b, h, w, w)

            return disp, \
                   (att_right2left.view(b, h, w, w), att_left2right.view(b, h, w, w)), \
                   (att_left2right2left, att_right2left2right), \
                   (valid_mask_left, valid_mask_right)
        else:
            return disp
        
# Output Module
class Output_cost3(nn.Module):
    def __init__(self):
        super(Output_cost3, self).__init__()
        self.bilateral_filter = BilateralFilter()

    def forward(self, cost, max_disp, cost_s3_fea):
        cost_right2left, cost_left2right = cost
        b, h, w, _ = cost_right2left.shape

        # M_right2left
        ## exclude negative disparities & disparities larger than max_disp (if available)
        cost_right2left = torch.tril(cost_right2left)
        if max_disp > 0:
            cost_right2left = cost_right2left - torch.tril(cost_right2left, -max_disp)
        cost_right2left = torch.exp(cost_right2left - cost_right2left.max(-1)[0].unsqueeze(-1))
        cost_right2left = torch.tril(cost_right2left)
        if max_disp > 0:
            cost_right2left = cost_right2left - torch.tril(cost_right2left, -max_disp)
        att_right2left = cost_right2left / (cost_right2left.sum(-1, keepdim=True) + 1e-8)
        att_right2left = self.bilateral_filter(att_right2left, cost_s3_fea)

        # M_left2right
        ## exclude negative disparities & disparities larger than max_disp (if available)
        cost_left2right = torch.triu(cost_left2right)
        if max_disp > 0:
            cost_left2right = cost_left2right - torch.triu(cost_left2right, max_disp)
        cost_left2right = torch.exp(cost_left2right - cost_left2right.max(-1)[0].unsqueeze(-1))
        cost_left2right = torch.triu(cost_left2right)
        if max_disp > 0:
            cost_left2right = cost_left2right - torch.triu(cost_left2right, max_disp)
        att_left2right = cost_left2right / (cost_left2right.sum(-1, keepdim=True) + 1e-8)
        att_left2right = self.bilateral_filter(att_left2right, cost_s3_fea)

        # valid mask (left image)
        valid_mask_left = torch.sum(att_left2right.detach(), -2) > 0.1
        valid_mask_left = valid_mask_left.view(b, 1, h, w)
        valid_mask_left = morphologic_process(valid_mask_left)

        # disparity
        disp = regress_disp(att_right2left, valid_mask_left)

        if self.training:
            # valid mask (right image)
            valid_mask_right = torch.sum(att_right2left.detach(), -2) > 0.1
            valid_mask_right = valid_mask_right.view(b, 1, h, w)
            valid_mask_right = morphologic_process(valid_mask_right)

            # cycle-attention maps
            att_left2right2left = torch.matmul(att_right2left, att_left2right).view(b, h, w, w)
            att_right2left2right = torch.matmul(att_left2right, att_right2left).view(b, h, w, w)

            return disp, \
                   (att_right2left.view(b, h, w, w), att_left2right.view(b, h, w, w)), \
                   (att_left2right2left, att_right2left2right), \
                   (valid_mask_left, valid_mask_right)
        else:
            return disp

# Morphological Operations
def morphologic_process(mask):
    b, _, _, _ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        mask_np[idx, 0, :, :] = morphology.binary_closing(mask_np[idx, 0, :, :], morphology.disk(3))
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(mask.device)


# Disparity Regression
def regress_disp(att, valid_mask):
    '''
    :param att:         B * H * W * W
    :param valid_mask:  B * 1 * H * W
    '''
    b, h, w, _ = att.shape
    index = torch.arange(w).view(1, 1, 1, w).to(att.device).float()    # index: 1*1*1*w
    disp_ini = index - torch.sum(att * index, dim=-1).view(b, 1, h, w)

    # partial conv
    filter1 = torch.zeros(1, 3).to(att.device)
    filter1[0, 0] = 1
    filter1[0, 1] = 1
    filter1 = filter1.view(1, 1, 1, 3)

    filter2 = torch.zeros(1, 3).to(att.device)
    filter2[0, 1] = 1
    filter2[0, 2] = 1
    filter2 = filter2.view(1, 1, 1, 3)

    valid_mask_0 = valid_mask
    disp = disp_ini * valid_mask_0

    valid_mask_num = 1
    while valid_mask_num > 0:
        valid_mask_1 = F.conv2d(valid_mask_0, filter1, padding=[0, 1])
        disp = disp * valid_mask_0 + \
               F.conv2d(disp, filter1, padding=[0, 1]) / (valid_mask_1 + 1e-4) * ((valid_mask_1 > 0).float() - valid_mask_0)
        valid_mask_num = (valid_mask_1 > 0).float().sum() - valid_mask_0.sum()
        valid_mask_0 = (valid_mask_1 > 0).float()

    valid_mask_num = 1
    while valid_mask_num > 0:
        valid_mask_1 = F.conv2d(valid_mask_0, filter2, padding=[0, 1])
        disp = disp * valid_mask_0 + \
               F.conv2d(disp, filter2, padding=[0, 1]) / (valid_mask_1 + 1e-4) * ((valid_mask_1 > 0).float() - valid_mask_0)
        valid_mask_num = (valid_mask_1 > 0).float().sum() - valid_mask_0.sum()
        valid_mask_0 = (valid_mask_1 > 0).float()

    return disp_ini * valid_mask + disp * (1 - valid_mask)


class BilateralFilter(nn.Module):
    def __init__(self):
        super(BilateralFilter, self).__init__()

        
    def _gaussian_kernel(self, size, sigma1):
        """计算空间高斯核 (size, size)"""
        coords = torch.arange(size) - size // 2
        grid = coords.repeat(size, 1) ** 2 + coords.repeat(size, 1).T ** 2
        space_kernel = torch.exp(-grid / (2 * sigma1 ** 2))

        return space_kernel / space_kernel.sum()  # 归一化

    def _color_kernel(self, cost, size, sigma2):
        """计算颜色高斯核 (size, size, H, W)，所有通道共享"""
        C, H, W = cost.shape

        # 计算灰度图像（所有通道取均值）
        cost_1channel = cost.mean(dim=0, keepdim=True)  # (1, H, W)
        # 提取 size×size 领域窗口
        unfolded = F.unfold(cost_1channel.unsqueeze(0), kernel_size=size, padding=size // 2)  # (1, size*size, H*W)
        unfolded = unfolded.view(size, size, H, W)  # (size, size, H, W)

        # 计算颜色差异（自动广播）
        color_diff = (unfolded - cost_1channel) ** 2  # (size, size, H, W)

        # 计算颜色高斯核
        color_kernel = torch.exp(-color_diff / (2 * sigma2 ** 2))

        return color_kernel / color_kernel.sum(dim=(0, 1), keepdim=True)  # 归一化


    def forward(self, x, params):
        B, C, H, W = x.shape
        kernel_size = params[:, 0].long()
        sigma_space = params[:, 1]
        sigma_color = params[:, 2]

        filtered_list = []

        for i in range(B):
            k_size = kernel_size[i].item()
            sigma1 = sigma_space[i].item()
            sigma2 = sigma_color[i].item()
            k_size = 1 + 2 * (1 / (1 + math.exp(-k_size)))  # kernel_size 缩放到 [3, 7]
            k_size = 2 * int(k_size) + 1
            sigma1 = 3.5 + 5.5 * (1 / (1 + math.exp(-sigma1)))  # sigma_space 缩放到 [3.5, 9.0]
            sigma2 = 5.5 + 7.5 * (1 / (1 + math.exp(-sigma2)))  # sigma_space 缩放到 [5.5, 13.0]

            # 计算空间高斯核（所有通道共享）
            space_kernel = self._gaussian_kernel(k_size, sigma1).to(x.device)

            # 计算颜色高斯核（所有通道共享）
            color_kernel = self._color_kernel(x[i], k_size, sigma2).to(x.device)  # (size, size, H, W)

            # 展开输入
            unfolded_x = F.unfold(x[i:i+1], kernel_size=k_size, padding=k_size // 2)  # (1, C*size*size, H*W)
            unfolded_x = unfolded_x.view(C, k_size, k_size, H, W)  # (C, size, size, H, W)

            # 计算最终权重 = 空间核 × 颜色核
            weight = space_kernel.unsqueeze(-1).unsqueeze(-1) * color_kernel  # (size, size, H, W)

            # 归一化
            weight = weight / (weight.sum(dim=(0, 1), keepdim=True) + 1e-8)

            # 进行加权求和（所有通道共享相同的权重）
            filtered = (unfolded_x * weight).sum(dim=(1, 2))  # (C, H, W)
            
            filtered_list.append(filtered.unsqueeze(0))  # (1, C, H, W)

        return torch.cat(filtered_list, dim=0)  # (B, C, H, W)


'''
class BilateralFilter(nn.Module):
    def __init__(self):
        super(BilateralFilter, self).__init__()


    def _gaussian_kernel(self, size, sigma1, sigma2):
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
'''