# -*- coding: utf-8 -*-


################################################
########        DESIGN   NETWORK        ########
################################################

import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import numpy as np

import torch
import torch.nn as nn
import numpy as np

class RandomNoiseGenerator(nn.Module):
    def __init__(self, noise_types=['gaussian', 'salt_and_pepper', 'poisson', 'speckle'], noise_params={}):
        super(RandomNoiseGenerator, self).__init__()
        self.noise_types = noise_types
        self.noise_params = noise_params

    def forward(self, feature):
        feature_with_noise = feature.clone()  # Make a copy of the original feature

        # Randomly choose a noise type
        selected_noise_type = np.random.choice(self.noise_types)

        if selected_noise_type == 'gaussian':
            mean = self.noise_params.get('gaussian', {}).get('mean', 0)
            std = self.noise_params.get('gaussian', {}).get('std', 0.1)
            noise = torch.randn_like(feature) * std + mean
        elif selected_noise_type == 'salt_and_pepper':
            amount = self.noise_params.get('salt_and_pepper', {}).get('amount', 0.05)
            salt_vs_pepper = self.noise_params.get('salt_and_pepper', {}).get('salt_vs_pepper', 0.5)
            salt_mask = torch.rand_like(feature) < amount
            pepper_mask = torch.rand_like(feature) < amount
            noise = torch.zeros_like(feature)
            noise[salt_mask] = 1
            noise[pepper_mask] = 0
        elif selected_noise_type == 'poisson':
            noise = torch.poisson(feature)
        elif selected_noise_type == 'speckle':
            std = self.noise_params.get('speckle', {}).get('std', 0.1)
            noise = torch.randn_like(feature) * std
            noise = noise * feature
        else:
            raise ValueError("Unsupported noise type: {}".format(selected_noise_type))

        feature_with_noise += noise

        return feature_with_noise



class SRAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(SRAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=3)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        # print('x.size',x.size)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height//4).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height//4)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height//4)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x


        return out
    
class deConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(deConv2, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class EncoderConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(EncoderConv2, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3,1,1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True), ) 
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3,1,1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_size, out_size, 1,1,1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True), ) 
            
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3,1,1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_size, out_size, 3,1,1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_size, out_size, 1,1,1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True), ) 
            
            self.conv4 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=1),
                nn.ReLU(inplace=True), )
            
            self.layer_norm = nn.LayerNorm(out_size)
                        
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3,1,1),
                nn.ReLU(inplace=True), ) 
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3,1,1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_size, out_size, 1,1,1),
                nn.ReLU(inplace=True), ) 
            
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3,1,1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_size, out_size, 3,1,1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_size, out_size, 1,1,1),
                nn.ReLU(inplace=True), ) 
            
            self.conv4 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=1),
                nn.ReLU(inplace=True), ) 
            
            self.layer_norm = nn.LayerNorm(out_size)

            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True), )
            self.conv3 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True), )                           

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        # Concatenate x1, x2, and x3
        concatenated = torch.cat((x1, x2, x3), dim=1)
        
        # Apply layer normalization
        normalized = self.layer_norm(concatenated)
        
        # Add normalized to x1
        added = normalized + x1
        
        # Pass through the final convolutional layer
        outputs = self.conv4(added)
        
        return outputs


class Encoder(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(Encoder, self).__init__()
        self.conv = EncoderConv2(in_size, out_size, is_batchnorm)
        self.en = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.en(outputs)
        return outputs


class Decoder(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(Decoder, self).__init__()
        self.conv = deConv2(in_size, out_size, True)
        # Transposed convolution
        if is_deconv:
            self.de = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.de = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.de(inputs2)
        offset1 = (outputs2.size()[2] - inputs1.size()[2])
        offset2 = (outputs2.size()[3] - inputs1.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]
        # Skip and concatenate
        outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], 1))



class SRTransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super(SRTransformerBlock, self).__init__()
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = SRAttention(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        
        self.conv1x1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        
    def forward(self, x):
        # Layer normalization
        x_norm = self.layer_norm1(x)
        
        # Self-attention using SR-Attention
        att_output = self.attention(x_norm)
        
        # Residual connection and layer normalization
        x_a = x + att_output
        x_a_norm = self.layer_norm2(x_a)
        
        # MLP
        mlp_output = self.ffn(x_a_norm)
        
        # Residual connection and layer normalization
        x_m = x_a + mlp_output
        x_m_norm = self.layer_norm3(x_m)
        
        # Apply 1x1 convolution
        output = self.conv1x1(x_m_norm.permute(0, 2, 1))
        output = output.permute(0, 2, 1)
        
        return output


class MFDStudent(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        super(MFDStudent, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes
        self.noise=RandomNoiseGenerator()

        filters = [64, 128, 256, 512, 1024]

        self.en1 = Encoder(self.in_channels, filters[0], self.is_batchnorm)
        self.en2 = Encoder(filters[0], filters[1], self.is_batchnorm)
        self.en3 = Encoder(filters[1], filters[2], self.is_batchnorm)
        self.en4 = Encoder(filters[2], filters[3], self.is_batchnorm)
        self.center = deConv2(filters[3], filters[4], self.is_batchnorm)
        self.de4 = Decoder(filters[4], filters[3], self.is_deconv)
        self.de3 = Decoder(filters[3], filters[2], self.is_deconv)
        self.de2 = Decoder(filters[2], filters[1], self.is_deconv)
        self.de1 = Decoder(filters[1], filters[0], self.is_deconv)
        self.srtfb1 = SRTransformerBlock(filters[1])
        self.srtfb2 = SRTransformerBlock(filters[2])
        self.srtfb3 = SRTransformerBlock(filters[3])
        self.final = nn.Conv2d(filters[0], self.n_classes, 1)

    def forward(self, inputs, label_dsp_dim):
        en1=self.noise(inputs)
        en1 = self.en1(en1)
        en2 = self.en2(en1)
        en3 = self.en3(en2)
        en4 = self.en4(en3)
        center = self.center(en4)
        de4 = self.de4(en4, center)
        de4 = self.srtfb3(de4)
        de3 = self.de3(en3, de4)
        de3 = self.srtfb2(de3)
        de2 = self.de2(en2, de3)
        de2 = self.srtfb1(de2)
        de1 = self.de1(en1, de2)
        de1 = de1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()

        return self.final(de1)

    # Initialization of Parameters
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

