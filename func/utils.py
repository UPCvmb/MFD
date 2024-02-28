# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from math import log10
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import scipy.io as scio
import scipy.io
# import scipy.io as scio
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from matplotlib import ticker, cm
from pathlib import *


def turn(GT):
    dim = GT.shape
    for j in range(0, dim[1]):
        for i in range(0, dim[0] // 2):
            temp = GT[i, j]
            GT[i, j] = GT[dim[0] - 1 - i, j]
            GT[dim[0] - 1 - i, j] = temp
    return GT


def turn_true(data_turn):
    data_turn = np.flip(data_turn, axis=0)  # 上下反转
    data_turn = np.flip(data_turn, axis=1)  # 左右反转
    return data_turn


def PSNR(prediction, target):
    prediction = Variable(torch.from_numpy(prediction))
    target = Variable(torch.from_numpy(target))
    zero = torch.zeros_like(target)
    # diff = prediction - target
    # diff = diff.flatten('C')
    # rmse = math.sqrt(np.mean(diff ** 2.))
    # return 20 * math.log10(1.0 / rmse)
    criterion = nn.MSELoss(size_average=True)
    MSE = criterion(prediction, target)
    total = criterion(target, zero)
    psnr = 10. * log10(total.item() / MSE.item())
    return psnr


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    L = 255
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(img1, img2, window_size=11, size_average=True):
    img1 = Variable(torch.from_numpy(img1))
    img2 = Variable(torch.from_numpy(img2))
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def creat_dir(dir_path):
    if Path(dir_path).exists():  # 如果已经存在，则跳过并提示
        # print("文件夹已经存在！")
        pass
    else:
        Path.mkdir(Path(dir_path), parents=True, exist_ok=True)  # 创建文件夹


def save_mat(key, path, Savepath):
    # mat = scio.loadmat(path)
    mat = scipy.io.loadmat(path)
    filepath = str(path)  # .mat文件全路径+名字＋后缀
    filename = str(Path(path).stem)  # 返回.mat文件名字 无后缀
    filepath_parent = str(Path(path).parent)  # .mat的文件目录
    filepath_png = filepath_parent + "/png/"  # png保存图片目录
    creat_dir(filepath_png)  # 新建 png目录

    gt = mat[key]  # 根据key取 gt或pre

    for i in range(0, gt.shape[0]):  #
        arr = turn_true(gt[i])  # 反转 内容
        arr = arr / 1000

        fig, ax = plt.subplots()
        cmap = plt.get_cmap('turbo')
        img = ax.matshow(arr, cmap=cmap)
        ax.set_xticks(np.arange(0, 301, 100), np.arange(0, 301, 100) / 100)
        ax.set_xlabel("Distance (km)", fontsize=12)
        ax.xaxis.set_label_position('top')

        ax.set_yticks(np.arange(0, 201, 100), np.arange(0, 201, 100) / 100)
        ax.set_ylabel("Depth (km)", fontsize=12)

        # 添加colorbar
        cbar = plt.colorbar(img, ax=ax, label="Velocity(km/s)", fraction=0.1, pad=0.15, shrink=0.9,
                            anchor=(0.0, 0.3))  # 对colorbar的大小进行设置
        # 设置颜色条的刻度
        tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
        cbar.locator = tick_locator
        cbar.ax.tick_params(labelsize=12)
        # 设置颜色条的title
        cbar.ax.set_title('', fontsize=12)
        cbar.update_ticks()  # 显示colorbar的刻度值

        png_path = filepath_png
        png_name = filename + "_" + key + "_" + str(i) + ".png"
        # print("png_path + png_name = !!", png_path + png_name)

        # plt.show()
        plt.savefig(png_path + png_name)
        plt.close()
        # result_data = {"GT": arr}
        # scio.savemat("TestResults" + str(i) + "_" + key + ".mat", result_data)


def save_mat_new(key, path, Savepath):
    mat = scipy.io.loadmat(path)
    filepath = str(path)  # .mat文件全路径+名字＋后缀
    filename = str(Path(path).stem)  # 返回.mat文件名字 无后缀
    filepath_parent = str(Path(path).parent)  # .mat的文件目录
    filepath_png = filepath_parent + "/png/"  # png保存图片目录
    creat_dir(filepath_png)  # 新建 png目录

    gt = mat[key]  # 根据key取 gt或pre

    for i in range(0, gt.shape[0]):  #
        arr = turn_true(gt[i])  # 反转 内容
        arr = arr / 1000

        fig, ax = plt.subplots()

        cmap = plt.cm.get_cmap("RdYlBu").copy()  # viridis plasma inferno magma cividis
        cmap.set_under(cmap(1))
        cmap.set_over(cmap(cmap.N - 1))
        img = ax.matshow(arr, cmap=cmap, vmin=2.0, vmax=5.0)  # 通过设置最大值和最小值来限定

        ax.set_xticks(np.arange(0, 301, 100), np.arange(0, 301, 100) / 100)
        ax.set_xlabel("Distance (km)", fontsize=12)
        ax.xaxis.set_label_position('top')

        ax.set_yticks(np.arange(0, 201, 100), np.arange(0, 201, 100) / 100)
        ax.set_ylabel("Depth (km)", fontsize=12)

        # 添加colorbar
        cbar = plt.colorbar(img, ax=ax, label="Velocity(km/s)", fraction=0.1, pad=0.15, shrink=0.9,
                            anchor=(0.0, 0.3), extend='both')  # 对colorbar的大小进行设置 extend参数两遍的三角头

        # 设置颜色条的刻度
        cbar.set_ticks([2.5, 3.0, 3.5, 4.0, 4.5])  # 自定义刻度
        # cbar.locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
        cbar.ax.tick_params(labelsize=12)

        # 设置颜色条的title
        cbar.ax.set_title('', fontsize=12)
        cbar.update_ticks()  # 显示colorbar的刻度值

        png_path = filepath_png + filename + "/" + key + "/"
        creat_dir(png_path)  # 新建 png_path 目录
        png_name = filename + "_" + key + "_" + str(i) + ".png"
        # print("png_path + png_name = !!", png_path + png_name)

        # plt.show()
        plt.savefig(png_path + png_name)
        plt.close()
        # result_data = {"GT": arr}
        # scio.savemat("TestResults" + str(i) + "_" + key + ".mat", result_data)

def save_mat_grey(key, path, Savepath):
    mat = scipy.io.loadmat(path)
    filepath = str(path)  # .mat文件全路径+名字＋后缀
    filename = str(Path(path).stem)  # 返回.mat文件名字 无后缀
    filepath_parent = str(Path(path).parent)  # .mat的文件目录
    filepath_png = filepath_parent + "/png/"  # png保存图片目录
    creat_dir(filepath_png)  # 新建 png目录
    gt = mat[key]  # 根据key取 gt或pre
    for i in range(0, gt.shape[0]):  #
        arr = turn_true(gt[i])  # 反转 内容
        arr = arr / 1000
        fig, ax = plt.subplots()
        cmap = plt.cm.get_cmap("Greys").copy()  # viridis plasma inferno magma cividis
        cmap.set_under(cmap(1))
        cmap.set_over(cmap(cmap.N - 1))
        img = ax.matshow(arr, cmap=cmap, vmin=2.0, vmax=5.0)  # 通过设置最大值和最小值来限定
        ax.set_xticks(np.arange(0, 301, 100), np.arange(0, 301, 100) / 100)
        ax.set_xlabel("Distance (km)", fontsize=12)
        ax.xaxis.set_label_position('top')
        ax.set_yticks(np.arange(0, 201, 100), np.arange(0, 201, 100) / 100)
        ax.set_ylabel("Depth (km)", fontsize=12)
        cbar = plt.colorbar(img, ax=ax, label="Velocity(km/s)", fraction=0.1, pad=0.15, shrink=0.9,
                            anchor=(0.0, 0.3), extend='both')  # 对colorbar的大小进行设置 extend参数两遍的三角头
        cbar.set_ticks([2.5, 3.0, 3.5, 4.0, 4.5])  # 自定义刻度
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_title('', fontsize=12)
        cbar.update_ticks()  # 显示colorbar的刻度值
        png_path = filepath_png + filename + "/" + key + "/"
        creat_dir(png_path)  # 新建 png_path 目录
        png_name = filename + "_" + key + "_" + str(i) + ".png"
        plt.savefig(png_path + png_name)
        plt.close()


def SaveTrainResults(loss, SavePath, font2, font3, name):
    fig, ax = plt.subplots()
    plt.plot(loss[1:], linewidth=2)
    ax.set_xlabel('Num. of epochs', font2)
    ax.set_ylabel('MSE Loss', font2)
    ax.set_title('Training', font3)
    ax.set_xlim([1, 6])
    ax.set_xticklabels(('0', '20', '40', '60', '80', '100'))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)
    # if not os.path.exists(SavePath+time_now):
    #     os.makedirs(SavePath+time_now)

    plt.savefig(SavePath + '/' + name + 'Train_Loss', transparent=True)
    data = {}
    data['loss'] = loss
    scipy.io.savemat(SavePath + '/' + name + 'Train_Loss.mat', data)
    # plt.show(fig)
    plt.close()


def SaveTestResults(TotPSNR, TotSSIM, Prediction, GT, SavePath, name):
    data = {}
    data['TotPSNR'] = TotPSNR
    data['TotSSIM'] = TotSSIM
    data['GT'] = GT
    data['Prediction'] = Prediction
    # if not os.path.exists(SavePath+time_now):
    #     os.makedirs(SavePath+time_now)
    scipy.io.savemat(SavePath + '/' + name + 'TestResults.mat', data)

    save_mat_new("GT", SavePath + '/' + name + 'TestResults.mat', SavePath)  # 'GT', 'Prediction'
    save_mat_new("Prediction", SavePath + '/' + name + 'TestResults.mat', SavePath)

def SaveTestResults_new(TotPSNR, TotSSIM, Prediction, GT, SavePath, name):
    data = {}
    data['TotPSNR'] = TotPSNR
    data['TotSSIM'] = TotSSIM
    data['GT'] = GT
    data['Prediction'] = Prediction
    # if not os.path.exists(SavePath+time_now):
    #     os.makedirs(SavePath+time_now)
    scipy.io.savemat(SavePath + '/' + name, data)

    save_mat_new("GT", SavePath + '/' + name, SavePath)  # 'GT', 'Prediction'
    save_mat_new("Prediction", SavePath + '/' + name, SavePath)

def SaveTestResults_grey(TotPSNR, TotSSIM, Prediction, GT, SavePath, name):
    data = {}
    data['TotPSNR'] = TotPSNR
    data['TotSSIM'] = TotSSIM
    data['GT'] = GT
    data['Prediction'] = Prediction
    # if not os.path.exists(SavePath+time_now):
    #     os.makedirs(SavePath+time_now)
    scipy.io.savemat(SavePath + '/' + name, data)

    save_mat_grey("GT", SavePath + '/' + name, SavePath)  # 'GT', 'Prediction'
    save_mat_grey("Prediction", SavePath + '/' + name, SavePath)


def PlotComparison(pd, gt, label_dsp_dim, label_dsp_blk, dh, minvalue, maxvalue, font2, font3, SavePath, name):
    PD = pd.reshape(label_dsp_dim[0], label_dsp_dim[1])
    GT = gt.reshape(label_dsp_dim[0], label_dsp_dim[1])
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    im1 = ax1.imshow(GT, extent=[0, label_dsp_dim[1] * label_dsp_blk[1] * dh / 1000.,
                                 0, label_dsp_dim[0] * label_dsp_blk[0] * dh / 1000.], vmin=minvalue, vmax=maxvalue)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, ax=ax1, cax=cax1).set_label('Velocity (m/s)')
    plt.tick_params(labelsize=12)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(14)
    ax1.set_xlabel('Position (km)', font2)
    ax1.set_ylabel('Depth (km)', font2)
    ax1.set_title('Ground truth', font3)
    ax1.invert_yaxis()
    plt.subplots_adjust(bottom=0.15, top=0.92, left=0.08, right=0.98)
    # if not os.path.exists(SavePath+time_now):
    #     os.makedirs(SavePath+time_now)
    plt.savefig(SavePath + '/' + name + 'GT', transparent=True)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    im2 = ax2.imshow(PD, extent=[0, label_dsp_dim[1] * label_dsp_blk[1] * dh / 1000., \
                                 0, label_dsp_dim[0] * label_dsp_blk[0] * dh / 1000.], vmin=minvalue, vmax=maxvalue)

    plt.tick_params(labelsize=12)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontsize(14)
    ax2.set_xlabel('Position (km)', font2)
    ax2.set_ylabel('Depth (km)', font2)
    ax2.set_title('Prediction', font3)
    ax2.invert_yaxis()
    plt.subplots_adjust(bottom=0.15, top=0.92, left=0.08, right=0.98)
    # if not os.path.exists(SavePath+time_now):
    #     os.makedirs(SavePath+time_now)
    plt.savefig(SavePath + '/' + name + 'PD', transparent=True)
    # plt.show(fig1)
    # plt.show(fig2)
    plt.close()
