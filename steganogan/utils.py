# -*- coding: utf-8 -*-

# https://docs.python.org/3/library/zlib.html
# zlib模块提供压缩和解压缩数据的函数
import zlib
# this module provice access to the mathematical function defined by the C standard.
from math import exp

import torch
# https://pypi.org/project/reedsolo/
# RSCodec模块：一个纯python通用错误和擦除Reed-Solomon编码器
# ---encode：编码器，对输入的信息使用Reed-Solomon进行编码
from reedsolo import RSCodec
from torch.nn.functional import conv2d

rs = RSCodec(250)

# text转换为bits
# bytearray转换位bits
# text转换为bytearray
# text为什么要转换成bytearray()?
# ---bytearray()：返回一个新字节数组，数组元素是可变化的，每个元素的值范围[0,256)
# ---为将信息嵌入到图像作准备

# 秘密信息是文本的情况下
# 将文本信息转换为{0,1}中的整数列表
# ---需要先将文本信息转换成bytearray
# ---bytearray再转换为bits
def text_to_bits(text):
    """Convert text to a list of ints in {0, 1}"""
    # bytearray_to_bits自行定义的函数
    return bytearray_to_bits(text_to_bytearray(text))

# 整数列表转换成文本信息
# ---bits先转换为bytearray
# ---bytearray转换为text
def bits_to_text(bits):
    """Convert a list of ints in {0, 1} to text"""
    return bytearray_to_text(bits_to_bytearray(bits))

# 秘密信息是bytearray（字节数组）的情况下
def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []# 新建一个空的列表
    for i in x:
        # bin()：返回一个整数int或长整数long int的二进制表示
        bits = bin(i)[2:] # 取i第三位以后的全部信息，使用自定义形式进行重新编码，对信息进行一定程度的加密
        # 补齐，让所有位的编码长度一致
        bits = '00000000'[len(bits):] + bits
        # 写进列表中
        result.extend([int(b) for b in bits])
    return result


def bits_to_bytearray(bits):
    """Convert a list of bits to a bytearray"""
    ints = []
    # 为什么是对8除？然后向下取整
    # ---编码是8位
    # ---灰度图像像素的数据类型一般为8位无符号整数的（int8）
    for b in range(len(bits) // 8):
        # ??
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))
    # 二进制ints转换成bytearray形式
    return bytearray(ints)

# 文本信息转换为字节数组bytearray
def text_to_bytearray(text):
    """Compress and add error correction"""
    # 压缩并添加纠错
    # assert:判断表达式，表达式条件位false时触发异常，并输出"expected a string"
    assert isinstance(text, str), "expected a string"
    # 使用utf-8对文本进行编码后，进行压缩
    x = zlib.compress(text.encode("utf-8"))
    # 将x转换为bytearray再进行编码
    x = rs.encode(bytearray(x))
    return x


def bytearray_to_text(x):
    """Apply error correction and decompress"""
    try:
        text = rs.decode(x)
        text = zlib.decompress(text)
        return text.decode("utf-8")
    except BaseException:
        return False
  

def first_element(storage, loc):
    """Returns the first element of two"""
    # 返回两个第一元素
    return storage

# 高斯变换----一种近似方法
# ---随机过程熊概率桀骜都选择目标函数的可能分布，能通过样本采样逼近真实的目标函数；
# ---高斯函数支持扩展到无穷打，需要再窗口的末端截断，或用另一个零端窗口进行窗口变换；
# ---与使用神经网络进行近似的过程相比，高斯过程不需要昂贵的训练阶段，
#    ---并根据观察值对潜在的真是函数进行推断，在测试阶段具有非常灵活的属性。【但计算复杂度高】
# --扩展知识：高斯卷积：高斯滤波消除图像高频信息：https://zhuanlan.zhihu.com/p/143264646
def gaussian(window_size, sigma):
    """Gaussian window.
    https://en.wikipedia.org/wiki/Window_function#Gaussian_window
    """
    # 高斯变换可用于频率估计近似精确二次插值，公式
    _exp = [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.Tensor(_exp)
    return gauss / gauss.sum()

# 创建高斯滑动窗口==等价于卷积核吗？
def create_window(window_size, channel):
    # .unsqueeze(1):在第二维增加一个维度
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

# 结构相似性评价体系--两张图像的相似程度
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    padding_size = window_size // 2# 填充大小

    mu1 = conv2d(img1, window, padding=padding_size, groups=channel)
    mu2 = conv2d(img2, window, padding=padding_size, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    _ssim_quotient = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    _ssim_divident = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = _ssim_quotient / _ssim_divident

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    # 将window数据类型转换成img1的数据类型
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
