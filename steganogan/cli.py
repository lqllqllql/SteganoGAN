# -*- coding: utf-8 -*-

import argparse
# 警告信息模块
import warnings
# 源更改警告
from torch.serialization import SourceChangeWarning

from steganogan.models import SteganoGAN
# 通过警告过滤器控制是否发出警告消息
# ---显然ignore是忽略源更改警告
warnings.filterwarnings('ignore', category=SourceChangeWarning)

# 获取steganogan
# https://zhuanlan.zhihu.com/p/50804195
# --args,kwargs函数调用，调用时相当于pack(打包)，unpack(解包)；类似元组的打包和解包
def _get_steganogan(args):
    # 元组解包后传给对应的实参
    steganogan_kwargs = {
        'cuda': not args.cpu,
        'verbose': args.verbose
    }

    if args.path:
        steganogan_kwargs['path'] = args.path
    else:
        steganogan_kwargs['architecture'] = args.architecture
    # https://blog.csdn.net/daerzei/article/details/100598901
    # json模块的一个操作：.load()：操作的是文件流；
    # ---最终转换为python对象【所有python基本数据类型，列表，元组，字典以及自己定义的类】
    
    # https://zhuanlan.zhihu.com/p/50804195
    # --1、**kwargs：可变参数
    #   --将一个可变的关键字参数的字典传给函数实参
    return SteganoGAN.load(**steganogan_kwargs)


def _encode(args):
    """Given loads a pretrained pickel, encodes the image with it."""
    # 加载一个给定的预训练pickel，使用其对图像进行编码
    steganogan = _get_steganogan(args)# 参数
    steganogan.encode(args.cover, args.output, args.message)


def _decode(args):
    try:
        steganogan = _get_steganogan(args)
        message = steganogan.decode(args.image)

        if args.verbose:
            print('Message successfully decoded:')

        print(message)

    except Exception as e:
        print('ERROR: {}'.format(e))
        # traceback模块：提供一个标准接口，用于提取，格式化和打印python程序的堆栈跟踪
        import traceback
        # .print_exc()：打印异常信息和将跟踪条目从回溯对象tb堆叠到文件
        # ---再堆栈跟踪之后打印异常类型和值
        traceback.print_exc()

# 解析器及参数
def _get_parser():
    # Parent Parser - Shared options
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    group = parent.add_mutually_exclusive_group()
    group.add_argument('-a', '--architecture', default='dense',
                       choices={'basic', 'dense', 'residual'},
                       help='Model architecture. Use the same one for both encoding and decoding')

    group.add_argument('-p', '--path', help='Load a pretrained model from a given path.')
    parent.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')

    parser = argparse.ArgumentParser(description='SteganoGAN Command Line Interface')

    subparsers = parser.add_subparsers(title='action', help='Action to perform')
    parser.set_defaults(action=None)

    # Encode Parser
    encode = subparsers.add_parser('encode', parents=[parent],
                                   help='Hide a message into a steganographic image')
    encode.set_defaults(action=_encode)
    encode.add_argument('-o', '--output', default='output.png',
                        help='Path and name to save the output image')
    encode.add_argument('cover', help='Path to the image to use as cover')
    encode.add_argument('message', help='Message to encode')

    # Decode Parser
    decode = subparsers.add_parser('decode', parents=[parent],
                                   help='Read a message from a steganographic image')
    decode.set_defaults(action=_decode)
    decode.add_argument('image', help='Path to the image with the hidden message')
    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        parser.exit()

    args.action(args)
