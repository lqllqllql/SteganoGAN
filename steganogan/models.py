# -*- coding: utf-8 -*-
# gc模块提供可选的垃圾回收器的接口
import gc
# inspect模块提供函数义获取有关实时对象（模块、类、方法、函数、回溯、框架对象和代码对象）
# ---类型检查，获取源代码，检查类和函数以及检查解析器堆栈。
# https://docs.python.org/3/library/inspect.html
import inspect
import json
import os
from collections import Counter
# imageio模块提供一个简单的界面来读取和写入各种图像数据
import imageio
from imageio import imread, imwrite
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam
from tqdm import tqdm

from steganogan.utils import bits_to_bytearray, bytearray_to_text, ssim, text_to_bits

import pdb
import tensorflow as tf

# 默认路径
# --os.path专门解析地址；
# --os.path.abspath(__file__):获取当下程序运行的完整地址
#   --__file__是此方法的特殊用法；必须是.py文件
# --os.path.dirname()：获取os.path.abspath(__file__)的上一层的路径【绝对路径】
# 获取训练集数据的路径
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'train')

# 如何在模型训练过程中输出这些数据呢？？？？
METRIC_FIELDS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',#准确率
    'val.cover_score',
    'val.generated_score',
    'val.ssim',# 评价指标
    'val.psnr',# 评价指标
    'val.bpp',# 载荷能力
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.generated_score',
]

#POINT=['point']

# SteganoGAN
class SteganoGAN(object):

    def _get_instance(self, class_or_instance, kwargs):
        """Returns an instance of the class"""
        # 判断这个class是否存在
        if not inspect.isclass(class_or_instance):
            return class_or_instance
        
        # **args和**kwargs：https://wiki.jikexueyuan.com/project/interpy-zh/args_kwargs/Usage_kwargs.html
        # inspect.getfullargspec():获取函数参数的名称和默认值。
        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        # 产生键值对参数
        init_args = {arg: kwargs[arg] for arg in argspec}
        return class_or_instance(**init_args)

    def set_device(self, cuda=True):
        """Sets the torch device depending on whether cuda is avaiable or not."""
        # 根据cuda是否存在设置模型设备
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')

        if self.verbose:
            if not cuda:
                print('Using CPU device')
            elif not self.cuda:
                print('CUDA is not available. Defaulting to CPU device')
            else:
                print('Using CUDA device')
        # 三个模型指定配置
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)
    
    # 定义一些参数
    def __init__(self, data_depth, encoder, decoder, critic,
                 cuda=False, verbose=False, log_dir=None, **kwargs):

        self.verbose = verbose

        self.data_depth = data_depth
        kwargs['data_depth'] = data_depth
        # 在三个模型中检查相应模型和参数是否存在
        self.encoder = self._get_instance(encoder, kwargs)
        self.decoder = self._get_instance(decoder, kwargs)
        self.critic = self._get_instance(critic, kwargs)
        self.set_device(cuda)

        self.critic_optimizer = None
        self.decoder_optimizer = None

        # Misc
        self.fit_metrics = None
        self.history = list()

        # 日志文件
        self.log_dir = log_dir
        if log_dir:
            # 创建一个文件夹
            os.makedirs(self.log_dir, exist_ok=True)
            # 样本存放路径
            self.samples_path = os.path.join(self.log_dir, 'samples')
            os.makedirs(self.samples_path, exist_ok=True)
    
    # 生成随机信息作为秘密信息（图像大小和cover图像一致）
    # 改进--对随机生成的秘密信息进行加密
    def _random_data(self, cover):
        """Generate random data ready to be hidden inside the cover image.
        范围(0,2)
        Args:
            cover (image): Image to use as cover.

        Returns:
            generated (image): Image generated with the encoded message.
        """
        N, _, H, W = cover.size()
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)

    def _encode_decode(self, cover, quantize=False):
        """Encode random data and then decode it.
        
        Args:
            cover (image): Image to use as cover.
            # 是否对生成的图像进行量化
            quantize (bool): whether to quantize the generated image or not.

        Returns:
            generated (image): Image generated with the encoded message.
            payload (bytes): Random data that has been encoded in the image.
            decoded (bytes): Data decoded from the generated image.
        """
        payload = self._random_data(cover)
        # 隐藏阶段
        generated = self.encoder(cover, payload)
        #标准化
        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0
        # 解码阶段
        decoded = self.decoder(generated)

        return generated, payload, decoded

    # 评价体系
    def _critic(self, image):
        """Evaluate the image using the critic"""
        return torch.mean(self.critic(image))

    # 优化器
    def _get_optimizers(self):
        _dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        critic_optimizer = Adam(self.critic.parameters(), lr=1e-4)
        decoder_optimizer = Adam(_dec_list, lr=1e-4)
        return critic_optimizer, decoder_optimizer
    
    # 评价模型
    def _fit_critic(self, train, metrics):
        """Critic process"""
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()# 无参数，启用完全的垃圾回收
            cover = cover.to(self.device)
            payload = self._random_data(cover)
            generated = self.encoder(cover, payload)
            cover_score = self._critic(cover)
            generated_score = self._critic(generated)

            self.critic_optimizer.zero_grad()
            (cover_score - generated_score).backward(retain_graph=False)
            self.critic_optimizer.step()

            for p in self.critic.parameters():
                p.data.clamp_(-0.1, 0.1)

            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.generated_score'].append(generated_score.item())

    # 评价encoder和decoder
    def _fit_coders(self, train, metrics):
        """Fit the encoder and the decoder on the train images."""
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover)
            # 准确度在这里输出
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(
                cover, generated, payload, decoded)
            generated_score = self._critic(generated)

            self.decoder_optimizer.zero_grad()
            (100.0 * encoder_mse + decoder_loss + generated_score).backward()
            self.decoder_optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())
    
    
    def _coding_scores(self, cover, generated, payload, decoded):
        encoder_mse = mse_loss(generated, cover)
        decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
        decoder_acc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()

        return encoder_mse, decoder_loss, decoder_acc

    def _validate(self, validate, metrics):
        """Validation process"""
        for cover, _ in tqdm(validate, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded = self._encode_decode(cover, quantize=True)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(
                cover, generated, payload, decoded)
            generated_score = self._critic(generated)
            cover_score = self._critic(cover)

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            metrics['val.ssim'].append(ssim(cover, generated).item())
            metrics['val.psnr'].append(10 * torch.log10(4 / encoder_mse).item())
            metrics['val.bpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))

    # 生成样本        
    def _generate_samples(self, samples_path, cover, epoch):
        cover = cover.to(self.device)
        generated, payload, decoded = self._encode_decode(cover)
        samples = generated.size(0)
        for sample in range(samples):
            cover_path = os.path.join(samples_path, '{}.cover.png'.format(sample))
            sample_name = '{}.generated-{:2d}.png'.format(sample, epoch)
            sample_path = os.path.join(samples_path, sample_name)

            image = (cover[sample].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
            imageio.imwrite(cover_path, (255.0 * image).astype('uint8'))

            sampled = generated[sample].clamp(-1.0, 1.0).permute(1, 2, 0)
            sampled = sampled.detach().cpu().numpy() + 1.0

            image = sampled / 2.0
            imageio.imwrite(sample_path, (255.0 * image).astype('uint8'))
     
    # 保存断点信息
    # https://docs.python.org/3.7/library/functions.html#breakpoint
    # 实操：https://api.bbs.cvmart.net/articles/4774
    #def breakpoint(self,):
    #def save_breakpoint(self,):
    
    # 调包哈哈哈
    # import tensorflow as tf
    #def _save_breakpoint(self,checkpath,point):
    #checkpath='/content/drive/MyDrive/models/steganoGAN/check'
    #checkpoint=tf.keras.callbacks.ModelCheckpoint('checkpath',save_weights_only=True,save_freq=5)
    
    
    # 我猜在这里加入断点保存
    def fit(self, train, validate, epochs=5, callbacks):
        """Train a new model with the given ImageLoader class."""

        if self.critic_optimizer is None:
            self.critic_optimizer, self.decoder_optimizer = self._get_optimizers()
            self.epochs = 0

        if self.log_dir:
            sample_cover = next(iter(validate))[0]

        # Start training
        total = self.epochs + epochs
        for epoch in range(1, epochs + 1):
            # Count how many epochs we have trained for this steganogan
            self.epochs += 1

            metrics = {field: list() for field in METRIC_FIELDS}

            if self.verbose:
                print('Epoch {}/{}'.format(self.epochs, total))

            self._fit_critic(train, metrics)
            self._fit_coders(train, metrics)
            self._validate(validate, metrics)

            self.fit_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
            self.fit_metrics['epoch'] = epoch

            if self.log_dir:
                self.history.append(self.fit_metrics)

                metrics_path = os.path.join(self.log_dir, 'metrics.log')
                with open(metrics_path, 'w') as metrics_file:
                    json.dump(self.history, metrics_file, indent=4)

                save_name = '{}.bpp-{:03f}.p'.format(
                    self.epochs, self.fit_metrics['val.bpp'])

                self.save(os.path.join(self.log_dir, save_name))
                self._generate_samples(self.samples_path, sample_cover, epoch)
            
            #self._save_breakpoint(checkpath,point)
            
            # Empty cuda cache (this may help for memory leaks)
            if self.cuda:
                torch.cuda.empty_cache()

            gc.collect()
            
            #callbacks['point'].append(cheackpoint.item())

    def _make_payload(self, width, height, depth, text):
        """
        This takes a piece of text and encodes it into a bit vector. It then
        fills a matrix of size (width, height) with copies of the bit vector.
        讲一段文本编码为位向量。使用位向量的副本填充大小位（w,h）的矩阵。
        """
        message = text_to_bits(text) + [0] * 32
        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width)
    
    # 编码网络
    def encode(self, cover, output, text):
        """Encode an image.
        Args:
            cover (str): Path to the image to be used as cover.
            output (str): Path where the generated image will be saved.
            text (str): Message to hide inside the image.
        """
        cover = imread(cover, pilmode='RGB') / 127.5 - 1.0
        cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        cover_size = cover.size()
        # _, _, height, width = cover.size()
        payload = self._make_payload(cover_size[3], cover_size[2], self.data_depth, text)

        cover = cover.to(self.device)
        payload = payload.to(self.device)
        generated = self.encoder(cover, payload)[0].clamp(-1.0, 1.0)

        generated = (generated.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(output, generated.astype('uint8'))

        if self.verbose:
            print('Encoding completed.')

    def decode(self, image):
        if not os.path.exists(image):
            raise ValueError('Unable to read %s.' % image)

        # extract a bit vector
        image = imread(image, pilmode='RGB') / 255.0
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = image.to(self.device)

        image = self.decoder(image).view(-1) > 0

        # split and decode messages
        candidates = Counter()
        bits = image.data.int().cpu().numpy().tolist()
        for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
            candidate = bytearray_to_text(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1

        # choose most common message
        if len(candidates) == 0:
            raise ValueError('Failed to find message.')

        candidate, count = candidates.most_common(1)[0]
        return candidate

    def save(self, path):
        """Save the fitted model in the given path. Raises an exception if there is no model."""
        torch.save(self, path)

    @classmethod
    def load(cls, architecture=None, path=None, cuda=True, verbose=False):
        """Loads an instance of SteganoGAN for the given architecture (default pretrained models)
        or loads a pretrained model from a given path.

        Args:
            architecture(str): Name of a pretrained model to be loaded from the default models.
            path(str): Path to custom pretrained model. *Architecture must be None.
            cuda(bool): Force loaded model to use cuda (if available).
            verbose(bool): Force loaded model to use or not verbose.
        """

        if architecture and not path:
            model_name = '{}.steg'.format(architecture)
            pretrained_path = os.path.join(os.path.dirname(__file__), 'pretrained')
            path = os.path.join(pretrained_path, model_name)

        elif (architecture is None and path is None) or (architecture and path):
            raise ValueError(
                'Please provide either an architecture or a path to pretrained model.')

        steganogan = torch.load(path, map_location='cpu')
        steganogan.verbose = verbose

        steganogan.encoder.upgrade_legacy()
        steganogan.decoder.upgrade_legacy()
        steganogan.critic.upgrade_legacy()

        steganogan.set_device(cuda)
        return steganogan
