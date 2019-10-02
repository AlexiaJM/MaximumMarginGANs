#!/usr/bin/env python3

# To get TensorBoard output, use the python command: tensorboard --logdir /home/alexia/Output/DCGAN
# TensorBoard disabled for now.

# To get CIFAR10
# wget http://pjreddie.com/media/files/cifar.tgz
# tar xzf cifar.tgz


## Parameters

# thanks https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def strToBool(str):
	return str.lower() in ('true', 'yes', 'on', 't', '1')

import argparse
parser = argparse.ArgumentParser()
parser.register('type', 'bool', strToBool)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=32) # DCGAN paper original value used 128 (32 is generally better to prevent vanishing gradients with SGAN and LSGAN, not important with relativistic GANs)
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--z_size', type=int, default=128)
parser.add_argument('--G_h_size', type=int, default=128, help='Number of hidden nodes in the Generator. Used only in arch=0. Too small leads to bad results, too big blows up the GPU RAM.') # DCGAN paper original value
parser.add_argument('--D_h_size', type=int, default=128, help='Number of hidden nodes in the Discriminator. Used only in arch=0. Too small leads to bad results, too big blows up the GPU RAM.') # DCGAN paper original value
parser.add_argument('--conv_size', type=int, default=64, help='Size of convolutions when using Self-Attention GAN.')
parser.add_argument('--resample', type=int, default=1, help="Resample data in the generator step (Recommended, may affect performance sightly)")
parser.add_argument('--centercrop', type=int, default=0, help="If not 0, CenterCrop with specified number the images")
parser.add_argument('--lr_D', type=float, default=.0001, help='Discriminator learning rate')
parser.add_argument('--lr_G', type=float, default=.0001, help='Generator learning rate')
parser.add_argument('--n_iter', type=int, default=100000, help='Number of iteration cycles')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam betas[0], DCGAN paper recommends .50 instead of the usual .90')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1]')
parser.add_argument('--decay', type=float, default=0, help='Decay to apply to lr each cycle. decay^n_iter gives the final lr. Ex: .00002 will lead to .13 of lr after 100k cycles')
parser.add_argument('--SELU', type='bool', default=False, help='Using scaled exponential linear units (SELU) which are self-normalizing instead of ReLU with BatchNorm. Used only in arch=0. This improves stability.')
parser.add_argument("--NN_conv", type='bool', default=False, help="This approach minimize checkerboard artifacts during training. Used only by arch=0. Uses nearest-neighbor resized convolutions instead of strided convolutions (https://distill.pub/2016/deconv-checkerboard/ and github.com/abhiskk/fast-neural-style).")
parser.add_argument('--seed', type=int)
parser.add_argument('--input_folder', default='/Datasets/Meow_32x32', help='input folder')
parser.add_argument('--output_folder', default='/Output/GANlosses', help='output folder')
parser.add_argument('--load', default=None, help='Full path to network state to load (ex: /network/home/output_folder/run-5/models/state_11.pth)')
parser.add_argument('--cuda', type='bool', default=True, help='enables cuda')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--loss_D', type=int, default=1, help='Loss of D, see code for details')
parser.add_argument('--Diters', type=int, default=1, help='Number of iterations of D')
parser.add_argument('--Giters', type=int, default=1, help='Number of iterations of G.')
parser.add_argument('--spectral', type='bool', default=False, help='If True, use spectral normalization to make the discriminator Lipschitz. This Will also remove batch norm in the discriminator.')
parser.add_argument('--spectral_G', type='bool', default=False, help='If True, use spectral normalization to make the generator Lipschitz (Generally only D is spectral, not G). This Will also remove batch norm in the discriminator.')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization weight. Helps convergence but leads to artifacts in images, not recommended.')
parser.add_argument('--gen_extra_images', type=int, default=50000, help='Generate additional images with random fake cats in calculating FID (Recommended to use the same amount as the size of the dataset; for CIFAR-10 we use 50k, but most people use 10k) It must be a multiple of 100.')
parser.add_argument('--gen_every', type=int, default=100000, help='Generate additional images with random fake cats every x iterations. Used in calculating FID.')
parser.add_argument('--logs_folder', default='/scratch/jolicoea/Output/Extra', help='Folder for models and FID logs')
parser.add_argument('--extra_folder', default='/Output/Extra', help='Folder for images for FID calculation')
parser.add_argument('--show_graph', type='bool', default=False, help='If True, show gradients graph. Really neat for debugging.')
parser.add_argument('--no_batch_norm_G', type='bool', default=False, help='If True, no batch norm in G.')
parser.add_argument('--no_batch_norm_D', type='bool', default=False, help='If True, no batch norm in D.')
parser.add_argument('--Tanh_GD', type='bool', default=False, help='If True, tanh everywhere.')
parser.add_argument('--arch', type=int, default=0, help='0:DCGAN with number of layers adjusted based on image size, 1: standard CNN  for 32x32 images from the Spectral GAN paper. Some options may be ignored by some architectures.')
parser.add_argument('--print_every', type=int, default=1000, help='Generate a mini-batch of images at every x iterations (to see how the training progress, you can do it often).')
parser.add_argument('--save', type='bool', default=False, help='Do we save models, yes or no? It will be saved in extra_folder')
parser.add_argument('--CIFAR10', type='bool', default=False, help='If True, use CIFAR-10 instead of your own dataset. Make sure image_size is set to 32!')
parser.add_argument('--CIFAR10_input_folder', default='/Datasets/CIFAR10', help='input folder (automatically downloaded)')
parser.add_argument('--LSUN', type='bool', default=False, help='If True, use LSUN instead of your own dataset.')
parser.add_argument('--LSUN_input_folder', default='/Datasets/LSUN', help='input folder')
parser.add_argument('--LSUN_classes', nargs='+', default='bedroom_train', help='Classes to use (see https://pytorch.org/docs/stable/torchvision/datasets.html#lsun)')
parser.add_argument("--no_bias", type='bool', default=False, help="Unbiased estimator when using RaLSGAN (loss_D=12) or RcLSGAN (loss_D=22)")

# Options for Gradient penalties
parser.add_argument('--penalty', type=float, default=20, help='Gradient penalty parameter for Gradien penalties')
parser.add_argument('--grad_penalty', type='bool', default=False, help='If True, use gradient penalty of WGAN-GP but with whichever loss_D chosen. No need to set this true with WGAN-GP.')
parser.add_argument('--no_grad_penalty', type='bool', default=False, help='If True, do not use gradient penalty when using WGAN-GP (If you want to try Spectral normalized WGAN).')
parser.add_argument('--grad_penalty_aug', type='bool', default=False, help='If True, use augmented lagrangian for gradient penalty (aka Sobolev GAN).')
parser.add_argument('--fake_only', type='bool', default=False, help='Using fake data only in gradient penalty')
parser.add_argument('--real_only', type='bool', default=False, help='Using real data only in gradient penalty')
parser.add_argument('--delta', type=float, default=1, help='(||grad_D(x)|| - delta)^2')
parser.add_argument('--rho', type=float, default=.0001, help='learning rate of lagrange multiplier when using augmented lagrangian')
parser.add_argument('--penalty-type', help='Gradient penalty type.  The default ("squared-diff") forces gradient norm *equal* to 1, which is not correct, but is what is done in the original WGAN paper. True Lipschitz constraint is with "clamp"',
					choices=['clamp','squared-clamp','squared-diff','squared','TV','abs','hinge','hinge2'], default='squared-diff')
parser.add_argument('--reduction', help='Summary statistic for gradient penalty over batches (default: "mean")',
					choices=['mean','max','softmax'],default='mean')

# Max Margin
parser.add_argument('--l1_margin', help='maximize L-1 margin (equivalent to penalizing L-infinity gradient norm)',action='store_true')
parser.add_argument('--l1_margin_logsumexp', help='maximize L-1 margin using logsumexp to approximate L-infinity gradient norm (equivalent to penalizing L-infinity gradient norm)',action='store_true')
parser.add_argument('--l1_margin_smoothmax', help='maximize L-1 margin using smooth max to approximate L-infinity gradient norm (equivalent to penalizing L-infinity gradient norm)',action='store_true')
parser.add_argument('--linf_margin', help='maximize L-infinity margin (equivalent to penalizing L-1 gradient norm)',action='store_true')
parser.add_argument('--smoothmax', type=float, default=.5, help='parameter for smooth max (higher = less smooth)')
parser.add_argument('--l1_margin_no_abs', help='Only penalize positive gradient (Shouldnt work, but it does)',action='store_true')

param = parser.parse_args()
print('Arguments:')
for p in vars(param).items():
	print('  ',p[0]+': ',p[1])
print('\n')

## Imports
import torch.nn.functional as F

# Time
import time
import sys
start = time.time()

# Setting the title for the file saved
if param.loss_D == 1:
	title = 'GAN_'
if param.loss_D == 2:
	title = 'LSGAN_'
if param.loss_D == 3:
	title = 'HingeGAN_'
if param.loss_D == 4:
	title = 'WGANGP_'

if param.loss_D == 11:
	title = 'RaSGAN_'
if param.loss_D == 12:
	title = 'RaLSGAN_'
if param.loss_D == 13:
	title = 'RaHingeGAN_'

if param.loss_D == 21:
	title = 'RcSGAN_'
if param.loss_D == 22:
	title = 'RcLSGAN_'
if param.loss_D == 23:
	title = 'RcHingeGAN_'

if param.loss_D == 31:
	title = 'RpSGAN_'
if param.loss_D == 32:
	title = 'RpLSGAN_'
if param.loss_D == 33:
	title = 'RpHingeGAN_'

if param.loss_D == 41:
	title = 'RpSGAN_MVUE_'
if param.loss_D == 42:
	title = 'RpLSGAN_MVUE_'
if param.loss_D == 43:
	title = 'RpHingeGAN_MVUE_'

if param.no_bias:
	title = title + 'nobias_'

if param.seed is not None:
	title = title + 'seed%i' % param.seed


# Check folder run-i for all i=0,1,... until it finds run-j which does not exists, then creates a new folder run-j
import os

# Add local variable to folders so we work in the local drive (removed for public released)
#param.input_folder = os.environ["SLURM_TMPDIR"] + param.input_folder
#param.output_folder = os.environ["SLURM_TMPDIR"] + param.output_folder
#param.CIFAR10_input_folder = os.environ["SLURM_TMPDIR"] + param.CIFAR10_input_folder
#param.LSUN_input_folder = os.environ["SLURM_TMPDIR"] + param.LSUN_input_folder
#param.extra_folder = os.environ["SLURM_TMPDIR"] + param.extra_folder

run = 0
base_dir = f"{param.output_folder}/{title}-{run}"
while os.path.exists(base_dir):
	run += 1
	base_dir = f"{param.output_folder}/{title}-{run}"
os.makedirs(base_dir)
logs_dir = f"{base_dir}/logs"
os.makedirs(logs_dir)
os.makedirs(f"{base_dir}/images")
if param.gen_extra_images > 0 and not os.path.exists(f"{param.extra_folder}"):
	os.makedirs(f"{param.extra_folder}")
if param.gen_extra_images > 0 and not os.path.exists(f"{param.logs_folder}"):
	os.makedirs(f"{param.logs_folder}")

# where we save the output
log_output = open(f"{logs_dir}/log.txt", 'w')
print(param, file=log_output)

import numpy
import torch
import torch.autograd as autograd
from torch.autograd import Variable

# For plotting the Loss of D and G using tensorboard
# To fix later, not compatible with using tensorflow
#from tensorboard_logger import configure, log_value
#configure(logs_dir, flush_secs=5)

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transf
import torchvision.models as models
import torchvision.utils as vutils
import torch.nn.utils.spectral_norm as spectral_norm

if param.cuda:
	import torch.backends.cudnn as cudnn
	cudnn.deterministic = True
	cudnn.benchmark = True

# To see images
from IPython.display import Image
to_img = transf.ToPILImage()

#import pytorch_visualize as pv

import math

torch.utils.backcompat.broadcast_warning.enabled=True

from fid import calculate_fid_given_paths as calc_fid
#from inception import get_inception_score
#from inception import load_images

## Setting seed
import random
if param.seed is None:
	param.seed = random.randint(1, 10000)
print(f"Random Seed: {param.seed}")
print(f"Random Seed: {param.seed}", file=log_output)
random.seed(param.seed)
numpy.random.seed(param.seed)
torch.manual_seed(param.seed)
if param.cuda:
	torch.cuda.manual_seed_all(param.seed)

## Transforming images
if param.centercrop != 0: 
	trans = transf.Compose([
		transf.CenterCrop(param.centercrop),
		transf.Resize((param.image_size, param.image_size)),
		# This makes it into [0,1]
		transf.ToTensor(),
		# This makes it into [-1,1]
		transf.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
	])
else:
	trans = transf.Compose([
		transf.Resize((param.image_size, param.image_size)),
		# This makes it into [0,1]
		transf.ToTensor(),
		# This makes it into [-1,1]
		transf.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
	])

## Importing dataset
if param.CIFAR10:
	data = dset.CIFAR10(root=param.CIFAR10_input_folder, train=True, download=True, transform=trans)
elif param.LSUN:
	print(param.LSUN_classes)
	data = dset.LSUN(root=param.LSUN_input_folder, classes=[param.LSUN_classes], transform=trans)
else:
	data = dset.ImageFolder(root=param.input_folder, transform=trans)

# Loading data randomly
def generate_random_sample(size=param.batch_size):
	while True:
		random_indexes = numpy.random.choice(data.__len__(), size=size, replace=False)
		batch = [data[i][0] for i in random_indexes]
		yield torch.stack(batch, 0)
random_sample = generate_random_sample(size=param.batch_size)

## Models

if param.activation=='leaky':
	class Activation(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.act = torch.nn.LeakyReLU(0.1 if param.arch == 1 else .02, inplace=True)
		def forward(self, x):
			return self.act(x)
elif param.activation=='softplus':
	class Activation(torch.nn.Module):
		def __init__(self):
			super().__init__()
			#self.a = torch.nn.Parameter(torch.FloatTensor(1).fill_(1.))
			self.a = torch.nn.Softplus(1, 20.)
		def forward(self, x):
			#return F.softplus(x, self.a, 20.)
			return self.a(x)

if param.arch == 1:
	title = title + '_CNN_'

	class DCGAN_G(torch.nn.Module):
		def __init__(self):
			super(DCGAN_G, self).__init__()

			self.dense = torch.nn.Linear(param.z_size, 512 * 4 * 4)

			if param.spectral_G:
				model = [spectral_norm(torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True))]
				model += [torch.nn.ReLU(True),
					spectral_norm(torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True))]
				model += [torch.nn.ReLU(True),
					spectral_norm(torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True))]
				model += [torch.nn.ReLU(True),
					spectral_norm(torch.nn.Conv2d(64, param.n_colors, kernel_size=3, stride=1, padding=1, bias=True)),
					torch.nn.Tanh()]
			else:
				model = [torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_G:
					model += [torch.nn.BatchNorm2d(256)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.ReLU(True)]
				model += [torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_G:
					model += [torch.nn.BatchNorm2d(128)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.ReLU(True)]
				model += [torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_G:
					model += [torch.nn.BatchNorm2d(64)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.ReLU(True)]
				model += [torch.nn.Conv2d(64, param.n_colors, kernel_size=3, stride=1, padding=1, bias=True),
					torch.nn.Tanh()]
			self.model = torch.nn.Sequential(*model)

		def forward(self, input):
			if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
				output = torch.nn.parallel.data_parallel(self.model(self.dense(input.view(-1, param.z_size)).view(-1, 512, 4, 4)), input, range(param.n_gpu))
			else:
				output = self.model(self.dense(input.view(-1, param.z_size)).view(-1, 512, 4, 4))
			#print(output.size())
			return output

	class DCGAN_D(torch.nn.Module):
		def __init__(self):
			super(DCGAN_D, self).__init__()

			self.dense = torch.nn.Linear(512 * 4 * 4, 1)

			if param.spectral:
				model = [spectral_norm(torch.nn.Conv2d(param.n_colors, 64, kernel_size=3, stride=1, padding=1, bias=True)),
					Activation(),
					spectral_norm(torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)),
					Activation(),

					spectral_norm(torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)),
					Activation(),
					spectral_norm(torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)),
					Activation(),

					spectral_norm(torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)),
					Activation(),
					spectral_norm(torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)),
					Activation(),

					spectral_norm(torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)),
					Activation()]
			else:
				model = [torch.nn.Conv2d(param.n_colors, 64, kernel_size=3, stride=1, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(64)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [Activation()]
				model += [torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(64)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [Activation()]
				model += [torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(128)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [Activation()]
				model += [torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(128)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [Activation()]
				model += [torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(256)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [Activation()]
				model += [torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(256)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [Activation()]
				model += [torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [Activation()]
			self.model = torch.nn.Sequential(*model)

		def forward(self, input):
			if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
				output = torch.nn.parallel.data_parallel(self.dense(self.model(input).view(-1, 512 * 4 * 4)).view(-1), input, range(param.n_gpu))
			else:
				output = self.dense(self.model(input).view(-1, 512 * 4 * 4)).view(-1)
			#print(output.size())
			return output

if param.arch == 0:

	# DCGAN generator
	class DCGAN_G(torch.nn.Module):
		def __init__(self):
			super(DCGAN_G, self).__init__()
			main = torch.nn.Sequential()

			# We need to know how many layers we will use at the beginning
			mult = param.image_size // 8

			### Start block
			# Z_size random numbers
			if param.spectral_G:
				main.add_module('Start-SpectralConvTranspose2d', torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.z_size, param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False)))
			else:
				main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(param.z_size, param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False))
			if param.SELU:
				main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
			else:
				if not param.no_batch_norm_G and not param.spectral_G:
					main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(param.G_h_size * mult))
				if param.Tanh_GD:
					main.add_module('Start-Tanh', torch.nn.Tanh())
				else:
					main.add_module('Start-ReLU', torch.nn.ReLU())
			# Size = (G_h_size * mult) x 4 x 4

			### Middle block (Done until we reach ? x image_size/2 x image_size/2)
			ii = 1
			while mult > 1:
				if param.NN_conv:
					main.add_module('Middle-UpSample [%d]' % ii, torch.nn.Upsample(scale_factor=2))
					if param.spectral_G:
						main.add_module('Middle-SpectralConv2d [%d]' % ii, torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=3, stride=1, padding=1)))
					else:
						main.add_module('Middle-Conv2d [%d]' % ii, torch.nn.Conv2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=3, stride=1, padding=1))
				else:
					if param.spectral_G:
						main.add_module('Middle-SpectralConvTranspose2d [%d]' % ii, torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False)))
					else:
						main.add_module('Middle-ConvTranspose2d [%d]' % ii, torch.nn.ConvTranspose2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
				if param.SELU:
					main.add_module('Middle-SELU [%d]' % ii, torch.nn.SELU(inplace=True))
				else:
					if not param.no_batch_norm_G and not param.spectral_G:
						main.add_module('Middle-BatchNorm2d [%d]' % ii, torch.nn.BatchNorm2d(param.G_h_size * (mult//2)))
					if param.Tanh_GD:
						main.add_module('Middle-Tanh [%d]' % ii, torch.nn.Tanh())
					else:
						main.add_module('Middle-ReLU [%d]' % ii, torch.nn.ReLU())
				# Size = (G_h_size * (mult/(2*i))) x 8 x 8
				mult = mult // 2
				ii += 1

			### End block
			# Size = G_h_size x image_size/2 x image_size/2
			if param.NN_conv:
				main.add_module('End-UpSample', torch.nn.Upsample(scale_factor=2))
				if param.spectral_G:
					main.add_module('End-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.G_h_size, param.n_colors, kernel_size=3, stride=1, padding=1)))
				else:
					main.add_module('End-Conv2d', torch.nn.Conv2d(param.G_h_size, param.n_colors, kernel_size=3, stride=1, padding=1))
			else:
				if param.spectral_G:
					main.add_module('End-SpectralConvTranspose2d', torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.G_h_size, param.n_colors, kernel_size=4, stride=2, padding=1, bias=False)))
				else:
					main.add_module('End-ConvTranspose2d', torch.nn.ConvTranspose2d(param.G_h_size, param.n_colors, kernel_size=4, stride=2, padding=1, bias=False))
			main.add_module('End-Tanh', torch.nn.Tanh())
			# Size = n_colors x image_size x image_size
			self.main = main

		def forward(self, input):
			if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
				output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
			else:
				output = self.main(input)
			return output

	# DCGAN discriminator (using somewhat the reverse of the generator)
	class DCGAN_D(torch.nn.Module):
		def __init__(self):
			super(DCGAN_D, self).__init__()
			main = torch.nn.Sequential()

			### Start block
			# Size = n_colors x image_size x image_size
			if param.spectral:
				main.add_module('Start-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.n_colors, param.D_h_size, kernel_size=4, stride=2, padding=1, bias=False)))
			else:
				main.add_module('Start-Conv2d', torch.nn.Conv2d(param.n_colors, param.D_h_size, kernel_size=4, stride=2, padding=1, bias=False))
			if param.SELU:
				main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
			else:
				if param.Tanh_GD:
					main.add_module('Start-Tanh', torch.nn.Tanh())
				else:
					main.add_module('Start-LeakyReLU', Activation())
			image_size_new = param.image_size // 2
			# Size = D_h_size x image_size/2 x image_size/2

			### Middle block (Done until we reach ? x 4 x 4)
			mult = 1
			ii = 0
			while image_size_new > 4:
				if param.spectral:
					main.add_module('Middle-SpectralConv2d [%d]' % ii, torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.D_h_size * mult, param.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
				else:
					main.add_module('Middle-Conv2d [%d]' % ii, torch.nn.Conv2d(param.D_h_size * mult, param.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False))
				if param.SELU:
					main.add_module('Middle-SELU [%d]' % ii, torch.nn.SELU(inplace=True))
				else:
					if not param.no_batch_norm_D and not param.spectral:
						main.add_module('Middle-BatchNorm2d [%d]' % ii, torch.nn.BatchNorm2d(param.D_h_size * (2*mult)))
					if param.Tanh_GD:
						main.add_module('Start-Tanh [%d]' % ii, torch.nn.Tanh())
					else:
						main.add_module('Middle-LeakyReLU [%d]' % ii, Activation())
				# Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
				image_size_new = image_size_new // 2
				mult *= 2
				ii += 1

			### End block
			# Size = (D_h_size * mult) x 4 x 4
			if param.spectral:
				main.add_module('End-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False)))
			else:
				main.add_module('End-Conv2d', torch.nn.Conv2d(param.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
			# Size = 1 x 1 x 1 (Is a real cat or not?)
			self.main = main

		def forward(self, input):
			if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
				output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
			else:
				output = self.main(input)
			# Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
			return output.view(-1)

if param.arch == 2:
# Taken directly from https://github.com/ozanciga/gans-with-pytorch/blob/master/wgan-gp/models.py

	class MeanPoolConv(torch.nn.Module):
		def __init__(self, n_input, n_output, k_size):
			super(MeanPoolConv, self).__init__()
			conv1 = torch.nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
			self.model = torch.nn.Sequential(conv1)
		def forward(self, x):
			out = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4.0
			out = self.model(out)
			return out

	class ConvMeanPool(torch.nn.Module):
		def __init__(self, n_input, n_output, k_size):
			super(ConvMeanPool, self).__init__()
			conv1 = torch.nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
			self.model = torch.nn.Sequential(conv1)
		def forward(self, x):
			out = self.model(x)
			out = (out[:,:,::2,::2] + out[:,:,1::2,::2] + out[:,:,::2,1::2] + out[:,:,1::2,1::2]) / 4.0
			return out

	class UpsampleConv(torch.nn.Module):
		def __init__(self, n_input, n_output, k_size):
			super(UpsampleConv, self).__init__()

			self.model = torch.nn.Sequential(
				torch.nn.PixelShuffle(2),
				torch.nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
			)
		def forward(self, x):
			x = x.repeat((1, 4, 1, 1)) # Weird concat of WGAN-GPs upsampling process.
			out = self.model(x)
			return out

	class ResidualBlock(torch.nn.Module):
		def __init__(self, n_input, n_output, k_size, resample='up', bn=True, spatial_dim=None):
			super(ResidualBlock, self).__init__()

			self.resample = resample

			if resample == 'up':
				self.conv1 = UpsampleConv(n_input, n_output, k_size)
				self.conv2 = torch.nn.Conv2d(n_output, n_output, k_size, padding=(k_size-1)//2)
				self.conv_shortcut = UpsampleConv(n_input, n_output, k_size)
				self.out_dim = n_output
			elif resample == 'down':
				self.conv1 = torch.nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
				self.conv2 = ConvMeanPool(n_input, n_output, k_size)
				self.conv_shortcut = ConvMeanPool(n_input, n_output, k_size)
				self.out_dim = n_output
				self.ln_dims = [n_input, spatial_dim, spatial_dim] # Define the dimensions for layer normalization.
			else:
				self.conv1 = torch.nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
				self.conv2 = torch.nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
				self.conv_shortcut = None # Identity
				self.out_dim = n_input
				self.ln_dims = [n_input, spatial_dim, spatial_dim]

			self.model = torch.nn.Sequential(
				torch.nn.BatchNorm2d(n_input) if bn else torch.nn.LayerNorm(self.ln_dims),
				torch.nn.ReLU(inplace=True),
				self.conv1,
				torch.nn.BatchNorm2d(self.out_dim) if bn else torch.nn.LayerNorm(self.ln_dims),
				torch.nn.ReLU(inplace=True),
				self.conv2,
			)

		def forward(self, x):
			if self.conv_shortcut is None:
				return x + self.model(x)
			else:
				return self.conv_shortcut(x) + self.model(x)

	class DiscBlock1(torch.nn.Module):
		def __init__(self, n_output):
			super(DiscBlock1, self).__init__()

			self.conv1 = torch.nn.Conv2d(3, n_output, 3, padding=(3-1)//2)
			self.conv2 = ConvMeanPool(n_output, n_output, 1)
			self.conv_shortcut = MeanPoolConv(3, n_output, 1)

			self.model = torch.nn.Sequential(
				self.conv1,
				torch.nn.ReLU(inplace=True),
				self.conv2
			)

		def forward(self, x):
			return self.conv_shortcut(x) + self.model(x)

	class DCGAN_G(torch.nn.Module):
		def __init__(self):
			super(DCGAN_G, self).__init__()

			self.model = torch.nn.Sequential(                     # 128 x 1 x 1
				torch.nn.ConvTranspose2d(128, 128, 4, 1, 0),      # 128 x 4 x 4
				ResidualBlock(128, 128, 3, resample='up'),  # 128 x 8 x 8
				ResidualBlock(128, 128, 3, resample='up'),  # 128 x 16 x 16
				ResidualBlock(128, 128, 3, resample='up'),  # 128 x 32 x 32
				torch.nn.BatchNorm2d(128),
				torch.nn.ReLU(inplace=True),
				torch.nn.Conv2d(128, 3, 3, padding=(3-1)//2),     # 3 x 32 x 32
				torch.nn.Tanh()
			)

		def forward(self, z):
			img = self.model(z)
			return img

	class DCGAN_D(torch.nn.Module):
		def __init__(self):
			super(DCGAN_D, self).__init__()
			n_output = 128
			'''
			This is a parameter but since we experiment with a single size
			of 3 x 32 x 32 images, it is hardcoded here.
			'''

			self.DiscBlock1 = DiscBlock1(n_output)                      # 128 x 16 x 16

			self.model = torch.nn.Sequential(
				ResidualBlock(n_output, n_output, 3, resample='down', bn=False, spatial_dim=16),  # 128 x 8 x 8
				ResidualBlock(n_output, n_output, 3, resample=None, bn=False, spatial_dim=8),    # 128 x 8 x 8
				ResidualBlock(n_output, n_output, 3, resample=None, bn=False, spatial_dim=8),    # 128 x 8 x 8
				torch.nn.ReLU(inplace=True),
			)
			self.l1 = torch.nn.Sequential(torch.nn.Linear(128, 1))                  # 128 x 1

		def forward(self, x):
			# x = x.view(-1, 3, 32, 32)
			y = self.DiscBlock1(x)
			y = self.model(y)
			y = y.view(x.size(0), 128, -1)
			y = y.mean(dim=2)
			out = self.l1(y).unsqueeze_(1).unsqueeze_(2) # or *.view(x.size(0), 128, 1, 1, 1)
			return out.view(-1)

## Initialization
G = DCGAN_G()
D = DCGAN_D()

# Initialize weights
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		# Estimated variance, must be around 1
		m.weight.data.normal_(1.0, 0.02)
		# Estimated mean, must be around 0
		m.bias.data.fill_(0)
if param.arch < 2:
	G.apply(weights_init)
	D.apply(weights_init)
	print("Initialized weights")
	print("Initialized weights", file=log_output)

# Criterion
BCE_stable = torch.nn.BCEWithLogitsLoss()

# Soon to be variables
x = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size)
x_fake = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size)
y = torch.FloatTensor(param.batch_size)
y2 = torch.FloatTensor(param.batch_size)
# Weighted sum of fake and real image, for gradient penalty
x_both = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size)
z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1)
# Uniform weight
u = torch.FloatTensor(param.batch_size, 1, 1, 1)
# This is to see during training, size and values won't change
z_test = torch.FloatTensor(param.batch_size, param.z_size, 1, 1).normal_(0, 1)
# For the gradients, we need to specify which one we want and want them all
grad_outputs = torch.ones(param.batch_size)
# For when calculating the approximate bias with RaGANs and RcGANs (contains log(2), nothing more)
log_2 = torch.FloatTensor(1)
w_grad = torch.FloatTensor([param.penalty]) # lagrange multipliers if using augmented lagrangian (initialized at given penalty value)

# Everything cuda
if param.cuda:
	G = G.cuda()
	D = D.cuda()
	BCE_stable.cuda()
	x = x.cuda()
	x_fake = x_fake.cuda()
	x_both = x_both.cuda()
	w_grad = w_grad.cuda()
	y = y.cuda()
	y2 = y2.cuda()
	u = u.cuda()
	z = z.cuda()
	z_test = z_test.cuda()
	grad_outputs = grad_outputs.cuda()
	log_2 = log_2.cuda()

# Now Variables
x = Variable(x)
x_fake = Variable(x_fake)
y = Variable(y)
y2 = Variable(y2)
z = Variable(z)
z_test = Variable(z_test)
w_grad = Variable(w_grad, requires_grad=True)

log_2.fill_(2)
log_2 = torch.log(log_2)

# Based on DCGAN paper, they found using betas[0]=.50 better.
# betas[0] represent is the weight given to the previous mean of the gradient
# betas[1] is the weight given to the previous variance of the gradient
optimizerD = torch.optim.Adam(D.parameters(), lr=param.lr_D, betas=(param.beta1, param.beta2), weight_decay=param.weight_decay)
optimizerG = torch.optim.Adam(G.parameters(), lr=param.lr_G, betas=(param.beta1, param.beta2), weight_decay=param.weight_decay)

# exponential weight decay on lr
decayD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=1-param.decay)
decayG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=1-param.decay)

# Load existing models
if param.load is not None:
	checkpoint = torch.load(param.load)
	current_set_images = checkpoint['current_set_images']
	iter_offset = checkpoint['i'] # iter_offset = checkpoint['i']
	G.load_state_dict(checkpoint['G_state'])
	D.load_state_dict(checkpoint['D_state'])
	optimizerG.load_state_dict(checkpoint['G_optimizer'])
	optimizerD.load_state_dict(checkpoint['D_optimizer'])
	decayG.load_state_dict(checkpoint['G_scheduler'])
	decayD.load_state_dict(checkpoint['D_scheduler'])
	z_test.copy_(checkpoint['z_test'])
	del checkpoint
	print(f'Resumed from iteration {current_set_images*param.gen_every}.')
else:
	current_set_images = 0
	iter_offset = 0

print(G)
print(G, file=log_output)
print(D)
print(D, file=log_output)

## Fitting model
for i in range(iter_offset, param.n_iter):

	# Fake images saved
	if i % param.print_every == 0:
		fake_test = G(z_test)
		vutils.save_image(fake_test.data, '%s/images/fake_samples_iter%05d.png' % (base_dir, i), normalize=True)

	for p in D.parameters():
		p.requires_grad = True

	for t in range(param.Diters):

		########################
		# (1) Update D network #
		########################

		D.zero_grad()
		images = random_sample.__next__()
		# Mostly necessary for the last one because if N might not be a multiple of batch_size
		current_batch_size = images.size(0)
		if param.cuda:
			images = images.cuda()
		# Transfer batch of images to x
		x.resize_as_(images).copy_(images)
		del images
		y_pred = D(x)

		if param.show_graph and i == 0:
			# Visualization of the autograd graph
			d = pv.make_dot(y_pred, D.state_dict())
			d.view()

		if param.loss_D in [1,2,3,4]:
			# Train with real data
			y.resize_(current_batch_size).fill_(1)
			if param.loss_D == 1:
				errD_real = BCE_stable(y_pred, y)
			if param.loss_D == 2:
				errD_real = torch.mean((y_pred - y) ** 2)
				#a = torch.abs(y_pred - y)
				#errD_real = torch.mean(a**(1+torch.log(1+a**4)))
			if param.loss_D == 4:
				errD_real = -torch.mean(y_pred)
			if param.loss_D == 3:
				errD_real = torch.mean(torch.nn.ReLU()(1.0 - y_pred))

			# Train with fake data
			z.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
			fake = G(z)
			x_fake.resize_(fake.data.size()).copy_(fake.data)
			y2.resize_(current_batch_size).fill_(0)
			# Detach y_pred from the neural network G and put it inside D
			y_pred_fake = D(x_fake.detach())
			if param.loss_D == 1:
				errD_fake = BCE_stable(y_pred_fake, y2)
			if param.loss_D == 2:
				errD_fake = torch.mean((y_pred_fake) ** 2)
				#a = torch.abs(y_pred_fake - y)
				#errD_fake = torch.mean(a**(1+torch.log(1+a**2)))
			if param.loss_D == 4:
				errD_fake = torch.mean(y_pred_fake)
			if param.loss_D == 3:
				errD_fake = torch.mean(torch.nn.ReLU()(1.0 + y_pred_fake))
			errD = errD_real + errD_fake
			#print(errD)
		else:
			y.resize_(current_batch_size).fill_(1)
			y2.resize_(current_batch_size).fill_(0)
			z.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
			fake = G(z)
			x_fake.resize_(fake.data.size()).copy_(fake.data)
			y_pred_fake = D(x_fake.detach())

			# Relativistic average GANs
			if param.loss_D == 11:
				errD = BCE_stable(y_pred - torch.mean(y_pred_fake), y) + BCE_stable(y_pred_fake - torch.mean(y_pred), y2)
			if param.loss_D == 12:
				if param.no_bias:
					errD = torch.mean((y_pred - torch.mean(y_pred_fake) - y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + y) ** 2) - (torch.var(y_pred, dim=0)+torch.var(y_pred_fake, dim=0))/param.batch_size
				else:
					errD = torch.mean((y_pred - torch.mean(y_pred_fake) - y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + y) ** 2)
			if param.loss_D == 13:
				errD = torch.mean(torch.nn.ReLU()(1.0 - (y_pred - torch.mean(y_pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 + (y_pred_fake - torch.mean(y_pred))))

			# Relativistic centered GANs
			if param.loss_D == 21:
				full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake))/2
				errD = BCE_stable(y_pred - full_mean, y) + BCE_stable(y_pred_fake - full_mean, y2)
			if param.loss_D == 22: 
				full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake))/2
				if param.no_bias:
					errD = torch.mean((y_pred - full_mean - y) ** 2) + torch.mean((y_pred_fake - full_mean + y) ** 2) + (torch.var(y_pred, dim=0)+torch.var(y_pred_fake, dim=0))/(2*param.batch_size)
				else:
					errD = torch.mean((y_pred - full_mean - y) ** 2) + torch.mean((y_pred_fake - full_mean + y) ** 2)
			if param.loss_D == 23:
				full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake))/2
				errD = torch.mean(torch.nn.ReLU()(1.0 - (y_pred - full_mean))) + torch.mean(torch.nn.ReLU()(1.0 + (y_pred_fake - full_mean)))

			# Relativistic paired GANs (Without the MVUE)
			if param.loss_D == 31:
				errD = 2*BCE_stable(y_pred - y_pred_fake, y)
			if param.loss_D == 32:
				errD = 2*torch.mean((y_pred - y_pred_fake - y) ** 2)
			if param.loss_D == 33:
				errD = 2*torch.mean(torch.nn.ReLU()(1.0 - (y_pred - y_pred_fake)))

			if param.loss_D in [41,42,43]:

				# Relativistic paired GANs (MVUE, slower)
				# Creating cartesian product substraction, very demanding sadly O(k^2), where k is the batch size
				grid_x, grid_y = torch.meshgrid([y_pred, y_pred_fake])
				y_pred_subst = (grid_x - grid_y)
				y.resize_(current_batch_size,current_batch_size).fill_(1)

				if param.loss_D == 41:
					errD = 2*BCE_stable(y_pred_subst, y)
				if param.loss_D == 42:
					errD = 2*torch.mean((y_pred_subst - y) ** 2)
				if param.loss_D == 43:
					errD = 2*torch.mean(torch.nn.ReLU()(1.0 - y_pred_subst))

			errD_real = errD
			errD_fake = errD

		errD.backward(retain_graph=True)

		if (param.loss_D in [4] or param.grad_penalty) and (not param.no_grad_penalty):
			# Gradient penalty
			u.resize_(current_batch_size, 1, 1, 1)
			u.uniform_(0, 1)
			if param.real_only:
				x_both = x.data
			elif param.fake_only:
				x_both = x_fake.data
			else:
				x_both = x.data*u + x_fake.data*(1-u)
			if param.cuda:
				x_both = x_both.cuda()
					
			# We only want the gradients with respect to x_both
			x_both = Variable(x_both, requires_grad=True)
			y0 = D(x_both)
			grad = torch.autograd.grad(outputs=y0,
					inputs=x_both, grad_outputs=grad_outputs,
					retain_graph=True, create_graph=True,
					only_inputs=True)[0]
			x_both.requires_grad_(False)
			sh = grad.shape
			grad = grad.view(current_batch_size,-1)

			if param.l1_margin_no_abs:
				grad_abs = torch.abs(grad)
			else:
				grad_abs = grad
			
			if param.l1_margin:
				grad_norm , _ = torch.max(grad_abs,1)
			elif param.l1_margin_smoothmax:
				grad_norm = torch.sum(grad_abs*torch.exp(param.smoothmax*grad_abs))/torch.sum(torch.exp(param.smoothmax*grad_abs))
			elif param.l1_margin_logsumexp:
				grad_norm = torch.logsumexp(grad_abs,1)
			elif param.linf_margin:
				grad_norm = grad.norm(1,1)
			else:
				grad_norm = grad.norm(2,1)

			if param.penalty_type == 'squared-diff':
				constraint = (grad_norm-1).pow(2)
			elif param.penalty_type == 'clamp':
				constraint = grad_norm.clamp(min=1.) - 1.
			elif param.penalty_type == 'squared-clamp':
				constraint = (grad_norm.clamp(min=1.) - 1.).pow(2)
			elif param.penalty_type == 'squared':
				constraint = grad_norm.pow(2)
			elif param.penalty_type == 'TV':
				constraint = grad_norm
			elif param.penalty_type == 'abs':
				constraint = torch.abs(grad_norm-1)
			elif param.penalty_type == 'hinge':
				constraint = torch.nn.ReLU()(grad_norm - 1)
			elif param.penalty_type == 'hinge2':
				constraint = (torch.nn.ReLU()(grad_norm - 1)).pow(2)
			else:
				raise ValueError('penalty type %s is not valid'%param.penalty_type)

			if param.reduction == 'mean':
				constraint = constraint.mean()
			elif param.reduction == 'max':
				constraint = constraint.max()
			elif param.reduction == 'softmax':
				sm = constraint.softmax(0)
				constraint = (sm*constraint).sum()
			else:
				raise ValueError('reduction type %s is not valid'%param.reduction)
			if param.print_grad:
				print(constraint)
				print(constraint, file=log_output)

			if param.grad_penalty_aug:
				grad_penalty = (-w_grad*constraint + (param.rho/2)*(constraint)**2)
				grad_penalty.backward(retain_graph=True)
			else:
				grad_penalty = param.penalty*constraint
				grad_penalty.backward(retain_graph=True)
		else:
			grad_penalty = 0.

		optimizerD.step()
		# Augmenten Lagrangian
		if param.grad_penalty_aug:
			w_grad.data += param.rho * w_grad.grad.data
			w_grad.grad.zero_()


	########################
	# (2) Update G network #
	########################

	# Make it a tiny bit faster
	for p in D.parameters():
		p.requires_grad = False

	for t in range(param.Giters):

		G.zero_grad()
		if param.resample == 1:
			y.resize_(current_batch_size).fill_(1)
			z.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
			fake = G(z)

			if param.loss_D not in [1, 2, 3, 4]:
				images = random_sample.__next__()
				current_batch_size = images.size(0)
				if param.cuda:
					images = images.cuda()
				x.resize_as_(images).copy_(images)
				del images

		y_pred_fake = D(fake)
		y2.resize_(current_batch_size).fill_(0)
		if param.loss_D == 1:
			errG = BCE_stable(y_pred_fake, y)
		if param.loss_D == 2:
			errG = torch.mean((y_pred_fake - y) ** 2)
		if param.loss_D == 4:
			errG = -torch.mean(y_pred_fake)
		if param.loss_D == 3:
			errG = -torch.mean(y_pred_fake)

		# Relativistic average GANs
		if param.loss_D == 11:
			y_pred = D(x)
			errG = BCE_stable(y_pred - torch.mean(y_pred_fake), y2) + BCE_stable(y_pred_fake - torch.mean(y_pred), y)
		if param.loss_D == 12:
			y_pred = D(x)
			if param.no_bias:
				errG = torch.mean((y_pred - torch.mean(y_pred_fake) + y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - y) ** 2) - (torch.var(y_pred_fake, dim=0)/param.batch_size)
			else:
				errG = torch.mean((y_pred - torch.mean(y_pred_fake) + y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - y) ** 2)
		if param.loss_D == 13:
			y_pred = D(x)
			errG = torch.mean(torch.nn.ReLU()(1.0 + (y_pred - torch.mean(y_pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 - (y_pred_fake - torch.mean(y_pred))))

		if param.loss_D == 21:
			y_pred = D(x)
			full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake))/2
			errG = BCE_stable(y_pred - full_mean, y2) + BCE_stable(y_pred_fake - full_mean, y)
		if param.loss_D == 22: # (y_hat-1)^2 + (y_hat+1)^2
			y_pred = D(x)
			full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake))/2
			if param.no_bias:
				errG = torch.mean((y_pred - full_mean + y) ** 2) + torch.mean((y_pred_fake - full_mean - y) ** 2) + (torch.var(y_pred_fake, dim=0)/(2*param.batch_size))
			else:
				errG = torch.mean((y_pred - full_mean + y) ** 2) + torch.mean((y_pred_fake - full_mean - y) ** 2)
		if param.loss_D == 23:
			y_pred = D(x)
			full_mean = (torch.mean(y_pred) + torch.mean(y_pred_fake))/2
			errG = torch.mean(torch.nn.ReLU()(1.0 + (y_pred - full_mean))) + torch.mean(torch.nn.ReLU()(1.0 - (y_pred_fake - full_mean)))

		# Relativistic paired GANs (Without the MVUE)
		if param.loss_D == 31:
			y_pred = D(x)
			errG = BCE_stable(y_pred_fake - y_pred, y)
		if param.loss_D == 32: # (y_hat-1)^2 + (y_hat+1)^2
			y_pred = D(x)
			errG = torch.mean((y_pred_fake - y_pred - y) ** 2)
		if param.loss_D == 33:
			y_pred = D(x)
			errG = torch.mean(torch.nn.ReLU()(1.0 - (y_pred_fake - y_pred)))

		if param.loss_D in [41,42,43]:

			# Relativistic paired GANs
			# Creating cartesian product substraction, very demanding sadly O(k^2), where k is the batch size
			y_pred = D(x)
			grid_x, grid_y = torch.meshgrid([y_pred, y_pred_fake])
			y_pred_subst = grid_y - grid_x
			y.resize_(current_batch_size,current_batch_size).fill_(1)

			if param.loss_D == 41:
				errG = 2*BCE_stable(y_pred_subst, y)
			if param.loss_D == 42: # (y_hat-1)^2 + (y_hat+1)^2
				errG = 2*torch.mean((y_pred_subst - y) ** 2)
			if param.loss_D == 43:
				errG = 2*torch.mean(torch.nn.ReLU()(1.0 - y_pred_subst))

		errG.backward()
		D_G = y_pred_fake.data.mean()
		optimizerG.step()
	decayD.step()
	decayG.step()

	# Log results so we can see them in TensorBoard after
	#log_value('Diff', -(errD.data.item()+errG.data.item()), i)
	#log_value('errD', errD.data.item(), i)
	#log_value('errG', errG.data.item(), i)

	if (i+1) % param.print_every == 0:
		end = time.time()
		fmt = '[%d] Diff: %.4f loss_D: %.4f loss_G: %.4f time:%.4f'
		s = fmt % (i, -errD.data.item()+errG.data.item(), errD.data.item(), errG.data.item(), end - start)
		print(s)
		print(s, file=log_output)

	# Evaluation metrics
	if (i+1) % param.gen_every == 0:

		current_set_images += 1

		# Save models
		if param.save:
			if not os.path.exists('%s/models/' % (param.logs_folder)):
				os.makedirs('%s/models/' % (param.logs_folder))
			torch.save({
				'i': i + 1,
				'current_set_images': current_set_images,
				'G_state': G.state_dict(),
				'D_state': D.state_dict(),
				'G_optimizer': optimizerG.state_dict(),
				'D_optimizer': optimizerD.state_dict(),
				'G_scheduler': decayG.state_dict(),
				'D_scheduler': decayD.state_dict(),
				'z_test': z_test,
			}, '%s/models/state_%02d.pth' % (param.logs_folder, current_set_images))
			s = 'Models saved'
			print(s)
			print(s, file=log_output)

		# Delete previously existing images
		if os.path.exists('%s/%01d/' % (param.extra_folder, current_set_images)):
			for root, dirs, files in os.walk('%s/%01d/' % (param.extra_folder, current_set_images)):
				for f in files:
					os.unlink(os.path.join(root, f))
		else:
			os.makedirs('%s/%01d/' % (param.extra_folder, current_set_images))

		# Generate 50k images for FID/Inception to be calculated later (not on this script, since running both tensorflow and pytorch at the same time cause issues)
		ext_curr = 0
		z_extra = torch.FloatTensor(100, param.z_size, 1, 1)
		if param.cuda:
			z_extra = z_extra.cuda()
		for ext in range(int(param.gen_extra_images/100)):
			fake_test = G(Variable(z_extra.normal_(0, 1)))
			for ext_i in range(100):
				vutils.save_image((fake_test[ext_i].data*.50)+.50, '%s/%01d/fake_samples_%05d.png' % (param.extra_folder, current_set_images,ext_curr), normalize=False, padding=0)
				ext_curr += 1
		del z_extra
		del fake_test
		# Later use this command to get FID of first set:
		# python fid.py "/home/alexia/Output/Extra/01" "/home/alexia/Datasets/fid_stats_cifar10_train.npz" -i "/home/alexia/Inception" --gpu "0"
end = time.time()
fmt = 'Total time: [%.4f]' % (end - start)
print(fmt)
print(fmt, file=log_output)