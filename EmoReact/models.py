from torch import nn
import torch.nn.functional
import torch
from .ops.basic_ops import ConsensusModule, Identity
from .transforms import *
from torch.nn.init import normal, constant
from torch.nn import Parameter
import os


class TSN(nn.Module):
	def __init__(self, num_class, num_segments, modality,
				 base_model='resnet18', new_length=None,
				 consensus_type='avg', before_softmax=True,
				 dropout=0.8, modalities_fusion='cat', embed=False,
				 crop_num=1, partial_bn=True, categorical=True,
				 num_feats=2048,
				 # Temporal Shift Module
				 is_shift=False, shift_div=8, shift_place='blockres', temporal_pool=False):

		super(TSN, self).__init__()
		self.num_feats = num_feats
		self.num_classes = num_class
		self.modality = modality
		self.num_segments = num_segments
		self.reshape = True
		self.before_softmax = before_softmax
		self.dropout = dropout
		self.crop_num = crop_num
		self.consensus_type = consensus_type
		self.categorical = categorical
		self.threshold = torch.nn.Parameter(torch.Tensor([0.5]))
		self.threshold.requires_grad = True
		self.modalities_fusion = modalities_fusion

		self.score_fusion = False

		self.name_base = base_model
		
		if not before_softmax and consensus_type != 'avg':
			raise ValueError("Only avg consensus can be used after Softmax")

		if new_length is None:
			self.new_length = 1 if modality == "RGB" else 5
		else:
			self.new_length = new_length

		print(("""
Initializing TSN with base model: {}.
TSN Configurations:
	input_modality:	 {}
	num_segments:	   {}
	new_length:		 {}
	consensus_module:   {}
	dropout_ratio:	  {}
		""".format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))
		
		# Temporal Shift Module
		self.is_shift = is_shift
		self.shift_div = shift_div
		self.shift_place = shift_place
		self.temporal_pool = temporal_pool
		self.reshape = True
		self._prepare_base_model(base_model)

		feature_dim = self._prepare_tsn(num_class)

		if self.modality == 'Flow':
			print("Converting the ImageNet model to a flow init model")
			self.base_model = self._construct_flow_model(self.base_model)
			print("Done. Flow model ready...")

		elif self.modality == "RGB" and self.new_length > 1 and ("resnet" in self.name_base or "mobilenet_v2" in self.name_base):
			self.base_model = self._construct_rgb_model(self.base_model)
			print("Done. RGB with more length model ready.")

		self.consensus = ConsensusModule(consensus_type)

		if not self.before_softmax:
			self.softmax = nn.Softmax()

		self._enable_pbn = partial_bn
		if partial_bn:
			self.partialBN(True)

	def _prepare_tsn(self, num_class):
		std = 0.001
		feature_dim = 2048
		if isinstance(self.base_model, torch.nn.modules.container.Sequential):
			if self.name_base=='mobilenet':
				feature_dim = 1280
		else:
			feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
			if self.dropout == 0:
				setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
				self.new_fc = None
			else:
				setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))

		if self.categorical:
			if 'resnet' in self.name_base:
				self.new_fc = nn.Linear(2048,num_class)
			elif 'mobilenet' in self.name_base:
				self.new_fc = nn.Linear(1280,num_class)

		return feature_dim


	def _prepare_base_model(self, base_model):
		import torchvision, torchvision.models

		if not os.path.exists('pretrained/'):
			os.mkdir('pretrained/')

		if 'resnet' in base_model or 'resnext' in base_model or 'densenet' in base_model:
			self.base_model = getattr(torchvision.models, base_model)(True)
			modules = list(self.base_model.children())[:-1]  # delete the last fc layer.
		
			# Temporal Shift Module
			if self.is_shift:
				print('Adding temporal shift...')
				from EmoReact.temporal_shift import make_temporal_shift
				make_temporal_shift(self.base_model, self.num_segments,
									n_div=self.shift_div, place=self.shift_place, 
									temporal_pool=self.temporal_pool)

			if self.is_shift:
				if not os.path.isfile('pretrained/TSM_adjusted_resnet50_best.pth'):
					os.system('gdown 1I3Q9pOAxiFEteF72HeijC8Ef_8ECef3e')
					os.system('mv TSM_adjusted_resnet50_best.pth pretrained/')
				cp = torch.load('pretrained/TSM_adjusted_resnet50_best.pth')
			else:
				if not os.path.isfile('pretrained/adjusted_resnet50_best.pth'):
					os.system('gdown 1ePxXSUZkQ6Vtzcs5fZBPXwpMUV5FRkKn')
					os.system('mv adjusted_resnet50_best.pth pretrained/')
				cp = torch.load('pretrained/adjusted_resnet50_best.pth')
				
			self.base_model = nn.Sequential(*modules)
		
			if cp != None:
				cp = cp['state_dict']
				newcp = {}
				for key in cp.keys():
					newcp[key.replace("features.","")] = cp[key]
		
				unwanted = ['classifier.0.weight', 'classifier.0.bias', 'categorical_layer.weight', 'categorical_layer.bias', 'continuous_layer.weight', 'continuous_layer.bias']
							#"module.threshold", "module.new_fc.weight", "module.new_fc.bias"]
				for key in unwanted:
					if key in newcp.keys():
						newcp.pop(key)
				self.base_model.load_state_dict(newcp,strict=True)


		elif 'mobilenet' in base_model:
			self.base_model = torch.hub.load('pytorch/vision:v0.10.0', base_model, pretrained=True)
			modules = list(self.base_model.children())[:-1]  # delete the last fc layer.
			modules.append(torch.nn.AdaptiveAvgPool2d(output_size=(1,1)))
			self.base_model = nn.Sequential(*modules)
			
			if not os.path.isfile('pretrained/mobilenet_face.pth'):
				os.system('gdown 1N0QARm9Zt5TiJ8mBkoZjCLlqm5uqVTqa')
				os.system('mv mobilenet_face.pth pretrained/')

			cp = torch.load('pretrained/mobilenet_face.pth')
			if cp != None:
				cp = cp['state_dict']
				newcp = {}
				for key in cp.keys():
					newcp[key.replace("module.features.","").replace("module.classifier.0", "3")] = cp[key]
		
				unwanted = ["3.weight", "3.bias"]
				for key in unwanted:
					if key in newcp.keys():
						newcp.pop(key)
				self.base_model.load_state_dict(newcp,strict=True)
		
		self.base_model.last_layer_name = 'fc'
		self.input_size = 224
		self.input_mean = [0.485, 0.456, 0.406]
		self.input_std = [0.229, 0.224, 0.225]

	def train(self, mode=True):
		"""
		Override the default train() to freeze the BN parameters
		:return:
		"""
		super(TSN, self).train(mode)
		count = 0
		if self._enable_pbn:
			print("Freezing BatchNorm2D except the first one.")
			for m in self.base_model.modules():
				if isinstance(m, nn.BatchNorm2d):
					count += 1
					if count >= (2 if self._enable_pbn else 1):
						m.eval()

						# shutdown update in frozen mode
						m.weight.requires_grad = False
						m.bias.requires_grad = False
		   


	def partialBN(self, enable):
		self._enable_pbn = enable

	def get_optim_policies(self, lr):
	
		return [
				{'params': self.features.parameters(), 'lr': lr}
				]
	


	def forward(self, input):
		sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
		
		body = input.view((-1, sample_len) + input.size()[-2:])
		base_out = self.base_model(body).squeeze(-1).squeeze(-1)
		
		outputs = {}
		
		if self.categorical:
			base_out_cat = self.new_fc(base_out)
			
		base_out = base_out_cat
		
		if self.categorical:
			if self.reshape:
				if self.is_shift and self.temporal_pool:
					base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
				else:
					base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

		if self.categorical:
			output = self.consensus(base_out)
			output = output.squeeze(1)
			outputs['categorical'] = output

		return outputs

	def _get_diff(self, input, keep_rgb=False):
		input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
		input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
		if keep_rgb:
			new_data = input_view.clone()
		else:
			new_data = input_view[:, :, 1:, :, :, :].clone()

		for x in reversed(list(range(1, self.new_length + 1))):
			if keep_rgb:
				new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
			else:
				new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

		return new_data

	def _construct_rgb_model(self, base_model):
		# modify the convolution layers
		# Torch models are usually defined in a hierarchical way.
		# nn.modules.children() return all sub modules in a DFS manner
		modules = list(self.base_model.modules())
		first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
		conv_layer = modules[first_conv_idx]
		container = modules[first_conv_idx - 1]

		# modify parameters, assume the first blob contains the convolution kernels
		params = [x.clone() for x in conv_layer.parameters()]
		kernel_size = params[0].size()
		new_kernel_size = kernel_size[:1] + (3 * self.new_length, ) + kernel_size[2:]
		new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

		new_conv = nn.Conv2d(3 * self.new_length, conv_layer.out_channels,
							 conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
							 bias=True if len(params) == 2 else False)
		new_conv.weight.data = new_kernels
		if len(params) == 2:
			new_conv.bias.data = params[1].data # add bias if neccessary
		layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

		# replace the first convolution layer
		setattr(container, layer_name, new_conv)
		return base_model

	def _construct_flow_model(self, base_model):
		# modify the convolution layers
		# Torch models are usually defined in a hierarchical way.
		# nn.modules.children() return all sub modules in a DFS manner
		modules = list(base_model.modules())
		first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
		conv_layer = modules[first_conv_idx]
		container = modules[first_conv_idx - 1]

		# modify parameters, assume the first blob contains the convolution kernels
		params = [x.clone() for x in conv_layer.parameters()]
		kernel_size = params[0].size()
		new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
		new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

		new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
							 conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
							 bias=True if len(params) == 2 else False)
		new_conv.weight.data = new_kernels
		if len(params) == 2:
			new_conv.bias.data = params[1].data # add bias if neccessary
		layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

		# replace the first convolution layer
		setattr(container, layer_name, new_conv)
		return base_model

	

	@property
	def crop_size(self):
		return self.input_size

	@property
	def scale_size(self):
		return self.input_size * 256 // 224

	def get_augmentation(self):
		if self.modality == 'RGB':
			return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1]),
												   GroupRandomHorizontalFlip(is_flow=False)])
		elif self.modality == 'Flow':
			return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
												   GroupRandomHorizontalFlip(is_flow=True)])
		elif self.modality == 'RGBDiff':
			return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
												   GroupRandomHorizontalFlip(is_flow=False)])