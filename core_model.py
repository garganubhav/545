from stubs.pytorch.layers import *
import numpy as np
import torch.nn.functional as F

class SingleLayer(BasicUnit):
  def __init__(self, blocks, algorithm="nad", beta=None, reduction=False):
    super(SingleLayer, self).__init__()
    self.blocks = nn.ModuleList(blocks)
    self.reduction = reduction
    self.algorithm = algorithm
    if self.algorithm == "nad" and reduction == False:
      self.epochs = 0
      self.beta = torch.tensor(beta)
      self.policy_p1 = torch.tensor(1.)

  def pad_image(self, x, padding_type, kernel_size, stride=1, dilation=1):
    if padding_type == 'SAME':
        p0 = ((x.shape[2] - 1) * (stride - 1) + dilation * (kernel_size[0] - 1))# //2
        p1 = ((x.shape[3] - 1) * (stride - 1) + dilation * (kernel_size[1] - 1))# //2
        #print(x.shape, kernel_size, p0, p1)
        input_rows = x.shape[2]
        filter_rows = kernel_size[0]

        x = F.pad(x, [0, p1, 0, p0])
        return x

  def forward(self, inputs):
    outputs = []

    if isinstance(inputs, tuple):
      x, epochs = inputs
    else:
      x = inputs

    if self.reduction:
      for block in self.blocks:
        #print(block(x).shape)
        outputs.append(block(x))
    else:
      if self.algorithm == "plnas":
        # ProxylessNAS sampling
        probs = []
        for block in self.blocks:
          probs.append(block.alpha)
        probs = torch.tensor(probs)
        probs = F.softmax(probs, dim=0)
        indexes = torch.multinomial(probs, 2, replacement=False)
      elif self.algorithm == "nad":
        # Neural Arch. Design sampling
        if torch.tensor(self.epochs) < epochs:
          self.policy_p1 *= self.beta
          self.epochs += 1
        policy = torch.bernoulli(self.policy_p1)
        if policy == 1:
          # Random policy
          indexes = np.sort(np.random.choice(range(len(self.blocks)),
            size=(2), replace=False))
        else:
          # exploitation policy
          probs = []
          for block in self.blocks:
            probs.append(block.alpha)
          probs = torch.tensor(probs)
          probs = F.softmax(probs, dim=0)
          indexes = torch.multinomial(probs, 2, replacement=False)
      elif self.algorithm == "random":
        # Random sampling
        indexes = np.sort(np.random.choice(range(len(self.blocks)),
          size=(2), replace=False))

      k0 = self.blocks[indexes[0]].kernel_size
      k1 = self.blocks[indexes[1]].kernel_size
      x0 = self.pad_image(x, padding_type='SAME', kernel_size=k0)
      x1 = self.pad_image(x, padding_type='SAME', kernel_size=k1)

      outputs.append(self.blocks[indexes[0]](x0))
      outputs.append(self.blocks[indexes[1]](x1))
      '''for block in self.blocks:
        k = block.kernel_size
        img = self.pad_image(x, padding_type='SAME', kernel_size=k)
        outputs.append(block(img))'''

    concat_output = torch.cat(outputs, dim=1)
    if isinstance(inputs, tuple):
      return concat_output, epochs

    return concat_output


def get_main_model(main_model_description, in_channels):
  num_layers = main_model_description["num_layers"]
  dropout = main_model_description["dropout"]
  activation = main_model_description["activation"]
  algorithm = main_model_description["algorithm"]["type"]
  out_channels = in_channels
  reduction_block = main_model_description["reduction_block"]
  layers = []

  if algorithm == "nad":
    epochs = main_model_description["algorithm"]["epochs"]
    final_prob = main_model_description["algorithm"]["final_prob"] 
    beta = torch.exp(torch.log(torch.tensor(final_prob)) / epochs)

  for layer in range(1, num_layers+1):
    block_list = []
    
    arg = {}
    arg["in_channels"] = in_channels
    arg["out_channels"] = in_channels // 2
    arg["learnable"] = True
    arg["activation"] = activation
    arg["dropout"] = dropout
    arg["name"] = "core_layer_{}_".format(layer)

    print("Adding layer {} with channels = {}".format(layer, in_channels))
    for block in main_model_description["blocks"]:

      arg["kernel_size"] = block["kernel_size"]

      if block["block"] == "conv2d":
        block_list.append(ConvLayer(arg))
      elif block["block"] == "conv2d_depth":
        block_list.append(DepthConvLayer(arg))
      elif block["block"] == "maxpool":
        arg["type"] = "max"
        block_list.append(PoolingLayer(arg))
      elif block["block"] == "avgpool":
        arg["type"] = "avg"
        block_list.append(PoolingLayer(arg))

    if algorithm == "nad":
      layers.append(SingleLayer(block_list, algorithm, beta))
    else:    
      layers.append(SingleLayer(block_list, algorithm))

    if layer in main_model_description["reduction_layers"]:
      block_list = []
      for block in reduction_block:
        arg = {}
        arg["in_channels"] = in_channels
        arg["out_channels"] = in_channels
        arg["learnable"] = False
        arg["kernel_size"] = block["kernel_size"]
        arg["stride"] = block["stride"]
        arg["activation"] = activation
        arg["dropout"] = dropout
        arg["name"] = "reduction_layer_"
        
        if block["block"] == "conv2d":
          block_list.append(ConvLayer(arg))
        elif block["block"] == "maxpool":
          arg["type"] = "max"
          block_list.append(PoolingLayer(arg))

      layers.append(SingleLayer(block_list, reduction=True))
      in_channels *= 2

  return layers, in_channels




