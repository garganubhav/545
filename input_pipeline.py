import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


from stubs.pytorch.layers import ConvLayer, PoolingLayer

class Cutout(object):
  def __init__(self, length):
    self.length = length

  def __call__(self, img):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - self.length // 2, 0, h)
    y2 = np.clip(y + self.length // 2, 0, h)
    x1 = np.clip(x - self.length // 2, 0, w)
    x2 = np.clip(x + self.length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img

def _data_transforms_cifar10(parameters):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if "cutout" in parameters["hyper_parameters"].keys():
    train_transform.transforms.append(Cutout(parameters["hyper_parameters"]["cutout"]["size"]))

  test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, test_transform


def get_train_test_queue(parameters):
  if parameters["trial_parameters"]["dataset"] == "cifar10":
    train_transform, test_transform = _data_transforms_cifar10(parameters)
    
    train_data = dset.CIFAR10(root=parameters["trial_parameters"]["data_dir"],
      train=True, download=True, transform=train_transform)
    train_queue = torch.utils.data.DataLoader(train_data,
      batch_size=parameters["hyper_parameters"]["batch_size"],
      shuffle=True, pin_memory=True, num_workers=2)

    test_data = dset.CIFAR10(root=parameters["trial_parameters"]["data_dir"],
      train=False, download=True, transform=test_transform)
    test_queue = torch.utils.data.DataLoader(test_data,
      batch_size=parameters["hyper_parameters"]["batch_size"],
      shuffle=False, pin_memory=True, num_workers=2)
  
  elif parameters["trial_parameters"]["dataset"] == "Imagenet":
    pass

  elif parameters["trial_parameters"]["dataset"] == "SVHN":
    pass

  return train_queue, test_queue


def get_init_model(init_model_description, in_channels):
  arg = {}
  layers = []

  for num_layer, layer in enumerate(init_model_description["layers"], 1):
    arg = {"name": "init_layer_{}_".format(num_layer)}
    arg["dropout"] = init_model_description["dropout"]
    if layer["block"] == "conv2d":
      arg["kernel_size"] = layer["kernel_size"]
      arg["in_channels"] = in_channels
      arg["out_channels"] = layer["outputs"]
      arg["name"] = "init_"
      if "activation" in layer.keys():
        arg["activation"] = layer["activation"]
      layers.append(ConvLayer(arg))

  return layers, arg["out_channels"]
