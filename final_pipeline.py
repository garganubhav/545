import torch.nn as nn
from stubs.pytorch.layers import *

def get_final_model(final_model_description, in_channels):
  layers = []
  for layer in final_model_description["layers"]:
    arg = {"name": "final_layer_{}_".format(layer)}
    if layer["block"] == "avg_pool":
      # global average pooling
      layers.append(GlobalAvgPooling(arg))
    elif layer["block"] == "fc":
      # fully connected layer
      #arg["dropout"] = final_model_description["dropout"]
      arg["out_features"] = layer["outputs"]
      arg["num_features"] = in_channels
      layers.append(FullyConnectedLayer(arg))

  return layers
