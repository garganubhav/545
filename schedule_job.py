import torch
import torch.nn as nn
import torch.optim as optim

from stubs.pytorch.input_pipeline import get_train_test_queue, get_init_model
from stubs.pytorch.core_model import get_main_model
from stubs.pytorch.final_pipeline import get_final_model

from stubs.pytorch.spawn_job import spawn_job

def construct_model(model_specification):
  init_model_description = model_specification["init_model"]
  init_model_layers, in_channels = get_init_model(init_model_description, in_channels=3)

  main_model_description = model_specification["core_model"]
  main_model_layers, in_channels = get_main_model(main_model_description, in_channels)

  final_model_description = model_specification["final_model"]
  final_model_layers = get_final_model(final_model_description, in_channels)

  layers = init_model_layers + main_model_layers + final_model_layers

  model = nn.Sequential(*layers)
  return model

def get_optimizer(model, optimizer_parameters):
  if optimizer_parameters["type"] == "momentum":
    momentum = optimizer_parameters["momentum"]
    optimizer = optim.SGD(model.parameters(), lr=optimizer_parameters["initial_lr"])
  if optimizer_parameters["type"] == "adam":
    optimizer = optim.Adam(model.parameters(), lr=optimizer_parameters["initial_lr"])
  return optimizer

def get_scheduler(optimizer, lr_scheduler_parameters):
  if lr_scheduler_parameters["type"] == "cosine_decay":
    T_max = lr_scheduler_parameters["T_max"]
    min_lr = lr_scheduler_parameters["min"]
    last_epoch = -1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max,
      eta_min=min_lr, last_epoch=last_epoch)
  return scheduler

def get_train_parameters(job_specification):
  parameters = {}
  parameters["trial_name"] = job_specification["trial_parameters"]["trial_name"]
  parameters["epochs"] = job_specification["trial_parameters"]["epochs"]
  parameters["save_frequency"] = job_specification["trial_parameters"]["save_frequency"]
  parameters["output_dir"] = job_specification["trial_parameters"]["output_path"]
  parameters["use_gpu"] = job_specification["system_parameters"]["gpus"]

  return parameters

def schedule_job(job_specification):
  use_gpu = job_specification["system_parameters"]["gpus"]
  hyper_parameters = job_specification["hyper_parameters"]
  components = {}

  train_queue, test_queue = get_train_test_queue(job_specification)
  model_specification = job_specification["model_specification"]
  model = construct_model(model_specification)
  criterion = nn.CrossEntropyLoss()

  if use_gpu:
    model = model.cuda()
    criterion = criterion.cuda()
  
  components["parameters"] = get_train_parameters(job_specification)
  components["train_queue"], components["test_queue"] = train_queue, test_queue
  components["model"] = model
  components["criterion"] = criterion
  components["optimizer"] = get_optimizer(model, hyper_parameters["optimizer"])
  components["scheduler"] = get_scheduler(components["optimizer"], hyper_parameters["lr_scheduler"])


  spawn_job(components)
