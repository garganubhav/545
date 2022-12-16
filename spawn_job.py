import logging, os, time, sys
import torch

def save_model(path, model):
  filename = os.path.join(path, 'checkpoint.pth.tar')
  torch.save(model.state_dict(), filename)

def train(train_queue, model, optimizer, criterion, use_gpu=False, epoch=None):
  for iteration, (inputs, targets) in enumerate(train_queue):
    optimizer.zero_grad()
    if use_gpu:
      inputs = inputs.cuda()
      targets = targets.cuda()
    logits = model((inputs, epoch))
    loss = criterion(logits, targets)

    loss.backward()
    optimizer.step()

  return loss.mean().item()

def spawn_job(components):

  parameters = components["parameters"]
  scheduler = components["scheduler"]

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p %Z')
  logging.getLogger().setLevel(logging.INFO)

  output_path = parameters["output_dir"] + parameters["trial_name"] + "/"
  if not os.path.exists(output_path):
    os.mkdir(output_path)
  
  log_file = output_path + 'exp.log'
  if not os.path.exists(log_file):
    open(log_file, 'a').close()

  fh = logging.FileHandler(os.path.join(log_file))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)



  for epoch in range(parameters["epochs"]):

    scheduler.step()
    logging.info("Epoch number {}".format(epoch))

    loss = train(components["train_queue"], components["model"],
      components["optimizer"], components["criterion"],
      parameters["use_gpu"], torch.tensor(epoch))

    logging.info("loss is {}".format(loss))
    if epoch % parameters["save_frequency"] == 0:
      save_model(output_path, components["model"])
