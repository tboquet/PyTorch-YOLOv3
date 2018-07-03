from __future__ import division

import time
import datetime

from .models import Darknet
from .utils.parse_config import parse_data_config
from .utils.parse_config import parse_model_config
from .utils.utils import create_dir
from .utils.utils import weights_init_normal
from .utils.datasets import ListDataset

import torch
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def train(input_path,
          model_config_path,
          data_config_path,
          weights_path,
          nb_epochs=30,
          batch_size=16,
          conf_thres=0.8,
          nms_thres=0.4,
          n_cpu=8,
          img_size=416,
          checkpoint_interval=1,
          checkpoint_dir='/checkpoints',
          use_cuda=True,
          verbose=True,
          freeze=False):

    cuda = torch.cuda.is_available() and use_cuda
    device = torch.device("cuda" if cuda else "cpu")
    checkpoint_dir = create_dir(checkpoint_dir)
    # Get data configuration
    data_config = parse_data_config(data_config_path)
    train_path = data_config['train']

    # Get hyper parameters
    hyperparams = parse_model_config(model_config_path)[0]
    learning_rate = float(hyperparams['learning_rate'])
    momentum = float(hyperparams['momentum'])
    decay = float(hyperparams['decay'])

    # Initiate model
    model = Darknet(model_config_path, freeze=freeze)
    if weights_path is not None:
        model.load_weights(weights_path)
    else:
        model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()

    model.train()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        print(model)

    model.to(device)
    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    parameters = [p for p in model.parameters() if p.requires_grad]
    if len(parameters) == 0:
        raise ValueError('Not training, empty parameters list')
    optimizer = optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        dampening=0,
        weight_decay=decay)

    prev_time = time.time()
    for epoch in range(nb_epochs):
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)

            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            loss = model(imgs, targets)

            loss.backward()
            optimizer.step()

            print(
                '[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]'
                % (epoch, nb_epochs, batch_i, len(dataloader), model.losses['x'],
                   model.losses['y'], model.losses['w'], model.losses['h'],
                   model.losses['conf'], model.losses['cls'], loss.item(),
                   model.losses['recall']))
            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print('\t+ Batch %d, training Time: %s' % (batch_i, inference_time))

            model.seen += imgs.size(0)

        if epoch % checkpoint_interval == 0:
            model.save_weights('%s/%d.weights' % (checkpoint_dir, epoch))
