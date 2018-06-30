from __future__ import division

from .models import Darknet
from .utils.parse_config import parse_data_config
from .utils.parse_config import parse_model_config
from .utils.utils import weights_init_normal
from .utils.datasets import ListDataset

import os

import torch
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
          verbose=True):
    cuda = torch.cuda.is_available() and use_cuda

    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(data_config_path)
    train_path = data_config['train']

    # Get hyper parameters
    hyperparams = parse_model_config(model_config_path)[0]
    learning_rate = float(hyperparams['learning_rate'])
    momentum = float(hyperparams['momentum'])
    decay = float(hyperparams['decay'])

    # Initiate model
    model = Darknet(model_config_path)
    if weights_path is not None:
        model.load_weights(weights_path)
    else:
        model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()

    model.train()

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        dampening=0,
        weight_decay=decay)

    for epoch in range(nb_epochs):
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)

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

            model.seen += imgs.size(0)

        if epoch % checkpoint_interval == 0:
            model.save_weights('%s/%d.weights' % (checkpoint_dir, epoch))
