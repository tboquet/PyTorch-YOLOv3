"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mmirrorsq` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``mirrorsq.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``mirrorsq.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import click

from . import __version__
from .detect import predict as predict_yolo
from .detect import plot_detection
from .test import test as test_yolo
from .train import train as train_yolo


@click.group()
def cli():
    click.secho('yolov3 version: {}'.format(__version__), bold=True)


@cli.command('predict')
@click.option(
    '--conf_thres',
    type=click.FLOAT,
    default=0.8,
    help='object confidence threshold')
@click.option(
    '--nms_thres',
    type=click.FLOAT,
    default=0.4,
    help='Nms thresshold for non-maximum suppression')
@click.option('--batch_size', type=int, default=1, help='size of the batches')
@click.option(
    '--n_cpu',
    type=click.INT,
    default=8,
    help='number of cpu threads to use during batch generation')
@click.option(
    '--img_size',
    type=click.INT,
    default=416,
    help='size of each image dimension')
@click.option(
    '--use_cuda', is_flag=True, help='whether to use cuda if available')
@click.option('--plot', is_flag=True, help='output images with bounding boxes')
@click.argument(
    'input_path', type=click.Path(exists=True), default='/data/samples')
@click.argument('output_path', type=click.Path(exists=True), default='/output')
@click.argument(
    'config_path', type=click.Path(exists=True), default='/config/yolov3.cfg')
@click.argument(
    'weights_path',
    type=click.Path(exists=True),
    default='/weights/yolov3.weights')
@click.argument(
    'class_path', type=click.Path(exists=True), default='/data/coco.names')
def predict(conf_thres, nms_thres, batch_size, n_cpu, img_size, use_cuda, plot,
            input_path, output_path, config_path, weights_path, class_path):
    imgs, img_detections = predict_yolo(
        input_path=input_path,
        config_path=config_path,
        weights_path=weights_path,
        conf_thres=conf_thres,
        nms_thres=nms_thres,
        batch_size=batch_size,
        n_cpu=n_cpu,
        img_size=img_size,
        use_cuda=use_cuda)
    if plot is True:
        plot_detection(imgs, img_detections, img_size, class_path, output_path)


@cli.command('train')
@click.option('--nb_epochs', type=int, default=30, help='number of epochs')
@click.option(
    '--batch_size', type=int, default=16, help='size of each image batch')
@click.option(
    '--conf_thres',
    type=float,
    default=0.8,
    help='object confidence threshold')
@click.option(
    '--nms_thres',
    type=float,
    default=0.4,
    help='nms thresshold for non-maximum suppression')
@click.option(
    '--n_cpu',
    type=int,
    default=0,
    help='number of cpu threads to use during batch generation')
@click.option(
    '--img_size', type=int, default=416, help='size of each image dimension')
@click.option(
    '--checkpoint_interval',
    type=int,
    default=1,
    help='interval between saving model weights')
@click.option(
    '--use_cuda',
    type=bool,
    is_flag=True,
    help='whether to use cuda if available')
@click.option(
    '--verbose', type=bool, help='Output training information', is_flag=True)
@click.argument('input_path', type=str, default='/data/samples')
@click.argument('model_config_path', type=str, default='/config/yolov3.cfg')
@click.argument('data_config_path', type=str, default='/config/coco.data')
@click.argument('weights_path', type=str, default='')
@click.argument('checkpoint_dir', type=str, default='/checkpoints')
def train(input_path, model_config_path, data_config_path, weights_path,
          checkpoint_dir, nb_epochs, batch_size, conf_thres, nms_thres, n_cpu,
          img_size, checkpoint_interval, use_cuda, verbose):
    if weights_path == '':
        weights_path = None
    if verbose is True:
        click.secho('Parameters:', color='green')
        click.secho('\tBatch_size: {}'.format(batch_size), color='blue')
        click.secho('\tnb epochs: {}'.format(nb_epochs), color='blue')
        click.secho(
            '\tNon maximum supression: {}'.format(nms_thres), color='blue')
        click.secho(
            '\tConfidence treshold: {}'.format(conf_thres), color='blue')
    train_yolo(
        input_path=input_path,
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        weights_path=weights_path,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        conf_thres=conf_thres,
        nms_thres=nms_thres,
        n_cpu=n_cpu,
        img_size=img_size,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        use_cuda=use_cuda,
        verbose=verbose)


@cli.command('test')
@click.option(
    '--batch_size', type=int, default=16, help='size of each image batch')
@click.option(
    '--iou_thres',
    type=float,
    default=0.5,
    help='Iou thresshold for non-maximum suppression')
@click.option(
    '--conf_thres',
    type=float,
    default=0.5,
    help='object confidence threshold')
@click.option(
    '--nms_thres',
    type=float,
    default=0.5,
    help='Nms thresshold for non-maximum suppression')
@click.option(
    '--n_cpu',
    type=int,
    default=0,
    help='number of cpu threads to use during batch generation')
@click.option(
    '--img_size', type=int, default=416, help='size of each image dimension')
@click.option(
    '--use_cuda',
    type=bool,
    is_flag=True,
    help='whether to use cuda if available')
@click.option(
    '--verbose', type=bool, help='Output training information', is_flag=True)
@click.argument('model_config_path', type=str, default='/config/yolov3.cfg')
@click.argument('data_config_path', type=str, default='/config/coco.data')
@click.argument('weights_path', type=str, default='')
def test(model_config_path, data_config_path, weights_path, batch_size,
         iou_thres, conf_thres, nms_thres, n_cpu, img_size, use_cuda, verbose):
    if verbose is True:
        click.secho('Parameters:', color='green')
        click.secho('\tBatch_size: {}'.format(batch_size), color='blue')
        click.secho(
            '\tNon maximum supression: {}'.format(nms_thres), color='blue')
        click.secho(
            '\tConfidence treshold: {}'.format(conf_thres), color='blue')
    mean_aps = test_yolo(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        weights_path=weights_path,
        batch_size=batch_size,
        iou_thres=iou_thres,
        conf_thres=conf_thres,
        nms_thres=nms_thres,
        n_cpu=n_cpu,
        img_size=img_size,
        use_cuda=use_cuda,
        verbose=verbose)
