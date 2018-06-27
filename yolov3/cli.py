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
# from .train import build_pipeline
# from .train import get_callbacks


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
    help='iou thresshold for non-maximum suppression')
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


# @cli.command('train')
# @click.option('--verbose', is_flag=True)
# @click.option('--batch_size', type=click.INT, required=True, default=32)
# @click.option('--lr', type=click.FLOAT, required=True, default=1e-4)
# @click.option('--architecture', required=True, default='inceptionv3')
# @click.option('--nb_workers', required=True, default=3)
# @click.option('--nb_epochs', required=True, default=20)
# @click.option('--augmentation', is_flag=True)
# @click.argument(
#     'train_dataset_path', type=click.Path(exists=True), required=True)
# @click.argument(
#     'valid_dataset_path', type=click.Path(exists=True), required=True)
# @click.argument('output_path', type=click.Path(exists=False), required=True)
# def train(verbose, batch_size, lr, architecture, nb_workers, nb_epochs,
#           augmentation, train_dataset_path, valid_dataset_path, output_path):
#     verbose_keras = 0
#     pipeline = build_pipeline(train_dataset_path, valid_dataset_path,
#                               batch_size, architecture, augmentation, lr)
#     train_generator = pipeline.module_pipeline['train_generator']
#     valid_generator = pipeline.module_pipeline['valid_generator']
#     nb_iter = int(train_generator.examples / train_generator.batch_size + 1)
#     nb_iter_valid = int(valid_generator.examples / valid_generator.batch_size +
#                         1)
#     if verbose is True:
#         verbose_keras = 1
#         click.secho('Parameters:', color='green')
#         click.secho('\tBatch_size: {}'.format(batch_size), color='blue')
#         click.secho('\tLearning rate: {}'.format(lr), color='blue')
#         click.secho('\tnb epochs: {}'.format(nb_epochs), color='blue')
#         click.secho('\tArchitecture: {}'.format(architecture), color='blue')
#         click.secho(
#             '\tNumber of train examples: {}'.format(train_generator.examples),
#             color='blue')
#         click.secho(
#             '\tNumber of valid examples: {}'.format(valid_generator.examples),
#             color='blue')
#         click.secho(
#             '\tNumber of train iterations: {}'.format(nb_iter), color='blue')
#         click.secho(
#             '\tNumber of valid iterations: {}'.format(nb_iter_valid),
#             color='blue')
#         click.secho('\tNumber of workers: {}'.format(nb_workers), color='blue')

#     splitted_path = output_path.split('.')
#     splitted_path[-2] = '_'.join([
#         splitted_path[-2], str(batch_size), str(lr), str(nb_epochs),
#         architecture
#     ])
#     splitted_path[-1] = '.h5'
#     path = ''.join(splitted_path)
#     ls_sched, early_stop, checkpoint = get_callbacks(
#         path, verbose=verbose_keras)
#     cbks = [ls_sched, early_stop, checkpoint]
#     pipeline.fit(
#         gen=train_generator,
#         validation_data=valid_generator,
#         verbose=verbose_keras,
#         use_multiprocessing=True,
#         workers=nb_workers,
#         epochs=nb_epochs,
#         callbacks=cbks)
