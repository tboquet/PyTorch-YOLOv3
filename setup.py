from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages

print([splitext(basename(path))[0] for path in glob('./*.py')])
print(find_packages('yolov3'))

setup(
    name='yolov3',
    version='0.1dev',
    packages=find_packages('.'),
    package_dir={'': '.'},
    py_modules=[splitext(basename(path))[0] for path in glob('./*.py')],
    license='',
    long_description='yolov3',
    entry_points={
        'console_scripts': [
            'yolov3 = yolov3.cli:cli',
        ]
    },
)
