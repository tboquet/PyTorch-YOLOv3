#!/bin/bash

cd /weights
wget https://pjreddie.com/media/files/yolov3.weights
cp /srv/app/config/yolov3.cfg /config
