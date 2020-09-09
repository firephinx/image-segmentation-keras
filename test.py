import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from keras_segmentation.pretrained import pspnet_101_voc12

model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset

# load any of the 3 pretrained models

out = model.predict_segmentation(
    inp="/home/klz/Documents/playing_with_food/processed_data/carrot/2/2/images/grasp_0.png",
    out_fname="/home/klz/out.png"
)