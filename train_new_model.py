from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50

from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

pretrained_model = pspnet_50_ADE_20K()

new_model = pspnet_50( n_classes=2 )

transfer_weights( pretrained_model , new_model  ) # transfer weights from pre-trained model to your model

new_model.train(
    train_images =  "/home/klz/food_training_images/",
    train_annotations = "/home/klz/food_training_annotations/",
    checkpoints_path = "/home/klz/checkpoints/vgg_unet_1" , epochs=5
)