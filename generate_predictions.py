from keras_segmentation.predict import model_from_checkpoint_path

from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model = model_from_checkpoint_path("/home/klz/checkpoints/vgg_unet_1")

out = model.predict_multiple(
    inp_dir="/home/klz/unlabeled_food_images/",
    out_dir="/home/klz/unlabeled_food_annotations/",
    colors=[(0,0,0),(255,255,255)]
)

# import matplotlib.pyplot as plt
# plt.imshow(out)

# print(model.evaluate_segmentation( inp_images_dir="/home/klz/food_test_images/", 
# 									annotations_dir="/home/klz/food_test_annotations/" ) )