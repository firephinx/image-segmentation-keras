from keras_segmentation.predict import model_from_checkpoint_path

from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model = model_from_checkpoint_path("/home/klz/checkpoints/vgg_unet_1")

out = model.predict_segmentation(
    inp="/home/klz/food_test_images/cooked_steak_9_1_ending_push_image.png",
    out_fname="/home/klz/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

print(model.evaluate_segmentation( inp_images_dir="/home/klz/food_test_images/", 
									annotations_dir="/home/klz/food_test_annotations/" ) )