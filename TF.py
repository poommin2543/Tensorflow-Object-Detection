#https://www.udacity.com/blog/2021/06/tensorflow-object-detection.html
# Mandatory imports
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pylab as plt
# We'll use requests and PIL to download and process images
import requests
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

def load_image(image_url):
    img = Image.open(requests.get(image_url, stream=True).raw)
    return img

def resize_image(img):
    # Resize image to be no larger than 1024x1024 pixels while retaining 
    # the image's aspect ratio
    maxsize = (1024, 1024)
    img.thumbnail(maxsize, Image.ANTIALIAS)
    return img

module_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_url).signatures['default']

IMAGE_URL = "https://farm1.staticflickr.com/6188/6082105629_db7abe41b9_o.jpg"

img = load_image(IMAGE_URL)
img = resize_image(img)

numpy_img = np.asarray(img)
print(numpy_img.shape)

plt.imshow(numpy_img)
plt.show()

Convert_img = tf.image.convert_image_dtype(img, tf.float32) 

