from matplotlib import pyplot as plt
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
from model import Deeplabv3, preprocess_input, pad_image

# deeplab_model = Deeplabv3(backbone='mobilenetv2',OS=16)
deeplab_model = Deeplabv3(backbone='xception',OS=16)
pic_path = "imgs/image3.jpg"
x = plt.imread(pic_path)
x = preprocess_input(x)
x, pad_x = pad_image(x)
y = deeplab_model.predict(x)
labels = np.argmax(y.squeeze(),-1)
plt.imshow(labels[:-pad_x])
plt.show()
np.save('outputs/labels3.npy', labels[:-pad_x])
