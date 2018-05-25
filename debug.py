from matplotlib import pyplot as plt
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
from model import Deeplabv3, preprocess_input

deeplab_model = Deeplabv3(OS=16)
pic_path = "imgs/image1.jpg"
x1 = plt.imread(pic_path)
x1 = preprocess_input(x1)
print(type(x1))