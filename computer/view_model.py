import matplotlib.pyplot as plt
import os
import pandas as pd
from keras.models import model_from_yaml
import glob
import random
import cv2
import numpy as np
import time

os.chdir('images')
df = pd.read_csv('target.csv')
plt.hist(df.target, bins=range(100,501))
plt.show()

os.chdir('..')
os.chdir('models')
df = pd.read_csv('nvidia_history.csv')
plt.plot(df.loss[1:])
plt.show()

yaml_file = open('nvidia.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
loaded_model.load_weights('nvidia.h5')
loaded_model.compile(loss='mse', optimizer='adam')

os.chdir('../images/color')
images = glob.glob('*.jpeg')
random.shuffle(images)
images = images[:10]

for im in images:
    image = cv2.imread(im, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32).reshape(1, 66, 200, 3)
    current = time.time()
    after = time.time()
    cv2.imshow('im', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
