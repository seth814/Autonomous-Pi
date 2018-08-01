'''
Trains convolutional neural network based on nvidia's paper:
https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
target scaling is individual to user, so adjust as needed. (line 34)
fit_generator is optional, but as data size increases, it will become nessesary.
'''

import pandas as pd
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight
import keras
from generator import DataGenerator
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.models import Sequential

class History(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []
    
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
    
    def save_history(self):
        losses = {'loss':self.loss}
        df = pd.DataFrame(losses)
        df.to_csv('nvidia_history.csv', index=False)

os.chdir('images')
df = pd.read_csv('target.csv')
df.target = (df.target - 100) / 400
unique = np.unique(df.target)
os.chdir('..')
class_weight = compute_class_weight('balanced',
                                     np.unique(df.target),
                                     df.target)

ids = list(map(str,df.index))
split = int(len(ids)*.95)
partition = {}
partition['train'] = ids[:split]
partition['val'] = ids[split:]
values = list(df.to_dict()['target'].values())
labels = dict(zip(ids, values))

params = {'dim': (66,200),
          'batch_size': 32,
          'n_channels': 3,
          'shuffle': True,
          'n_classes': None}

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['val'], labels, **params)

model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(66,200,3)))
model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')

history = History()
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=20,
                    callbacks=[history],
                    class_weight=class_weight)

os.chdir('models')
model_yaml = model.to_yaml()
with open('nvidia.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)
model.save_weights('nvidia.h5')
history.save_history()
