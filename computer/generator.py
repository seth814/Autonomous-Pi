'''
Data generator reference:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
You only need to change __data_generation method to navigate where images are stored.
'''

import numpy as np
import keras
import cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(66,200), n_channels=3,
                 shuffle=True, n_classes=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            im = cv2.imread('images/color/image_' + ID + '.jpeg', 1)
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32).reshape(*self.dim, 3)
            X[i,] = rgb

            # Store class
            y[i] = self.labels[ID]
            #keras.utils.to_categorical(y, num_classes=self.n_classes)
            
        return X, y
