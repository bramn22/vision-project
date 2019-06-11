import numpy as np
from keras.backend import mean, square
from keras.models import Model
from keras.layers import Input, Dense
from utils import normalize_data
from keras import optimizers

class AutoEncoder:

    def __init__(self, input_shape, pcs):
        dim = input_shape[0]
        self.input = Input(shape=input_shape)
        code = Dense(pcs, activation='linear')(self.input)
        self.encoder = Model(inputs=self.input, outputs=code)
        self.output = Dense(dim, activation='linear')(code)
        self.model = Model(inputs=self.input, outputs=self.output)

    def build_model(self, lr=0.001):
        self.model.summary()
        adam = optimizers.Adam(lr=lr)
        self.model.compile(optimizer=adam,
                           loss='mse', metrics=['mse'])

    def train(self, x, epochs=1000, batch_size=32):
        return self.model.fit(x, x, epochs=epochs, verbose=2, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)

    def save_weights(self, name='autoencoder_linear.h5'):
        self.model.save_weights(name)

    def load_weights(self, name='autoencoder_linear.h5'):
        self.model.load_weights(name)

    def extract_features(self, x):
        return self.encoder.predict(x)
