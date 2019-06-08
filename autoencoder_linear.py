import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from utils import normalize_data

class AutoEncoder:

    def __init__(self, input_shape, pcs):
        dim = input_shape[0]
        self.input = Input(shape=input_shape)
        # encoder = Dense(250)(input)
        code = Dense(pcs, activation='linear')(self.input)
        # decoder = Dense(250)(code)
        self.output = Dense(dim, activation='linear')(code)
        self.model = Model(inputs=self.input, outputs=self.output)

    def build_model(self):
        self.model = Model(inputs=self.input, outputs=self.output)
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss='mse')

    def train(self, x_raw):
        x, mean, std = normalize_data(x_raw)
        self.model.fit(x, x, epochs=1000, verbose=2, batch_size=64)

    def save_weights(self):
        self.model.save_weights('autoencoder_linear.h5')

    def load_weights(self):
        self.model.load_weights('autoencoder_linear.h5')

    def extract_features(self, x):
        pass
