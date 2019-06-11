import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, MaxPool2D, Reshape, BatchNormalization
from keras import optimizers

class AutoEncoder:

    def __init__(self, input_shape, dks=4):
        self.input = Input(shape=input_shape)

        encoder = Reshape((*input_shape, 1))(self.input)  #(256,256,1)
        encoder = Conv2D(16, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)  #(128,128,8)
        encoder = Conv2D(16, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)  #(128,128,8)
        encoder = Conv2D(16, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)  #(32,32,8)
        encoder = Conv2D(16, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)  #(32,32,8)

        code = Conv2D(16, (5, 5), strides=(2, 2), activation='linear', padding='same')(encoder)

        self.encoder = Model(inputs=self.input, outputs=code)

        decoder = Conv2DTranspose(16, (dks, dks), strides=2, activation='elu', padding='same')(code)  #(128,128,8)
        decoder = Conv2DTranspose(16, (dks, dks), strides=2, activation='elu', padding='same')(decoder)  # (128,128,8)
        decoder = Conv2DTranspose(16, (dks, dks), strides=2, activation='elu', padding='same')(decoder)  #(128,128,8)
        decoder = Conv2DTranspose(16, (dks, dks), strides=2, activation='elu', padding='same')(decoder)  #(128,128,8)
        decoder = Conv2DTranspose(1, (dks, dks), strides=2, activation='linear', padding='same')(decoder)  #(256,256,1)

        self.output = Reshape(input_shape)(decoder)
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

    def save_weights(self, name='autoencoder.h5'):
        self.model.save_weights(name)

    def load_weights(self, name='autoencoder.h5'):
        self.model.load_weights(name)

    def extract_features(self, x):
        return self.encoder.predict(x)
