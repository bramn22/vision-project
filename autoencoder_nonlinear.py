import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, MaxPool2D, Reshape
from keras import optimizers

class AutoEncoder:

    def __init__(self, input_shape, pcs):
        self.input = Input(shape=input_shape)

        encoder = Conv2D(16, (5, 5), strides=(2, 2), activation='elu', padding='same')(self.input)  #(128,128,8)
        # encoder = MaxPool2D((2, 2), padding='same')(encoder)  #(64,64,8)

        encoder = Conv2D(16, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)
        encoder = Conv2D(16, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)  #(32,32,8)

        # encoder = MaxPool2D((2, 2), padding='same')(encoder)  #(16,16,16)
        encoder = Conv2D(16, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)

        code = Flatten()(encoder)  #(4096)
        code = Dense(pcs, name='code', activation='linear')(code)  #(pcs)
        self.encoder = Model(inputs=self.input, outputs=code)

        decoder = Dense(4096)(code, activation='elu')  #(65536)
        decoder = Reshape((16, 16, 16))(decoder)  #(64,64,16)
        decoder = Conv2DTranspose(16, (4, 4), strides=2, activation='elu', padding='same')(decoder)  #(128,128,8)
        decoder = Conv2DTranspose(16, (4, 4), strides=2, activation='elu', padding='same')(decoder)  #(128,128,8)
        decoder = Conv2DTranspose(16, (4, 4), strides=2, activation='elu', padding='same')(decoder)  #(128,128,8)
        decoder = Conv2DTranspose(1, (4, 4), strides=2, activation='linear', padding='same')(decoder)  #(256,256,1)

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
