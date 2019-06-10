import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, MaxPool2D, Reshape, BatchNormalization
from keras import optimizers

class AutoEncoder:

    def __init__(self, input_shape, last_activation='linear'):
        self.input = Input(shape=input_shape)

        encoder = Reshape((*input_shape, 1))(self.input)  #(256,256,1)
        encoder = Conv2D(8, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)  #(128,128,8)
        encoder = BatchNormalization()(encoder)
        encoder = Conv2D(8, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)  #(128,128,8)
        encoder = BatchNormalization()(encoder)


        # encoder = MaxPool2D((2, 2), padding='same')(encoder)  #(64,64,8)

        # encoder = Conv2D(8, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)
        encoder = Conv2D(16, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)  #(32,32,8)
        encoder = BatchNormalization()(encoder)
        encoder = Conv2D(16, (5, 5), strides=(2, 2), activation='elu', padding='same')(encoder)  #(32,32,8)
        encoder = BatchNormalization()(encoder)

        # encoder = MaxPool2D((2, 2), padding='same')(encoder)  #(16,16,16)
        code = Conv2D(16, (5, 5), strides=(2, 2), activation='linear', padding='same')(encoder)

        # code = Flatten()(encoder)  #(4096)
        # code = Dense(pcs, name='code', activation='linear')(code)  #(pcs)
        self.encoder = Model(inputs=self.input, outputs=code)

        # decoder = Dense(4096)(code)  #(65536)
        # decoder = Reshape((16, 16, 16))(code)  #(64,64,16)
        decoder = Conv2DTranspose(16, (4, 4), strides=2, activation='elu', padding='same')(code)  #(128,128,8)
        decoder = BatchNormalization()(decoder)
        decoder = Conv2DTranspose(16, (4, 4), strides=2, activation='elu', padding='same')(decoder)  # (128,128,8)
        decoder = BatchNormalization()(decoder)

        # decoder = Conv2DTranspose(8, (4, 4), strides=2, activation='elu', padding='same')(decoder)  # (128,128,8)
        decoder = Conv2DTranspose(8, (4, 4), strides=2, activation='elu', padding='same')(decoder)  #(128,128,8)
        decoder = BatchNormalization()(decoder)
        decoder = Conv2DTranspose(8, (4, 4), strides=2, activation='elu', padding='same')(decoder)  #(128,128,8)
        decoder = BatchNormalization()(decoder)
        decoder = Conv2DTranspose(1, (4, 4), strides=2, activation=last_activation, padding='same')(decoder)  #(256,256,1)

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

    def save_weights(self):
        self.model.save_weights('autoencoder.h5')

    def load_weights(self):
        self.model.load_weights('autoencoder.h5')

    def extract_features(self, x):
        return self.encoder.predict(x)
