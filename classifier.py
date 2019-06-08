from keras.models import Model
from keras.layers import Dense


class Classifier:

    def __init__(self, n_classes, encoder, freeze=False):
        if encoder is None:
            raise ValueError('This model requires an encoder. None was given!')

        if freeze:
            for layer in encoder.layers:
                layer.trainable = False
        self.input = encoder.input
        self.output = Dense(n_classes, activation='softmax')(encoder.output)

        self.model = Model(inputs=self.input, outputs=self.output)

    def build_model(self):
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy')

    def train(self, x, y, epochs=1000):
        self.model.fit(x, y, epochs=epochs, verbose=2, batch_size=32)

    def save_weights(self):
        self.model.save_weights('classifier.h5')

    def load_weights(self):
        self.model.load_weights('classifier.h5')
