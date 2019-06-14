from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping


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

    def build_model(self, lr=0.001):
        self.model.summary()
        adam = optimizers.Adam(lr=lr)
        self.model.compile(optimizer=adam,
                           loss='categorical_crossentropy')

    def train(self, x, y, batch_size=32, epochs=1000, validation_split=0.1, patience=10):
        earlystopping = EarlyStopping(patience=patience, restore_best_weights=True)
        return self.model.fit(x, y, epochs=epochs, verbose=2, batch_size=batch_size, validation_split=validation_split, callbacks=[earlystopping])

    def save_weights(self, name='classifier.h5'):
        self.model.save_weights(name)

    def load_weights(self, name='classifier.h5'):
        self.model.load_weights(name)
