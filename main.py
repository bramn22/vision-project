import pickle
import numpy as np
from pca import PCA
from autoencoder_linear import AutoEncoder as LinearAutoEncoder
from autoencoder_nonlinear import AutoEncoder as NonLinearAutoEncoder
from classifier import Classifier
import utils



data = pickle.load(open("data_gray.p", "rb"))
x_train, y_train, x_val, y_val = data.values()
print("x_train: {}, y_train: {}, x_val: {}, y_val: {}".format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))

# Flatten data
n, w, h = x_train.shape
x_train_flat = np.reshape(x_train, (n, w*h))

# Run PCA
# pca = PCA(x_train_flat)
# pca.get_reconstruction_error(pcs=50)

# Run linear AutoEncoder
# x_train_norm, train_mean, train_std = normalize_data(x_train)
# ae = LinearAutoEncoder(input_shape=(w, h), pcs=50)
# ae.train(x_train_norm)

# Run non-linear AutoEncoder
x_train_norm, train_mean, train_std = utils.normalize_data(x_train)
ae = NonLinearAutoEncoder(input_shape=(w, h), pcs=50)
ae.load_weights()
ae.build_model()
ae.train(x_train_norm, epochs=15)
ae.save_weights()
result = ae.predict(x_train_norm[:5])
utils.denormalize_data(result, train_mean, train_std)
utils.disp_images(np.concatenate([x_train[:5], result]), title="disp", cols=5, cmap='gray')



# Run classifier with frozen pretrained layers
# x_train_norm, train_mean, train_std = normalize_data(x_train)
# ae = NonLinearAutoEncoder(input_shape=(w, h), pcs=50)
# ae.load_weights()
# vecs = ae.extract_features(x_train_norm)
# plot_tsne(2, vecs[:300], y_train[:300])
cp = Classifier(5, ae.encoder, freeze=True)
# cp.build_model()
# cp.train(x_train_norm, y_train, epochs=10)
#
# ae = NonLinearAutoEncoder(input_shape=(w, h), pcs=50)
# classifier_new = Classifier(ae.code, freeze=False)
