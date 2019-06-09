import pickle
import numpy as np
from pca import PCA
from autoencoder_linear import AutoEncoder as LinearAutoEncoder
from autoencoder_nonlinear import AutoEncoder as NonLinearAutoEncoder
from classifier import Classifier
import utils
import seaborn as sns
from matplotlib import pyplot as plt


def mse(y_true, y_predict):
    return ((y_true - y_predict) ** 2).mean(axis=None)

data = pickle.load(open("pickles/classification_gray.p", "rb"))
x_train, y_train, x_val, y_val = data.values()
print("x_train: {}, y_train: {}, x_val: {}, y_val: {}".format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))

utils.disp_images(x_train[-5:], y_train[-5:], title="disp", cmap='gray')
# utils.disp_images(np.concatenate([x_train[-5:], y_train[-5:]]), title="disp", cols=5)

# Flatten data
n, w, h = x_train.shape
x_train_flat = np.reshape(x_train, (n, w*h))

# Run PCA
x_train_norm, train_mean, train_std = utils.normalize_data(x_train_flat)
pca = PCA(x_train_norm)
recon_norm = pca.get_reconstruction(pcs=10)
recon_pca = np.reshape(utils.denormalize_data(recon_norm, train_mean, train_std), (n, w, h))
mse_pca = mse(x_train_norm, recon_norm)
print(mse_pca)
mse_pca = mse(x_train, recon_pca)
print(mse_pca)

features = pca.extract_pcs(pcs=2)
# labels = utils.onehots_2_labels(y_train, ['aeroplane', 'car', 'chair', 'dog', 'bird'])
utils.plot_2dims(features, y_train)
utils.plot_tsne(2, features, y_train)

# Strange better performing linear autoencoder (becomes convolutional??)
# x_train_flat = np.reshape(x_train, (n, w, h))
# x_train_norm, train_mean, train_std = utils.normalize_data(x_train_flat)
# ae = LinearAutoEncoder(input_shape=(w, h), pcs=100)
# ae.build_model()
# ae.train(x_train_norm, epochs=20)
# recon = ae.predict(x_train_norm)*np.reshape(train_std, (w,h)) + np.reshape(train_mean, (w, h))

# Run linear AutoEncoder
x_train_flat = np.reshape(x_train, (n, w*h))
x_train_norm, train_mean, train_std = utils.normalize_data(x_train_flat)
ae = LinearAutoEncoder(input_shape=(w*h,), pcs=100)
ae.build_model()
history = ae.train(x_train_norm, epochs=10)
recon_lae = np.reshape(utils.denormalize_data(ae.predict(x_train_norm), train_mean, train_std), (n, w, h))
utils.disp_images(np.concatenate([x_train[-5:], recon_pca[-5:], recon_lae[-5:]]), title="disp", cols=5, cmap='gray')

plt.plot(history.epoch, history.history['mean_squared_error'])
plt.plot(history.epoch, [mse_pca]*len(history.epoch))
plt.show()
plt.close()


# Run non-linear AutoEncoder
x_train_norm, train_mean, train_std = utils.normalize_data(x_train)
ae = NonLinearAutoEncoder(input_shape=(w, h), pcs=50)
# ae.load_weights()
ae.build_model()
ae.train(x_train_norm, epochs=15)
ae.save_weights()
result = ae.predict(x_train_norm[:5])
result = utils.denormalize_data(result, train_mean, train_std)
utils.disp_images(np.concatenate([x_train[:5], result]), title="disp", cols=5, cmap='gray')



# Run classifier with frozen pretrained layers
# x_train_norm, train_mean, train_std = normalize_data(x_train)
# ae = NonLinearAutoEncoder(input_shape=(w, h), pcs=50)
# ae.load_weights()
# vecs = ae.extract_features(x_train_norm)
# plot_tsne(2, vecs[:300], y_train[:300])
# cp = Classifier(5, ae.encoder, freeze=True)
# cp.build_model()
# cp.train(x_train_norm, y_train, epochs=10)
#
# ae = NonLinearAutoEncoder(input_shape=(w, h), pcs=50)
# classifier_new = Classifier(ae.code, freeze=False)
