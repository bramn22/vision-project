import numpy as np
from matplotlib import pyplot as plt

def normalize_data(data_raw):
    """ Expects data to have shape: (batch_size, ...) """
    shape = np.shape(data_raw)
    data_reshaped = np.reshape(data_raw, (shape[0], -1))

    mean = np.mean(data_reshaped, axis=0)
    std = np.std(data_reshaped, axis=0)
    data = (data_reshaped - mean)/std

    return np.reshape(data, shape), mean, std

def denormalize_data(data, mean, std):
    shape = np.shape(data)
    data_reshaped = np.reshape(data, (shape[0], -1))

    data_raw = data_reshaped*std + mean
    return np.reshape(data_raw, shape)

def onehots_2_labels(vecs, labels):
    indices = np.argmax(vecs, axis=1)
    return [labels[i] for i in indices]

def plot_2dims(vecs, labels):
    import seaborn as sns
    import matplotlib.cm as cm
    sns.set()
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))

    plt.scatter(vecs[:, 0], vecs[:, 1], c=colors)
    # plt.legend(labels=label_txts, loc='lower left')
    plt.show()

def plot_tsne(dim, vecs, labels):
    import seaborn as sns
    import matplotlib.cm as cm
    from sklearn.manifold import TSNE
    sns.set()

    tsne_results = TSNE(n_components=dim, verbose=1).fit_transform(vecs)

    colors = cm.rainbow(np.linspace(0, 1, len(labels)))

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors)
    # plt.legend(loc='lower left')
    plt.show()

def disp_images(imgs, txts=None, cols=10, title='', cmap=None):
    if txts is None:
        txts = ['']*len(imgs)
    if len(imgs) <= cols:
        f, axarr = plt.subplots(1, len(imgs), figsize=(15, 4), dpi=80, sharex='all')
        for i, (img, txt) in enumerate(zip(imgs, txts)):
            axarr[i % cols].imshow(img, cmap=cmap)
            axarr[i % cols].axis('off')
            axarr[i % cols].set_title(txt)
        f.suptitle(title, fontsize=16)
        plt.show()
    else:
        f, axarr = plt.subplots(len(imgs) // cols, cols, figsize=(15, 4), dpi=80, sharex='all')
        for i, (img, txt) in enumerate(zip(imgs, txts)):
            axarr[i // cols][i % cols].imshow(img, cmap=cmap)
            axarr[i // cols][i % cols].axis('off')
            axarr[i // cols][i % cols].set_title(txt)
        f.suptitle(title, fontsize=16)
        plt.show()

