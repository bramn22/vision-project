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

def plot_tsne(dim, vecs, labels):
    import seaborn as sns
    import matplotlib.cm as cm
    from sklearn.manifold import TSNE


    tsne_results = TSNE(n_components=dim, verbose=1).fit_transform(vecs)

    # colors = [int(l % 23) for l in labels]
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, label=labels)
    plt.legend(loc='lower left')
    plt.show()

    # tsne_results.append()
    # facet = sns.lmplot(data=tsne_results, x='x', y='y', hue='label',
    #                    fit_reg=False, legend=True, legend_out=True)

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