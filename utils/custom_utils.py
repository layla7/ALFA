from tsnecuda import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

import torch
import torch.nn as nn


class TSNE_logger:
    def __init__(self, num_classes=None, **kwargs):
        self.num_classes = num_classes
        self.tsne = TSNE(**kwargs)

    def run(self, features, labels=None, plot=False):
        embeddings = self.tsne.fit_transform(features)
        if plot:
            self.plot(embeddings, labels, self.num_classes)
        
    def plot(self, embeddings, labels, num_classes):
        vis_x = embeddings[:, 0]
        vis_y = embeddings[:, 1]
        plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap('jet', num_classes), marker='.')
        plt.colorbar(ticks=range(num_classes))
        #plt.clim(-0.5, 9.5)
        plt.show()

        
class PCA_logger:
    def __init__(self, num_classes=None, **kwargs):
        self.num_classes = num_classes
        self.pca = PCA(**kwargs)

    def run(self, features, labels=None, plot=False):
        embeddings = self.pca.fit_transform(features)
        if plot:
            self.plot(embeddings, labels, self.num_classes)
        
    def plot(self, embeddings, labels, num_classes):
        vis_x = embeddings[:, 0]
        vis_y = embeddings[:, 1]
        plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap('jet', num_classes), marker='.')
        plt.colorbar(ticks=range(num_classes))
        #plt.clim(-0.5, 9.5)
        plt.show()

        
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

        
if __name__ == '__main__':
    from sklearn.datasets import load_digits
    import numpy as np

    def softmax(v):
        e = np.exp(v)
        return e / e.sum()
    
    digits = load_digits()
    logger = TSNE_logger(10, perplexity=10, learning_rate=10)
    logger.run(digits.data, digits.target, True)

    logger = PCA_logger(10)
    logger.run(softmax(digits.data), digits.target, True)