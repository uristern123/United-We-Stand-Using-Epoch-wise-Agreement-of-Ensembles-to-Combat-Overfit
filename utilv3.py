import os
import sys
import numpy as np
import scipy
from PIL import Image
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))
class KMeans:
    def __init__(self, n_clusters=8, centroids = None, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = centroids
    def fit(self, X_train):
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        if self.centroids is None:
            self.centroids = [random.choice(X_train)]
            for _ in range(self.n_clusters-1):
                # Calculate distances from points to the centroids
                dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
                # Normalize the distances
                dists /= np.sum(dists)
                # Choose remaining points based on their distances
                new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
                self.centroids += [X_train[new_centroid_idx]]
        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs

class BetaMixture1D(object):
    def __init__(self, max_iters=3000,
                 alphas_init=[2, 5],
                 betas_init=[5, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps
        #print("fitting BMM!")
        for i in tqdm(range(self.max_iters)):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        #plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

class MyDataset(Dataset):
    def __init__(self, data, targets):

        self.data = data
        self.targets = targets
        self.place_holder = "_"

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        return data, self.place_holder, target, index

    def __len__(self):
        return len(self.data)

    def __remove__(self, remove_list):
        data = np.delete(self.data, remove_list, axis=0)
        targets = np.delete(self.targets, remove_list, axis=0)
        return data, targets

class NoiseFilteringDataset(Dataset):
    def __init__(self, dataset, labels):

        self.data = dataset.data
        self.labels = labels
        self.targets = dataset.targets

    def __getitem__(self, index):
        data, target = self.final_data[index], self.final_targets[index]
        return data, target, index

    def __len__(self):
        return len(self.final_data)

    def __remove__(self, remove_list):
        data = np.delete(self.data, remove_list, axis=0)
        targets = np.delete(self.targets, remove_list, axis=0)
        return data, targets

class ROC_curve:
    def __init__(self, targets):
        self.targets = targets
        self.progress_bar = ProgressBar()
        self.softmax = torch.nn.Softmax(dim=1)
        self.Sigmoid = torch.nn.Sigmoid()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def calc_tpr_fpr(self, score, thresholds):
        tpr = []
        tnr = []
        for t in thresholds:
            tp = 0
            tn = 0
            new_score = [int(score[i] > t) for i in range(len(score))]
            for j in range(len(new_score)):
                if new_score[j] == self.targets[j]:
                    if new_score[j] == 1: tp+=1
                    else: tn+=1
            tpr.append(tp/sum(self.targets)) #fraction of positive caught
            tnr.append(tn/(len(self.targets) - sum(self.targets))) #fraction of negative caught
        return tpr, tnr
    def plot_roc_curve(self,epoch, scores ,title ,plot_path, results_path, test_names):
        #tpr, fpr = self.calc_tpr_fpr(score, thresholds)
        #plt.plot(fpr,tpr)
        best_auc = 0
        best_overall_tnr = 0
        best_overall_tnr_tpr = 0
        best_overall_tpr = 0
        best_overall_threshold = 0
        f_best = open(results_path+"_best.txt", 'a')
        best_test = 0
        for i in range(len(scores)):
            score = scores[i]
            test_name = test_names[i]
            fpr, tpr, thresholds = metrics.roc_curve(self.targets, score)
            tnr = 1-fpr
            auc = metrics.roc_auc_score(self.targets, score)
            f1scores = []
            print("AUC score = " + str(auc))
            if len(thresholds>1000): thresholds = [i/1000.0 for i in range(1000)]
            for t in thresholds:
                noisypred = [int(score[j]>=t) for j in range(len(score))]
                f1scores.append(metrics.f1_score(self.targets, noisypred))
            best_index = (np.asarray(f1scores)).argmax()
            threshold = thresholds[best_index]
            best_tpr = tpr[best_index]
            best_tnr = tnr[best_index]
            best_f1 = max(f1scores)
            if(best_auc < auc):
                best_auc = auc
                best_overall_tnr = best_tnr
                best_overall_f1 = best_f1
                best_overall_tpr = best_tpr
                best_overall_threshold = threshold
                best_test = i
            print("max F1 score is " + str(best_f1)+" when the threshold is " +str(threshold)+" and the tpr and tnr are "+str(best_tpr)+","+str(best_tnr))
        # plt.plot(fpr, tpr)
            #plt.plot(fpr, tpr, label=test_name)
        f_best.write("\n epoch "+str(epoch)+ " AUC score = " + str(best_auc) + "\n")
        f_best.write("best F1 score is " + str(best_overall_f1) + " when the threshold is " + str(
            best_overall_threshold) + " and the tpr and tnr are " + str(best_overall_tpr) + "," + str(best_overall_tnr) + "\n")
        f_best.close()
        return best_overall_threshold, best_test, thresholds, tpr ,best_auc
    def calc_confidence_score(self, confidence_score, results_matrix, conf_matrix, correct_score, labels, consistency, consistency_score, num_of_nets):
        most_common = mostCommon(results_matrix)
        result = [res for (res, res_accesibility) in most_common]
        result_accesibility = [res_accesibility for (res, res_accesibility) in most_common]
        result_accesibility = np.asarray(result_accesibility) /num_of_nets
        conf_score = np.array(conf_matrix).max(axis=0)
        conf_score *= result_accesibility
        if consistency:
            conf_score = conf_score * consistency_score
        if correct_score:
            correct = torch.eq(torch.Tensor(result), labels.flatten())
            for j in range(len(labels)):
                if correct[j]:
                    confidence_score = np.append(confidence_score, conf_score[j])
                else:
                    confidence_score = np.append(confidence_score, 0)
        else:
            confidence_score = np.append(confidence_score, conf_score)
        return confidence_score

    def calc_correct_labels(self, nets, labels, filter, old_labels, correction_threshold):
        new_labels = []
        remove_data = []
        with torch.no_grad():
            filter_index = 0
            for i in range(len(labels)):
                results_matrix = []
                for net in nets:
                    results_matrix.append(net.predictions[i])
                #most_commons = [Counter(col).most_common(2)[0] for col in zip(*results_matrix)]
                two_results = [Counter(col).most_common(2) for col in zip(*results_matrix)]
                first_result = [results[0] for results in two_results]
                second_result = [results[1] if len(results)>1 else (-1,0) for results in two_results]
                for j in range(len(labels[i])):
                    if(first_result[j][0] == old_labels[filter_index] and second_result[j]!=-1):
                        newlabel, result_accesibility = second_result[j]
                    else: newlabel, result_accesibility = first_result[j]
                    if filter[filter_index]:
                        new_labels.append(old_labels[filter_index])
                    elif (result_accesibility/len(nets) < correction_threshold):
                        new_labels.append(old_labels[filter_index])
                        remove_data.append(filter_index)
                    else:
                        new_labels.append(newlabel)
                    filter_index+=1
            return new_labels, remove_data

    def correct_score(self, nets, labels):
        confidence_score = np.empty(0)
        with torch.no_grad():
            for i in range(len(labels)):
                results_matrix = []
                for net in nets:
                    results_matrix.append(net.predictions[i])
                most_common = mostCommon(results_matrix)
                result = [res for (res, res_accesibility) in most_common]
                result_accesibility = [res_accesibility for (res, res_accesibility) in most_common]
                result_accesibility = np.asarray(result_accesibility) / len(nets)
                correct = torch.eq(torch.Tensor(result), labels.flatten())
                for j in range(len(labels)):
                    if correct[j]:
                        confidence_score = np.append(confidence_score, result_accesibility[j])
                    else:
                        confidence_score = np.append(confidence_score, 0)
            return confidence_score

    def loss_score(self,train_dataset, nets, consistency_rates=None, correct_score=False,
                  consistency=False):
        self.progress_bar.new_line()
        # first test - using loss as confidence score
        criterion = torch.nn.CrossEntropyLoss().cuda() if self.use_cuda else torch.nn.CrossEntropyLoss()
        confidence_score = np.empty(0)
        consistency_score = np.array(consistency_rates).max(axis=0)
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                results_matrix = []
                conf_matrix = []
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                for net in nets:
                    outputs = net.net.forward(images)
                    conf_score_tmp = criterion(outputs, labels)
                    _, predicted = torch.max(outputs, dim=1)
                    predicted = predicted.cpu()
                    results_matrix.append(predicted.tolist())
                    conf_score_tmp = conf_score_tmp.cpu()
                    conf_matrix.append(conf_score_tmp.tolist())
                confidence_score = self.calc_confidence_score(confidence_score, results_matrix, conf_matrix,
                                                              correct_score, labels,
                                                              consistency, consistency_score, len(nets))
        return confidence_score
    def max_score(self, nets, labels, consistency_rates=None, correct_score=False, consistency=False):
        self.progress_bar.new_line()
        # first test - using max activate value as confidence score
        confidence_score = np.empty(0)
        consistency_score = np.array(consistency_rates).max(axis=0)
        with torch.no_grad():
            for i in range(len(labels)):
                results_matrix = []
                conf_matrix = []
                for net in nets:
                    results_matrix.append(net.predictions[i])
                    conf_matrix.append(net.max_logits[i])
                    confidence_score = self.calc_confidence_score(confidence_score, results_matrix, conf_matrix,
                                                                  correct_score, labels[i],
                                                                  consistency, consistency_score, len(nets))
        return confidence_score

    def margin_score(self, nets, labels, consistency_rates = None, correct_score = False, consistency = False):
        self.progress_bar.new_line()
        confidence_score = np.empty(0)
        consistency_score = np.array(consistency_rates).max(axis=0)
        with torch.no_grad():
            for i in range(len(labels)):
                results_matrix = []
                activation_matrix = []
                for net in nets:
                    activation_matrix.append(net.margin[i])
                    results_matrix.append(net.predictions[i])
                if consistency_score is None:
                    confidence_score = self.calc_confidence_score(confidence_score, results_matrix, activation_matrix,
                                                                  correct_score, labels[i],
                                                                  consistency, consistency_score, len(nets))
        return confidence_score
    def entropy_score(self, nets, labels, batch_size, consistency_rates = None, correct_score = False, consistency = False):
        self.progress_bar.new_line()
        confidence_score = np.empty(0)
        consistency_score = np.array(consistency_rates).max(axis=0)
        with torch.no_grad():
            for i in range(len(labels)):
                results_matrix = []
                entropy_matrix = []
                for net in nets:
                    results_matrix.append(net.predictions[i])
                    entropy_matrix.append(net.entropy[i])
                if consistency_score is None:
                    confidence_score = self.calc_confidence_score(confidence_score, results_matrix, entropy_matrix,
                                                              correct_score, labels[i],
                                                              consistency, consistency_score, len(nets))

        return confidence_score

    def calc_noise_dist(self, clean_labels, noisy_labels, num_classes):
        noise_dist = [[0] * num_classes for _ in range(num_classes)]
        noise_level_for_class = [0]*num_classes
        total_noisy_for_class = [0]*num_classes #sum of noisy labels per class
        for i in range(len(clean_labels)):
            noise_level_for_class[clean_labels[i]]+=1 #summing the appearnces of instances for each class
            if clean_labels[i]!=noisy_labels[i]: #noisy label
                noise_dist[clean_labels[i]][noisy_labels[i]]+=1
                total_noisy_for_class[clean_labels[i]]+=1
        for class_idx in range(num_classes):
            noise_dist[class_idx] = [noise_dist[class_idx][j]/total_noisy_for_class[class_idx] for j in range(num_classes)]
            noise_level_for_class[class_idx] = total_noisy_for_class[class_idx]/noise_level_for_class[class_idx]
        return noise_dist, noise_level_for_class




class ProgressBar:
    """
    Prints a progress bar to the standard output (very similar to Keras).
    """

    def __init__(self, width=30):
        """
        Parameters
        ----------
        width : int
            The width of the progress bar (in characters)
        """
        self.width = width

    def update(self, max_value, current_value, info):
        """Updates the progress bar with the given information.
        Parameters
        ----------
        max_value : int
            The maximum value of the progress bar
        current_value : int
            The current value of the progress bar
        info : str
            Additional information that will be displayed to the right of the progress bar
        """
        progress = int(round(self.width * current_value / max_value))
        bar = '=' * progress + '.' * (self.width - progress)
        prefix = '{}/{}'.format(current_value, max_value)

        prefix_max_len = len('{}/{}'.format(max_value, max_value))
        buffer = ' ' * (prefix_max_len - len(prefix))

        sys.stdout.write('\r {} {} [{}] - {}'.format(prefix, buffer, bar, info))
        sys.stdout.flush()

    def new_line(self):
        print()


def mostCommon(lst):
    return [Counter(col).most_common(1)[0] for col in zip(*lst)]

class Cutout(object):
    """
    Implements Cutout regularization as proposed by DeVries and Taylor (2017), https://arxiv.org/pdf/1708.04552.pdf.
    """

    def __init__(self, num_cutouts, size, p=0.5):
        """
        Parameters
        ----------
        num_cutouts : int
            The number of cutouts
        size : int
            The size of the cutout
        p : float (0 <= p <= 1)
            The probability that a cutout is applied (similar to keep_prob for Dropout)
        """
        self.num_cutouts = num_cutouts
        self.size = size
        self.p = p

    def __call__(self, img):

        height, width = img.size

        cutouts = np.ones((height, width))

        if np.random.uniform() < 1 - self.p:
            return img

        for i in range(self.num_cutouts):
            y_center = np.random.randint(0, height)
            x_center = np.random.randint(0, width)

            y1 = np.clip(y_center - self.size // 2, 0, height)
            y2 = np.clip(y_center + self.size // 2, 0, height)
            x1 = np.clip(x_center - self.size // 2, 0, width)
            x2 = np.clip(x_center + self.size // 2, 0, width)

            cutouts[y1:y2, x1:x2] = 0

        cutouts = np.broadcast_to(cutouts, (3, height, width))
        cutouts = np.moveaxis(cutouts, 0, 2)
        img = np.array(img)
        img = img * cutouts
        return Image.fromarray(img.astype('uint8'), 'RGB')

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count