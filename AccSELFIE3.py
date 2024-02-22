import argparse
import sys

import numpy as np
import os
import scipy.signal
import scipy.stats
import torch
import copy
import random
import myClothing1m
import webvision_dataloader
from TinyImagenet2 import TinyImageNet200
from tqdm import tqdm
from scipy.signal import argrelextrema
from utilv3 import BetaMixture1D
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from findpeaks import findpeaks
from DenseNet import DenseNet
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.mixture import GaussianMixture
from TinyImagenet import TinyImageNet
from ImageNet import MyImageNet


"""
Algorithm framework:
1) Use trained ensemble to filter noisy data from clean in the original noisy dataset
2) Use trained ensemble to generate alternative labels to the entire data, with confidence score to each alternative label (the entropy whatever)
3) Generate new dataset, composed of the high confidence alternative labels set R and the clean set C\R
4) Train the ensemble again on the new dataset and go back to step 1

Main functions:
filter_noisy_dataset - filter noisy from clean
generate_alternative_labels - create the other labels
generate_new_labels - create the new dataset
save_dataset - write the new labels

helping functions:
get_dataset - takes the labels from file
calc_precision_recall - check if it works well

new filtering functions:
calc_learning_epoch - finds the minimal learning epoch of each example
filter_by_epoch - gets the subset of examples learned at epoch e and filter the subset by accessibility
"""
def createMyGraph(pred_array, logits_array, clean_indices, number_of_nets = 10): #arrays of binary predictions and ground truth logits of 50000 x 200 x 10
    agreement_per_epoch = np.mean(pred_array, axis = 2)
    logits_per_epoch = np.mean(logits_array, axis = 2)
    learning_time_start = np.argmax(agreement_per_epoch > 0.2, axis = 1)
    learning_time_end = np.argmax(agreement_per_epoch == 1, axis = 1)
    learning_time_end[np.where(learning_time_end == 0)[0]] = len(pred_array[0])
    equation_vals = []
    for epoch in tqdm(np.unique(learning_time_start)):
        indices = np.where(learning_time_start == epoch)[0]
        cidx = np.intersect1d(np.where(clean_indices == 1)[0],indices)
        nidx = np.intersect1d(np.where(clean_indices == 0)[0],indices)
        clean_agreement_per_epoch = np.mean(agreement_per_epoch[cidx], axis=0)
        noisy_agreement_per_epoch = np.mean(agreement_per_epoch[nidx], axis=0)
        agr_diff = clean_agreement_per_epoch - noisy_agreement_per_epoch
        clean_logits_per_epoch = np.mean(agreement_per_epoch[cidx], axis=0)
        noisy_logits_per_epoch = np.mean(logits_per_epoch[nidx], axis=0)
        logits_diff = clean_logits_per_epoch - noisy_logits_per_epoch

        epoch_end = int(max(np.mean(learning_time_end[nidx]),np.mean(learning_time_end[cidx])))
        cagr = np.sum(agreement_per_epoch[cidx][:,epoch:epoch_end])/(len(cidx)*(epoch_end - epoch))
        nagr = np.sum(agreement_per_epoch[nidx][:,epoch:epoch_end])/(len(nidx)*(epoch_end - epoch))
        agr_avg_diff = cagr - nagr
        cnotlearnedlogits = 0
        nnotlearnedlogits = 0
        #idx_epoch = lambda x: np.where(x == 0)[0]
        #idx_example = lambda arr: np.apply_along_axis(func1d = idx_epoch, axis = 0, arr = arr)
        #clean_not_learned_idx = np.apply_along_axis(func1d = idx_example, axis = 0, arr = pred_array[cidx])
        #noisy_not_learned_idx = np.apply_over_axes(func = idx, a = pred_array[nidx], axes = [0,1])
        #clean_not_learned = logits_array[cidx][clean_not_learned_idx]
        #noisy_not_learned = logits_array[nidx][noisy_not_learned_idx]
        for e in range(len(pred_array[0])):
            clean_not_learned = np.asarray([logits_array[cidx][e][np.where(1*pred_array[cidx][i,e] == 0)[0]] for i in cidx])
            noisy_not_learned = np.asarray([logits_array[nidx][e][np.where(1*pred_array[nidx][j,e] == 0)[0]] for j in nidx])
            cnotlearnedlogits += np.sum(clean_not_learned)/(len(cidx)*len(clean_not_learned))
            nnotlearnedlogits += np.sum(noisy_not_learned)/(len(nidx)*len(noisy_not_learned))

        cnotlearnedlogits = np.mean(cnotlearnedlogits)
        equation_val = (epoch_end - epoch) - (len(pred_array[0])/(agr_avg_diff*cnotlearnedlogits+agr_avg_diff - number_of_nets))
        equation_vals.append(equation_val)

    return equation_vals


def calc_learning_epoch(data_pred): #a vector of T predictions for data exampls
    try:
        last_zero = max(np.where(data_pred == 0)[0])
        first_one = min(np.where(data_pred == 1)[0])
    except:
        last_zero = -1
        first_one = len(data_pred) - 1
    return last_zero+1
    #return first_one

def calc_net_learning_epochs(net_data_pred, net): # a matrix of T x data predictions
    print(net)
    return np.apply_along_axis(calc_learning_epoch, axis = 0, arr = net_data_pred)

def calc_agreement_per_epoch(epoch_data_pred,epoch): # a matrix if K x data predictions
    print(epoch)
    return np.sum(epoch_data_pred, axis = 0) > 1
"""
def calc_min_learning_time(all_predictions): # a matrix of k x T x data predictions
    all_learning_times = 1*np.asarray([calc_net_learning_epochs(1*calc_agreement_per_epoch[:,epoch,:], epoch) for epoch in range(len(all_predictions[0]))])
    result = [min(np.where(all_learning_times[i] == 1)) for i in range(len(all_learning_times))]
    return result
"""
def calc_min_learning_time(all_predictions): # a matrix of k x T x data predictions
    #all_learning_times = np.asarray([calc_net_learning_epochs(1*all_predictions[net], net) for net in range(len(all_predictions))])
    agreement_per_epoch = np.flip(np.mean(all_predictions, axis = 0), axis = 0)
    easiest_data = np.zeros(len(agreement_per_epoch))
    all_learning_times = 200 - np.argmax(agreement_per_epoch == 0, axis = 0)
    for e in range(len(agreement_per_epoch[0])):
        if np.array_equal((agreement_per_epoch[:,e] == 0), easiest_data):
            all_learning_times[e] = 0
    return all_learning_times

def calc_precision_recall(pred, target):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(pred)):
        if int(pred[i]) == 1: #positive prediction
            if target[i] == 1: #true positive
                TP+=1
            else: #false positive
                FP+=1
        else: #negative prediction
            if target[1] == 1: #false negative
                FN+=1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return precision, recall

def get_original_noisy_dataset(labels_path):
    try:
        clean_indices = np.load(labels_path + '/clean_indices.npy')
        current_labels = np.load(labels_path + '/noisy_labels.npy')
    except:
        current_labels = []
        clean_indices = []

    return current_labels, clean_indices

def calc_test_dist(folder_name, number_of_nets, epochs_num, classes_num, real_labels, Step_idx):
    last_pred = []
    synthetic_labels = []
    for i in range(40, 40+ number_of_nets):
        pred = np.load(
            folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'test_resnet_' + str(i) + '_' + str(
                epochs_num-1) + '.npy')
        if len(pred) < len(real_labels):
            raise ValueError("Net " + str(i) + " in epoch " + str(
                epochs_num-1) + " doesn't have all the predictions for the dataset")
        last_pred.append(pred)
    last_pred = np.asarray(last_pred)
    for j in range(len(last_pred[0])):
        values, counts = np.unique(last_pred[:,j], return_counts=True)
        synthetic_labels.append(values[np.argmax(counts)])

    synthetic_labels = np.asarray(synthetic_labels)
    prev_score = np.zeros(len(synthetic_labels))
    all_pred = []
    for epoch in tqdm(range(epochs_num)):
        all_pred.append([])
        # loading the current epoch
        nets_binary_prediction = []
        for i in range(40,40+number_of_nets):
            pred = np.load(
                folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'test_resnet_' + str(i) + '_' + str(
                    epoch) + '.npy')
            if len(pred) < len(synthetic_labels):
                raise ValueError("Net " + str(i) + " in epoch " + str(
                    epoch) + " doesn't have all the predictions for the dataset")
            else:
                all_pred[epoch].append(pred)
                binary_pred = pred == synthetic_labels
                binary_pred = 1 * binary_pred
                nets_binary_prediction.append(binary_pred)
        nets_binary_prediction = np.asarray(nets_binary_prediction)
        score = np.mean(nets_binary_prediction, axis=0)
        score += prev_score
        prev_score = score
    score = score / (epoch + 1)

    all_pred = np.asarray(all_pred)
    c_array = []
    for c in tqdm(range(classes_num)):
        labels = np.ones(all_pred.shape) * c
        binary_all_pred = labels == all_pred
        c_array.append(binary_all_pred)

    c_array = np.asarray(c_array)
    AL_scores = np.mean(c_array, axis=(1, 2))
    AL_scores[synthetic_labels, np.arange(AL_scores.shape[1])] = 0
    alternative_labels = np.argmax(AL_scores, axis=0)

    bmm = BetaMixture1D()
    bmm.fit(score)
    """
    x = np.linspace(0, 1, 100)
    negative = bmm.weighted_likelihood(x, 0)
    positive = bmm.weighted_likelihood(x, 1)
    dist = np.abs(negative - positive)
    close_to_zero = np.where(dist < 0.1)[0]
    if close_to_zero[len(close_to_zero) - 1] == len(dist) - 1:  # the distributions intersect at the end
        i = 1
        while close_to_zero[len(close_to_zero) - i - 1] == close_to_zero[len(close_to_zero) - i] - 1:
            i += 1
        end_idx = close_to_zero[len(close_to_zero) - i - 1] + 1
    else:
        end_idx = close_to_zero[len(close_to_zero) - 1] + 1
    begin_idx = end_idx - 1
    while dist[begin_idx - 1] < 0.1:
        begin_idx -= 1
    filter_threshold = (dist[begin_idx:end_idx].argmin() + begin_idx) / 100
    
    bins, _ = np.histogram(score, bins=50)
    fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
    results = fp.fit(bins)
    bin_score = np.array(results['df'].score)
    peaks = np.array(results['df'].x)
    strongest_peaks = np.argsort(bin_score)[::-1]
    try:
        indices1, indices2 = peaks[strongest_peaks[0]], peaks[strongest_peaks[1]]
    except:
        indices1, indices2 = 0, len(bins) - 1
    peak_left, peak_right = min(indices1, indices2), max(indices1, indices2)
    smallest_bin = np.argmin(bins[peak_left:peak_right]) + peak_left
    filter_threshold = smallest_bin / len(bins)
    """
    correct_synthesis = synthetic_labels == np.asarray(real_labels)
    best_acc = 0
    best_threshold = 0
    for t in range(100):
        filter_threshold = t/100
        pred_filter = 1 * (score > filter_threshold)
        switch_pred_idx = np.where(pred_filter == 0)[0]
        l = synthetic_labels.copy()
        l[switch_pred_idx] = alternative_labels[switch_pred_idx]
        if np.mean(l == np.asarray(real_labels)) > best_acc:
            best_acc = np.mean(l == np.asarray(real_labels))
            best_threshold = t/100
    print("threshold " + str(best_threshold) + " accuracy after switch = " + str(
        best_acc))
    bins, _ = np.histogram(score, bins=20)
    plt.hist(score, bins=20, color='blue', alpha=0.5)
    plt.hist(score[np.where(correct_synthesis == 0)[0]], bins=20, color='red', alpha=0.5)
    plt.hist(score[np.where(correct_synthesis == 1)[0]], bins=20, color='green', alpha=0.5)
    plt.plot(best_threshold, bins[int(best_threshold*20)], marker = 'o', color = 'black')
    plt.title("test ELP dist for last epoch predictions")
    plt.show()
    bmm.plot()
    plt.show()
    print("original accuracy = "+str(np.mean(synthetic_labels == np.asarray(real_labels))))


def filter_noisy_dataset(filter, folder_name, number_of_nets, epochs_num, noisy_labels, Step_idx, classes_num, clean_indices):
    daniel_bimodal_values = []
    all_scores = []
    all_pred = []
    acc_alternative_label_matrix = []
    all_binary_pred = []
    if filter == 'AccPerLT':
        data_filter = np.empty(len(noisy_labels))
        for i in range(number_of_nets):
            print("net "+str(i))
            all_binary_pred.append([])
            for epoch in range(epochs_num):
                pred = np.load(
                    folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(i) + '_' + str(
                        epoch) + '.npy')
                if len(pred) < len(noisy_labels):
                    raise ValueError("Net " + str(i) + " in epoch " + str(
                        epoch) + " doesn't have all the predictions for the dataset")
                all_binary_pred[i].append(pred == noisy_labels)

        LearningTime = calc_min_learning_time(all_binary_pred)
        score_per_epoch = np.mean(all_binary_pred, axis = 0)
        unique = np.asarray([np.unique(row, return_counts=True) for row in score_per_epoch])
        agreement_values_per_epoch, agreement_count_tmp = unique[:,0], unique[:,1]
        agreement_count_per_epoch = np.zeros((len(score_per_epoch), number_of_nets+1))
        for l in range(len(agreement_count_per_epoch)):
            np.put(agreement_count_per_epoch[l], (agreement_values_per_epoch[l] * number_of_nets).astype(int), agreement_count_tmp[l])
        bimodal_values = np.sqrt(agreement_count_per_epoch[:,0] / len(noisy_labels)) + np.sqrt(
            agreement_count_per_epoch[:,number_of_nets] / len(noisy_labels))
        if Step_idx == 1:
            filter_epoch = bimodal_values[bimodal_values.argmax():].argmin()+bimodal_values.argmax()
        else:
            filter_epoch = epochs_num - 1
        score = np.mean(np.asarray(all_binary_pred)[:,:filter_epoch,:], axis = (0,1))
        for e in np.unique(LearningTime):
            clean_idx = np.intersect1d(np.where(LearningTime == e)[0], np.where(clean_indices == 1))
            noisy_idx = np.intersect1d(np.where(LearningTime == e)[0], np.where(clean_indices == 0))
            bins, _ = np.histogram(score[np.where(LearningTime == e)], bins=50)
            fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
            results = fp.fit(bins)
            bin_score = np.array(results['df'].score)
            peaks = np.array(results['df'].x)
            strongest_peaks = np.argsort(bin_score)[::-1]
            try:
                indices1, indices2 = peaks[strongest_peaks[0]], peaks[strongest_peaks[1]]
            except:
                indices1, indices2 = 0, len(bins) - 1
            peak_left, peak_right = min(indices1, indices2), max(indices1, indices2)
            smallest_bin = np.argmin(bins[peak_left:peak_right]) + peak_left
            filter_threshold = smallest_bin / len(bins)
            data_filter[np.where(LearningTime == e)] = np.asarray(score[np.where(LearningTime == e)] >= filter_threshold) * 1
            if e > 130:
                plt.hist(score[np.where(LearningTime == e)], bins=40, color='blue')
                plt.hist(score[clean_idx], bins=40, color='green')
                plt.hist(score[noisy_idx], bins=40, color='red')
                plt.title("epoch: "+str(e)+"# examples: "+str(len(np.where(LearningTime == e)[0]))+ "# clean: " + str(len(clean_idx))+"# noisy: "+ str(len(noisy_idx)))
                plt.show()
                plt.close()
            print("epoch: "+str(e)+"# examples: "+str(len(np.where(LearningTime == e)[0]))+ "# clean: " + str(len(clean_idx))+"# noisy: "+ str(len(noisy_idx)))
    if filter == 'AccPerClass':
        data_filter = np.empty(len(noisy_labels))
        for i in range(number_of_nets):
            print("net "+str(i))
            all_binary_pred.append([])
            for epoch in range(epochs_num):
                pred = np.load(
                    folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(i) + '_' + str(
                        epoch) + '.npy')
                if len(pred) < len(noisy_labels):
                    raise ValueError("Net " + str(i) + " in epoch " + str(
                        epoch) + " doesn't have all the predictions for the dataset")
                all_binary_pred[i].append(pred == noisy_labels)

        score_per_epoch = np.mean(all_binary_pred, axis = 0)
        unique = np.asarray([np.unique(row, return_counts=True) for row in score_per_epoch])
        agreement_values_per_epoch, agreement_count_tmp = unique[:,0], unique[:,1]
        agreement_count_per_epoch = np.zeros((len(score_per_epoch), number_of_nets+1))
        for l in range(len(agreement_count_per_epoch)):
            np.put(agreement_count_per_epoch[l], (agreement_values_per_epoch[l] * number_of_nets).astype(int), agreement_count_tmp[l])
        bimodal_values = np.sqrt(agreement_count_per_epoch[:,0] / len(noisy_labels)) + np.sqrt(
            agreement_count_per_epoch[:,number_of_nets] / len(noisy_labels))
        if Step_idx == 1:
            filter_epoch = bimodal_values[bimodal_values.argmax():].argmin()+bimodal_values.argmax()
        else:
            filter_epoch = epochs_num - 1
        score = np.mean(np.asarray(all_binary_pred)[:,:filter_epoch,:], axis = (0,1))
        for e in range(classes_num):
            clean_idx = np.intersect1d(np.where(noisy_labels == e)[0], np.where(clean_indices == 1))
            noisy_idx = np.intersect1d(np.where(noisy_labels == e)[0], np.where(clean_indices == 0))
            bins, _ = np.histogram(score[np.where(noisy_labels == e)], bins=50)
            fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
            results = fp.fit(bins)
            bin_score = np.array(results['df'].score)
            peaks = np.array(results['df'].x)
            strongest_peaks = np.argsort(bin_score)[::-1]
            try:
                indices1, indices2 = peaks[strongest_peaks[0]], peaks[strongest_peaks[1]]
            except:
                indices1, indices2 = 0, len(bins) - 1
            peak_left, peak_right = min(indices1, indices2), max(indices1, indices2)
            smallest_bin = np.argmin(bins[peak_left:peak_right]) + peak_left
            filter_threshold = smallest_bin / len(bins)
            data_filter[np.where(noisy_labels == e)] = np.asarray(score[np.where(noisy_labels == e)] >= filter_threshold) * 1
            #plt.hist(score[np.where(noisy_labels == e)], bins=50, color='blue')
            #plt.hist(score[clean_idx], bins=50, color='green')
            #plt.hist(score[noisy_idx], bins=50, color='red')
            #plt.title("# examples: "+str(len(np.where(noisy_labels == e)[0]))+ "# clean: " + str(len(clean_idx))+"# noisy: "+ str(len(noisy_idx)))
            #plt.show()
            #plt.close()
    if filter == 'accessibility':
        prev_score = np.zeros(len(noisy_labels))
        for epoch in range(epochs_num):
            print("\nepoch: " + str(epoch))
            # loading the current epoch
            nets_prediction = []
            nets_binary_prediction = []
            for i in range(number_of_nets):
                pred = np.load(
                    folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(i) + '_' + str(
                        epoch) + '.npy')
                if len(pred) < len(noisy_labels):
                    raise ValueError("Net " + str(i) + " in epoch " + str(
                        epoch) + " doesn't have all the predictions for the dataset")
                else:
                    nets_prediction.append(pred)
                    binary_pred = pred == noisy_labels
                    binary_pred = 1 * binary_pred
                    nets_binary_prediction.append(binary_pred)
            nets_prediction = np.asarray(nets_prediction)
            nets_binary_prediction = np.asarray(nets_binary_prediction)
            agreement_count = np.zeros(number_of_nets + 1)
            score = np.mean(nets_binary_prediction, axis=0)
            agreement_value, agreement_count_temp = np.unique(score, return_counts=True)
            np.put(agreement_count, (agreement_value * number_of_nets).astype(int), agreement_count_temp)
            score += prev_score
            prev_score = score
            score = score / (epoch + 1)
            all_scores.append(score)
            """
            if epoch == 0:
                acc_alternative_label_matrix = np.zeros((len(noisy_labels),classes_num))
            agreement_over_classes = np.zeros((len(noisy_labels),classes_num))
            for i in range(len(nets_prediction[0])):
                agreement_value, agreement_count_temp = np.unique(nets_prediction[:,i], return_counts=True)
                agreement_over_classes[i][agreement_value.astype(int)]+= agreement_count_temp
            
            
            if epoch == 0:
                acc_alternative_label_matrix = np.zeros((len(noisy_labels),classes_num))
            for network_prediction in nets_prediction:
                for i in range(len(network_prediction)):
                    acc_alternative_label_matrix[i][int(network_prediction[i])] += 1
            """
            daniel_bimodal_value = np.sqrt(agreement_count[0] / len(noisy_labels)) + np.sqrt(
                agreement_count[number_of_nets] / len(noisy_labels))
            daniel_bimodal_values.append(daniel_bimodal_value)

        if Step_idx == 1:
            daniel_bimodal_values = np.asarray(daniel_bimodal_values)
            filter_epoch = daniel_bimodal_values[daniel_bimodal_values.argmax():].argmin()+daniel_bimodal_values.argmax()
        else:
            filter_epoch = epochs_num - 1

        plt.hist(score, bins=100, color='blue')
        plt.hist(score[np.where(clean_indices == 1)[0]], bins=100, color='green')
        plt.hist(score[np.where(clean_indices == 0)[0]], bins=100, color='red')
        plt.show()
        # bins_values, bins = np.unique(score, return_counts=True)
        bins, _ = np.histogram(all_scores[filter_epoch], bins=100)
        fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
        results = fp.fit(bins)
        bin_score = np.array(results['df'].score)
        peaks = np.array(results['df'].x)
        strongest_peaks = np.argsort(bin_score)[::-1]
        try:
            indices1, indices2 = peaks[strongest_peaks[0]], peaks[strongest_peaks[1]]
        except:
            indices1, indices2 = 0, len(bins) - 1
        peak_left, peak_right = min(indices1, indices2), max(indices1, indices2)
        smallest_bin = np.argmin(bins[peak_left:peak_right]) + peak_left
        filter_threshold = smallest_bin / len(bins)
        data_filter = np.asarray(score >= filter_threshold)*1
    if filter == 'ELP':
        prev_score = np.zeros(len(noisy_labels))
        for epoch in tqdm(range(epochs_num)):
            # loading the current epoch
            nets_prediction = []
            nets_binary_prediction = []
            for i in range(1,number_of_nets+1):
                pred = np.load(
                    folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(i) + '_' + str(
                        epoch) + '.npy')
                if len(pred) < len(noisy_labels):
                    raise ValueError("Net " + str(i) + " in epoch " + str(
                        epoch) + " doesn't have all the predictions for the dataset")
                else:
                    nets_prediction.append(pred)
                    binary_pred = pred == noisy_labels
                    binary_pred = 1 * binary_pred
                    nets_binary_prediction.append(binary_pred)
            nets_prediction = np.asarray(nets_prediction)
            nets_binary_prediction = np.asarray(nets_binary_prediction)
            all_pred.append(nets_prediction)
            agreement_count = np.zeros(number_of_nets + 1)
            score = np.mean(nets_binary_prediction, axis=0)
            agreement_value, agreement_count_temp = np.unique(score, return_counts=True)
            np.put(agreement_count, (agreement_value * number_of_nets).astype(int), agreement_count_temp)
            score += prev_score
            prev_score = score
            score = score / (epoch + 1)
            all_scores.append(score)
            """
            if epoch == 0:
                acc_alternative_label_matrix = np.zeros((len(noisy_labels),classes_num))
            agreement_over_classes = np.zeros((len(noisy_labels),classes_num))
            for i in range(len(nets_prediction[0])):
                agreement_value, agreement_count_temp = np.unique(nets_prediction[:,i], return_counts=True)
                agreement_over_classes[i][agreement_value.astype(int)]+= agreement_count_temp


            if epoch == 0:
                acc_alternative_label_matrix = np.zeros((len(noisy_labels),classes_num))
            for network_prediction in nets_prediction:
                for i in range(len(network_prediction)):
                    acc_alternative_label_matrix[i][int(network_prediction[i])] += 1
            """
            daniel_bimodal_value = np.sqrt(agreement_count[0] / len(noisy_labels)) + np.sqrt(
                agreement_count[number_of_nets] / len(noisy_labels))
            daniel_bimodal_values.append(daniel_bimodal_value)

        if Step_idx == 1:
            daniel_bimodal_values = np.asarray(daniel_bimodal_values)
            filter_epoch = daniel_bimodal_values[
                           daniel_bimodal_values.argmax():].argmin() + daniel_bimodal_values.argmax()
        else:
            filter_epoch = epochs_num - 1

        # bins_values, bins = np.unique(score, return_counts=True)
        bins, bins_threshold = np.histogram(score, bins=100)
        fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
        results = fp.fit(bins)
        bin_score = np.array(results['df'].score)
        peaks = np.array(results['df'].x)
        strongest_peaks = np.argsort(bin_score)[::-1]
        try:
            indices1, indices2 = peaks[strongest_peaks[0]], peaks[strongest_peaks[1]]
        except:
            indices1, indices2 = 0, len(bins) - 1
        peak_left, peak_right = min(indices1, indices2), max(indices1, indices2)
        smallest_bin = np.argmin(bins[peak_left:peak_right]) + peak_left
        filter_threshold = bins_threshold[smallest_bin]
        data_filter = np.asarray(score >= filter_threshold) * 1
        plt.hist(score, bins=100, color='blue')
        plt.hist(score[np.where(clean_indices == 1)[0]], bins=100, color='green')
        plt.hist(score[np.where(clean_indices == 0)[0]], bins=100, color='red')
        plt.plot(filter_threshold, bins[int(filter_threshold * 100)], marker='o', color='black')
        plt.show()
    if filter == 'GMM2':
        prev_score = np.zeros(len(noisy_labels))
        for epoch in tqdm(range(epochs_num)):
            # loading the current epoch
            nets_prediction = []
            nets_binary_prediction = []
            for i in range(number_of_nets):
                pred = np.load(
                    folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(i) + '_' + str(
                        epoch) + '.npy')
                if len(pred) < len(noisy_labels):
                    raise ValueError("Net " + str(i) + " in epoch " + str(
                        epoch) + " doesn't have all the predictions for the dataset")
                else:
                    nets_prediction.append(pred)
                    binary_pred = pred == noisy_labels
                    binary_pred = 1 * binary_pred
                    nets_binary_prediction.append(binary_pred)
            nets_prediction = np.asarray(nets_prediction)
            nets_binary_prediction = np.asarray(nets_binary_prediction)
            agreement_count = np.zeros(number_of_nets + 1)
            score = np.mean(nets_binary_prediction, axis=0)
            agreement_value, agreement_count_temp = np.unique(score, return_counts=True)
            np.put(agreement_count, (agreement_value * number_of_nets).astype(int), agreement_count_temp)
            score += prev_score
            prev_score = score
            score = score / (epoch + 1)

        filter_threshold, means, stds = calc_filter_threshold_GMM(score, n_components=2)
        data_filter = np.asarray(score >= filter_threshold) * 1
        bins, _ = np.histogram(score, bins=100)
        plt.hist(score[np.where(clean_indices == 1)[0]], bins=100, color='green', alpha = 0.5)
        plt.hist(score[np.where(clean_indices == 0)[0]], bins=100, color='red', alpha = 0.5)
        plt.plot(filter_threshold, bins[int(filter_threshold*100)], color = 'black', marker = 'o')
        plt.show()
        for i in range(len(means)):
            x = np.linspace(means[i] - 3 * stds[i][0][0], means[i] + 3 * stds[i][0][0], 100)
            plt.plot(x, scipy.stats.norm.pdf(x, means[i], stds[i][0][0]))
        plt.show()
    if filter == 'LM':
        all_logits = []
        for epoch in tqdm(range(epochs_num)):
            # loading the current epoch
            nets_prediction = []
            for i in range(20, 20 + number_of_nets):
                pred = np.load(
                    folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'clogits_' + str(i) + '_' + str(
                        epoch) + '.npy')
                if len(pred) < len(noisy_labels):
                    raise ValueError("Net " + str(i) + " in epoch " + str(
                        epoch) + " doesn't have all the predictions for the dataset")
                else:
                    nets_prediction.append(pred)
            nets_prediction = np.asarray(nets_prediction)
            all_logits.append(nets_prediction)
        all_logits = np.asarray(all_logits)
        score = np.mean(all_logits, axis=(0, 1))
        plt.hist(score, bins=100, color='blue')
        plt.hist(score[np.where(clean_indices == 1)[0]], bins=100, color='green')
        plt.hist(score[np.where(clean_indices == 0)[0]], bins=100, color='red')
        plt.show()
        # bins_values, bins = np.unique(score, return_counts=True)
        bins, _ = np.histogram(score, bins=100)
        fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
        results = fp.fit(bins)
        bin_score = np.array(results['df'].score)
        peaks = np.array(results['df'].x)
        strongest_peaks = np.argsort(bin_score)[::-1]
        try:
            indices1, indices2 = peaks[strongest_peaks[0]], peaks[strongest_peaks[1]]
        except:
            indices1, indices2 = 0, len(bins) - 1
        peak_left, peak_right = min(indices1, indices2), max(indices1, indices2)
        smallest_bin = np.argmin(bins[peak_left:peak_right]) + peak_left
        filter_threshold = smallest_bin / len(bins)
        data_filter = np.asarray(score >= filter_threshold) * 1
    if filter == 'GMM':
        prev_score = np.zeros(len(noisy_labels))
        for epoch in tqdm(range(epochs_num)):
            # loading the current epoch
            nets_prediction = []
            nets_binary_prediction = []
            for i in range(number_of_nets):
                pred = np.load(
                    folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(i) + '_' + str(
                        epoch) + '.npy')
                if len(pred) < len(noisy_labels):
                    raise ValueError("Net " + str(i) + " in epoch " + str(
                        epoch) + " doesn't have all the predictions for the dataset")
                else:
                    nets_prediction.append(pred)
                    binary_pred = pred == noisy_labels
                    binary_pred = 1 * binary_pred
                    nets_binary_prediction.append(binary_pred)
            nets_prediction = np.asarray(nets_prediction)
            nets_binary_prediction = np.asarray(nets_binary_prediction)
            all_pred.append(nets_prediction)
            score = np.mean(nets_binary_prediction, axis=0)
            score += prev_score
            prev_score = score
        score = score / (epoch + 1)
        gmm = GaussianMixture(n_components=2)
        clean_prediction = gmm.fit_predict(score.reshape(-1, 1))
        clean_label = clean_prediction[score.argmax()]
        data_filter = clean_prediction == clean_label
        plt.figure(0)
        plt.hist(score[np.where(clean_indices == 1)[0]], bins=100, color='green', alpha = 0.5)
        plt.hist(score[np.where(clean_indices == 0)[0]], bins=100, color='red', alpha = 0.5)
        plt.show()
        plt.figure(1)
        indices = np.argsort(gmm.means_.flatten())
        means = gmm.means_[indices]
        covs = gmm.covariances_[indices]
        stds = np.sqrt(covs / 2)
        for i in range(len(means)):
            x = np.linspace(means[i] - 3 * stds[i][0][0], means[i] + 3 * stds[i][0][0], 100)
            plt.plot(x, scipy.stats.norm.pdf(x, means[i], stds[i][0][0]))
        plt.show()
        print("avg log-likelihood: "+str(gmm.score(score.reshape(-1,1))))
    if filter == 'BMM':
        prev_score = np.zeros(len(noisy_labels))
        for epoch in tqdm(range(epochs_num)):
            # loading the current epoch
            nets_prediction = []
            nets_binary_prediction = []
            for i in range(number_of_nets):
                pred = np.load(
                    folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(i) + '_' + str(
                        epoch) + '.npy')
                if len(pred) < len(noisy_labels):
                    raise ValueError("Net " + str(i) + " in epoch " + str(
                        epoch) + " doesn't have all the predictions for the dataset")
                else:
                    nets_prediction.append(pred)
                    binary_pred = pred == noisy_labels
                    binary_pred = 1 * binary_pred
                    nets_binary_prediction.append(binary_pred)
            nets_prediction = np.asarray(nets_prediction)
            nets_binary_prediction = np.asarray(nets_binary_prediction)
            all_pred.append(nets_prediction)
            score = np.mean(nets_binary_prediction, axis=0)
            score += prev_score
            prev_score = score
        score = score / (epoch + 1)
        bmm = BetaMixture1D()
        bmm.fit(score)
        clean_prediction = bmm.predict(score)
        clean_label = True
        data_filter = clean_prediction == clean_label
        plt.figure(0)
        plt.hist(score[np.where(clean_indices == 1)[0]], bins=100, color='green', alpha = 0.5)
        plt.hist(score[np.where(clean_indices == 0)[0]], bins=100, color='red', alpha = 0.5)
        plt.show()
        plt.figure(1)
        bmm.plot()
        plt.show()
    if filter == 'BMM2':
        prev_score = np.zeros(len(noisy_labels))
        for epoch in tqdm(range(epochs_num)):
            # loading the current epoch
            nets_prediction = []
            nets_binary_prediction = []
            for i in range(number_of_nets):
                pred = np.load(
                    folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(i) + '_' + str(
                        epoch) + '.npy')
                if len(pred) < len(noisy_labels):
                    raise ValueError("Net " + str(i) + " in epoch " + str(
                        epoch) + " doesn't have all the predictions for the dataset")
                else:
                    nets_prediction.append(pred)
                    binary_pred = pred == noisy_labels
                    binary_pred = 1 * binary_pred
                    nets_binary_prediction.append(binary_pred)
            nets_prediction = np.asarray(nets_prediction)
            nets_binary_prediction = np.asarray(nets_binary_prediction)
            all_pred.append(nets_prediction)
            score = np.mean(nets_binary_prediction, axis=0)
            score += prev_score
            prev_score = score
        score = score / (epoch + 1)
        bmm = BetaMixture1D()
        bmm.fit(score)
        x = np.linspace(0, 1, 100)
        negative = bmm.weighted_likelihood(x, 0)
        positive = bmm.weighted_likelihood(x, 1)
        dist = np.abs(negative - positive)
        close_to_zero = np.where(dist < 0.1)[0]
        if close_to_zero[len(close_to_zero)-1] == len(dist) -1: #the distributions intersect at the end
            i = 1
            while close_to_zero[len(close_to_zero)-i-1] == close_to_zero[len(close_to_zero)-i] - 1:
                i+=1
            end_idx = close_to_zero[len(close_to_zero)-i-1]+1
        else:
            end_idx = close_to_zero[len(close_to_zero) - 1]+1
        begin_idx = end_idx - 1
        while dist[begin_idx - 1] < 0.1:
            begin_idx -= 1
        filter_threshold = (dist[begin_idx:end_idx].argmin()+begin_idx)/100
        data_filter = np.asarray(score >= filter_threshold) * 1
        plt.figure(0)
        bins, _ = np.histogram(score, bins=100)
        plt.hist(score[np.where(clean_indices == 1)[0]], bins=100, color='green', alpha=0.5)
        plt.hist(score[np.where(clean_indices == 0)[0]], bins=100, color='red', alpha=0.5)
        plt.plot(filter_threshold, bins[int(filter_threshold*100)], marker ='o', color = 'black')
        plt.show()
        plt.figure(1)
        bmm.plot()
        plt.plot(filter_threshold, negative[int(filter_threshold*100)], marker ='o', color = 'black')
        plt.show()
    if filter == "GMM3":
        prev_score = np.zeros(len(noisy_labels))
        for epoch in tqdm(range(epochs_num)):
            # loading the current epoch
            nets_prediction = []
            nets_binary_prediction = []
            for i in range(number_of_nets):
                pred = np.load(
                    folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(i) + '_' + str(
                        epoch) + '.npy')
                if len(pred) < len(noisy_labels):
                    raise ValueError("Net " + str(i) + " in epoch " + str(
                        epoch) + " doesn't have all the predictions for the dataset")
                else:
                    nets_prediction.append(pred)
                    binary_pred = pred == noisy_labels
                    binary_pred = 1 * binary_pred
                    nets_binary_prediction.append(binary_pred)
            nets_prediction = np.asarray(nets_prediction)
            nets_binary_prediction = np.asarray(nets_binary_prediction)
            all_pred.append(nets_prediction)
            score = np.mean(nets_binary_prediction, axis=0)
            score += prev_score
            prev_score = score
        score = score / (epoch + 1)
        gmm = GaussianMixture(n_components=1)
        best_log_likelihood = -10
        filter_threshold = 0
        for i in range(0, 100):
            threshold = i/100
            tmp_score = np.delete(score.copy(), np.where(score > threshold)[0])
            if len(tmp_score) > len(noisy_labels)/10:
                gmm.fit(tmp_score.reshape(-1,1))
                print(gmm.score(tmp_score.reshape(-1,1)))
                if gmm.score(tmp_score.reshape(-1,1)) > best_log_likelihood:
                    best_log_likelihood = gmm.score(tmp_score.reshape(-1,1))
                    filter_threshold = threshold
        data_filter = np.asarray(score >= filter_threshold) * 1

    #print("filter threshold = " + str(filter_threshold))
    print("noise level according to the algorithm: " + str(1 - sum(data_filter) / len(data_filter)))
    #intersection = np.intersect1d(np.where(data_filter == 0), np.where(clean_indices == 0)).size
    #recall = intersection / len(np.where(clean_indices == 0)[0])
    #precision = intersection / len(np.where(data_filter == 0)[0])
    #print("filter precision = " + str(precision) + " and recall = " + str(recall))
    #print("f1score = " + str(precision * recall * 2 / (precision + recall)))
    return data_filter, np.asarray(all_pred)

#score1 = np.argsort(score)
#filter_threshold = score[score1[10000]]
#data_filter = np.asarray(score >= filter_threshold) * 1


def calc_filter_threshold_GMM(elp_scores, n_components):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(elp_scores.reshape(-1,1))
    print("likelihood score: "+str(gmm.score(elp_scores.reshape(-1,1))))
    indices = np.argsort(gmm.means_.flatten())
    means = gmm.means_[indices]
    covs = gmm.covariances_[indices]
    stds = np.sqrt(covs/n_components)
    m1, m2 = means[0], means[1]
    std1 = stds[0][0][0]
    std2 = stds[1][0][0]
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    intersections = (np.roots([a, b, c]))
    if intersections[0] > 1:
        res = intersections[1]
    elif intersections[0] < 0:
        res = intersections[1]
    else: res = intersections[0]
    return res, means, stds

def generate_alternative_labels(all_pred, classes_num, correction = 'support', epochs_num = 200, number_of_nets = 10, folder_name = '', Step_idx = 2, noisy_labels = []):
    if correction == 'logits':
        prev_score = np.zeros((classes_num, len(noisy_labels)))
        for epoch in tqdm(range(epochs_num)):
            # loading the current epoch
            score = np.empty((classes_num, len(noisy_labels)))
            nets_prediction_logits = []
            for i in range(30,30+number_of_nets):
                pred_logits = np.load(
                    folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'alllogits_' + str(i) + '_' + str(
                        epoch) + '.npy')
                if len(pred_logits) < len(noisy_labels):
                    raise ValueError("Net " + str(i) + " in epoch " + str(
                        epoch) + " doesn't have all the predictions for the dataset")
                else:
                    nets_prediction_logits.append(pred_logits)
            nets_prediction_logits = np.asarray(nets_prediction_logits)
            for c in range(classes_num):
                labels = np.ones(all_pred[epoch].shape) * c
                binary_all_pred = labels == all_pred[epoch]
                score[c] = np.mean(binary_all_pred * nets_prediction_logits[:, :, c], axis=0)
            score += prev_score
            prev_score = score
        score = score/epochs_num
        alternative_labels_scores = np.amax(score, axis=0)
        alternative_labels = np.argmax(score, axis=0)

    if correction == 'support':
        AL_scores = np.zeros((classes_num, len(all_pred[0][0])))
        for c in tqdm(range(classes_num)):
            labels = np.ones(all_pred[:,:number_of_nets,:].shape)*c
            binary_all_pred = labels == all_pred[:,:number_of_nets,:]
            AL_scores[c] = np.mean(binary_all_pred,axis = (0,1))
        alternative_labels_scores = np.amax(AL_scores, axis = 0)
        alternative_labels = np.argmax(AL_scores, axis = 0)
    return np.asarray(alternative_labels), np.asarray(alternative_labels_scores)

"""
def generate_alternative_labels(correction, acc_alternative_label_matrix, epochs_num, nets_num, classes_num, score_method, current_labels):
    if correction == 'first_label' or correction == 'None':
        alternative_labels = [acc_alternative_label_matrix[i].argmax() for i in
                        range(len(acc_alternative_label_matrix))]
        alternative_labels_scores = []
        alternative_label_prob_matrix = [acc_alternative_label_matrix[i]/np.sum(acc_alternative_label_matrix[i]) for i in
                        range(len(acc_alternative_label_matrix))]
        if score_method == 'entropy':
            alternative_labels_scores = [(-np.log(1/classes_num)**-1)*scipy.stats.entropy(alternative_label_prob_matrix[i]) for i in range(len(alternative_label_prob_matrix))]
        if score_method == 'support':
            alternative_labels_scores = 1-np.asarray([acc_alternative_label_matrix[i].max() for i in
                        range(len(acc_alternative_label_matrix))])/(epochs_num * nets_num)

    if correction == 'second_label':
        first_result = [(acc_alternative_label_matrix[i].argmax(), acc_alternative_label_matrix[i].max()) for i in
                        range(len(acc_alternative_label_matrix))]
        second_result = []
        for i in range(len(first_result)):
            second_best_idx = -1
            second_best_agreement = 0
            for j in range(0, len(acc_alternative_label_matrix[i])):
                if acc_alternative_label_matrix[i][j] > second_best_agreement and j != acc_alternative_label_matrix[
                    i].argmax():
                    second_best_idx = j
                    second_best_agreement = acc_alternative_label_matrix[i][j]
            second_result.append((second_best_idx, second_best_agreement))
        alternative_labels = []
        for j in range(len(current_labels)):
            if (first_result[j][0] == current_labels[j]):
                alternative_labels.append(second_result[j])
            else:
                alternative_labels.append(first_result[j])

        alternative_label_prob_matrix = [acc_alternative_label_matrix[i] / np.sum(acc_alternative_label_matrix[i]) for i
                                         in
                                         range(len(acc_alternative_label_matrix))]
        if score_method == 'entropy':
            alternative_labels_scores = [
                (-np.log(1 / classes_num) ** -1) * scipy.stats.entropy(alternative_label_prob_matrix[i]) for i in
                range(len(alternative_label_prob_matrix))]
        if score_method == 'support':
            alternative_labels_scores = 1 - np.asarray(alternative_labels)[:, 1] / (epochs_num * nets_num)
        alternative_labels = np.asarray(alternative_labels)[:, 0]

    return np.asarray(alternative_labels), np.asarray(alternative_labels_scores)
"""
def bmm_correction_threshold(alternative_labels_scores, correct_correction):
    bmm = BetaMixture1D()
    bmm.fit(alternative_labels_scores)
    x = np.linspace(0, 1, 100)
    negative = bmm.weighted_likelihood(x, 0)
    positive = bmm.weighted_likelihood(x, 1)
    dist = np.abs(negative - positive)
    bins, _ = np.histogram(alternative_labels_scores, bins=100)
    t = argrelextrema(dist[argrelextrema(dist, np.greater)[0][0]:], np.less)[0][0] + argrelextrema(dist, np.greater)[0][0]
    plt.hist(alternative_labels_scores, bins=100, color='blue', alpha=0.3)
    plt.hist(alternative_labels_scores[np.where(correct_correction == 1)[0]], bins=100, color='green', alpha=0.5)
    plt.hist(alternative_labels_scores[np.where(correct_correction == 0)[0]], bins=100, color='red', alpha=0.5)
    plt.plot(t/100,  bins[t], color = 'black', marker = 'o')
    plt.show()
    plt.figure(1)
    bmm.plot()
    plt.show()
    #plt.figure(2)
    #plt.plot(x, dist)
    #plt.show()
    return t/100
def calc_correction_thresholds(al_scores_bins, Step_idx, correction, correct_everything, first_label = False):
    if correction == 'None':
        correction_thresholds = [-1]
        correction_thresholds_names = ['None']
    else:
        if correct_everything:
            if Step_idx == 1:
                fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
                results = fp.fit(al_scores_bins)
                bin_score = np.array(results['df'].score)
                peaks = np.array(results['df'].x)
                strongest_peaks = np.argsort(bin_score)[::-1]
                if correction == 'entropy':
                    peak = peaks[strongest_peaks[0]]
                    peak_threshold = peak / len(al_scores_bins)
                    correct_everything_threshold = 1
                    zero_threshold = 0
                    half_peak_threshold = peak / (2 * len(al_scores_bins))
                    peak_one_avg_threshold = (1 + peak_threshold) / 2
                    correction_thresholds = [zero_threshold, half_peak_threshold, peak_threshold, peak_one_avg_threshold,
                                             correct_everything_threshold]
                    correction_thresholds_names = ['zero', 'half_peak', 'peak', 'almost_everything', 'correct_everything']
                if correction == 'support':
                    try:
                        indices1, indices2 = peaks[strongest_peaks[0]], peaks[strongest_peaks[1]]
                    except:
                        indices1, indices2 = 0, len(al_scores_bins) - 1
                    peak_left, peak_right = min(indices1, indices2), max(indices1, indices2)
                    smallest_bin = np.argmin(al_scores_bins[peak_left:peak_right]) + peak_left
                    easy_threshold = peak_left / len(al_scores_bins)
                    half_easy_threshold = easy_threshold / 2
                    tough_threshold = peak_right / len(al_scores_bins)
                    really_tough_threshold = (tough_threshold + 1) / 2
                    correct_everything_threshold = 1
                    zero_threshold = 0
                    min_threshold = smallest_bin / len(al_scores_bins)
                    tough_min_avg_threshold = (min_threshold + tough_threshold) / 2
                    easy_min_avg_threshold = (min_threshold + easy_threshold) / 2
                    correction_thresholds = [zero_threshold, half_easy_threshold, easy_threshold, easy_min_avg_threshold, min_threshold,
                                             tough_min_avg_threshold, tough_threshold, really_tough_threshold,
                                             correct_everything_threshold]
                    correction_thresholds_names = ['zero', 'half_easy', 'easy', 'easy_min_avg', 'min', 'min_tough_avg', 'tough',
                                                   'really_tough', 'correct_everything']
            else: #maybe just take the formula in the documentation for now... it might be better for now (because derivative  won't work in CIFAR100)
                normalized_bins = al_scores_bins / max(al_scores_bins)
                distance_from_origin = np.sqrt(normalized_bins**2+(1-(np.asarray(range(len(al_scores_bins)))/len(al_scores_bins)))**2)
                highest_bin_idx = np.argmax(normalized_bins)
                closest_bin_idx = np.argmin(distance_from_origin[:highest_bin_idx])
                closest_highest_avg = (highest_bin_idx+closest_bin_idx)/2
                #correction_thresholds = [highest_bin_idx/len(al_scores_bins), closest_highest_avg/len(al_scores_bins), closest_bin_idx/len(al_scores_bins)]
                #correction_thresholds_names = ['peak', 'peak_closest_avg', 'closest']
                correction_thresholds = [closest_bin_idx/len(al_scores_bins)]
                correction_thresholds_names = ['closest']

                """
                fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
                results = fp.fit(al_scores_bins)
                bin_score = np.array(results['df'].score)
                peaks = np.array(results['df'].x)
                strongest_peaks = np.argsort(bin_score)[::-1]
                peak = peaks[strongest_peaks[0]]
                peak_threshold = peak / len(al_scores_bins)
                almost_threshold = peak_threshold + al_scores_bins.argmin() / (2 * len(al_scores_bins))
                peak_almost_avg = (peak_threshold + almost_threshold) / 2
                correction_thresholds = [peak_threshold, peak_almost_avg, almost_threshold]
                correction_thresholds_names = ['peak', 'after_peak', 'almost_everything']
                
                diff_low = al_scores_bins[1:len(al_scores_bins) - 1] - al_scores_bins[2:len(al_scores_bins)]
                diff_high = al_scores_bins[0:len(al_scores_bins) - 2] - al_scores_bins[1:len(al_scores_bins) - 1]
                diff_score = diff_high / diff_low
                idx = np.nan_to_num(diff_score, posinf=0, neginf=0).argmax()
                mid_threshold = idx / len(al_scores_bins)
                correct_everything_threshold = 1
                correction_thresholds = [peak_threshold, mid_threshold, correct_everything_threshold]
                correction_thresholds_names = ['peak', 'mid', 'correct_everything']
                
                if correction == 'support':
                    peak = peaks[strongest_peaks[0]]
                    zero = 0
                    peak_threshold = peak / len(al_scores_bins)
                    almost_threshold = peak_threshold+al_scores_bins.argmin()/(2*len(al_scores_bins))
                    correction_thresholds = [zero,peak_threshold, almost_threshold]
                    correction_thresholds_names = ['zero','peak', 'almost_everything']
                if correction == 'entropy':
                    peak = peaks[strongest_peaks[0]]
                    zero = 0
                    peak_threshold = peak / len(al_scores_bins)
                    almost_threshold = peak_threshold + al_scores_bins.argmin() / (2*len(al_scores_bins))
                    peak_almost_avg = (peak_threshold+almost_threshold) / 2
                    correction_thresholds = [peak_threshold, peak_almost_avg, almost_threshold]
                    correction_thresholds_names = ['peak', 'after_peak', 'almost_everything']
                """


        else:
            if Step_idx == 1:
                if correction == 'entropy':
                    fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
                    results = fp.fit(al_scores_bins)
                    bin_score = np.array(results['df'].score)
                    peaks = np.array(results['df'].x)
                    strongest_peaks = np.argsort(bin_score)[::-1]
                    peak = peaks[strongest_peaks[0]]
                    peak_threshold = peak/len(al_scores_bins)
                    correct_everything_threshold = 1
                    zero_threshold = 0
                    half_peak_threshold = peak / (2*len(al_scores_bins))
                    peak_one_avg_threshold = (1+peak_threshold)/2
                    correction_thresholds = [zero_threshold, half_peak_threshold, peak_threshold, peak_one_avg_threshold, correct_everything_threshold]
                    correction_thresholds_names = ['zero', 'half_peak', 'peak', 'almost_everything','correct_everything']
                if correction == 'support':
                    fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
                    results = fp.fit(al_scores_bins)
                    bin_score = np.array(results['df'].score)
                    peaks = np.array(results['df'].x)
                    strongest_peaks = np.argsort(bin_score)[::-1]
                    peak = peaks[strongest_peaks[0]]
                    if first_label:
                        diff_high = al_scores_bins[1:len(al_scores_bins)-1] - al_scores_bins[0:len(al_scores_bins)-2]
                        highest_jump_threshold = diff_high.argmax()/len(al_scores_bins)
                        peak_threshold = peak/len(al_scores_bins)
                        correct_everything_threshold = 1
                        zero_threshold = 0
                        peak_one_avg_threshold = (1+peak_threshold)/2
                        correction_thresholds = [zero_threshold, highest_jump_threshold, peak_threshold, peak_one_avg_threshold, correct_everything_threshold]
                        correction_thresholds_names = ['zero', 'highest_jump', 'peak', 'almost_everything','correct_everything']
                    else:
                        peak_threshold = peak / len(al_scores_bins)
                        correct_everything_threshold = 1
                        zero_threshold = 0
                        half_peak_threshold = peak / (2 * len(al_scores_bins))
                        peak_one_avg_threshold = (1 + peak_threshold) / 2
                        correction_thresholds = [zero_threshold, half_peak_threshold, peak_threshold,
                                                 peak_one_avg_threshold, correct_everything_threshold]
                        correction_thresholds_names = ['zero', 'half_peak', 'peak', 'almost_everything',
                                                       'correct_everything']

            else:
                normalized_bins = al_scores_bins / max(al_scores_bins)
                distance_from_origin = np.sqrt(
                    normalized_bins ** 2 + (np.asarray(range(len(al_scores_bins))) / len(al_scores_bins)) ** 2)
                highest_bin_idx = np.argmax(normalized_bins)
                closest_bin_idx = np.argmin(distance_from_origin[highest_bin_idx:])
                closest_highest_avg = (highest_bin_idx + closest_bin_idx) / 2
                correction_thresholds = [highest_bin_idx / len(al_scores_bins),
                                         closest_highest_avg / len(al_scores_bins),
                                         closest_bin_idx / len(al_scores_bins)]
                correction_thresholds_names = ['peak', 'peak_closest_avg', 'closest']
                """
                fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
                results = fp.fit(al_scores_bins)
                bin_score = np.array(results['df'].score)
                peaks = np.array(results['df'].x)
                strongest_peaks = np.argsort(bin_score)[::-1]
                try:
                    indices1, indices2 = peaks[strongest_peaks[0]], peaks[strongest_peaks[1]]
                except:
                    indices1, indices2 = 0, len(al_scores_bins) - 1
                peak_left, peak_right = min(indices1, indices2), max(indices1, indices2)
                smallest_bin = np.argmin(al_scores_bins[peak_left:peak_right]) + peak_left
                peak_threshold = peak_left / len(al_scores_bins)
                mid_threshold = smallest_bin / len(al_scores_bins)
                peak_mid_avg = (peak_threshold+mid_threshold) / 2
                correction_thresholds = [peak_threshold, peak_mid_avg, mid_threshold]
                correction_thresholds_names = ['peak', 'peak_mid_avg', 'mid']
                """
    return correction_thresholds, correction_thresholds_names

def generate_new_labels(correction_threhold, old_labels, data_filter, new_labels, new_labels_scores):
    new_labels_filter = (np.asarray(new_labels_scores) > correction_threhold)*1 #indicator vector for which new label we believe is "correct" a correct one
    correct_new_labels_indices = np.nonzero(new_labels_filter)[0] #indices for new labels we want to take
    clean_labels_indices = np.setdiff1d(np.nonzero(data_filter)[0], correct_new_labels_indices) #indices for data we believe that is clean, and didn't havee a "new" label for
    #old_labels[correct_new_labels_indices] = new_labels[correct_new_labels_indices] #replacing our noisy labels with alternative labels in the "correct" new labels indicnes
    #final_labels = old_labels
    final_labels = old_labels.copy()
    final_labels[correct_new_labels_indices] = new_labels[correct_new_labels_indices]
    data_remove = np.delete(range(len(old_labels)), np.union1d(clean_labels_indices, correct_new_labels_indices))
    return final_labels, data_remove, correct_new_labels_indices, new_labels_filter

def save_new_dataset(path, new_labels, data_remove, clean_train_dataset_targets):
    if not os.path.exists(path):
        os.makedirs(path)
    clean_indices = np.asarray(new_labels) == np.asarray(clean_train_dataset_targets)
    np.save(path+'clean_indices.npy', clean_indices)
    np.save(path+'noisy_labels.npy', new_labels)
    np.save(path+'final_remove.npy', data_remove)

def print_results(f, folder_names, Step_idx, net):
    for folder_name in folder_names:
        print(folder_name)
        folder_name = f + folder_name + '/'
        test_acc = []
        for i in range(3):
            try:
                net.load_parameters(
                    path=folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(
                        i) + '_' + str(
                        199) + '.pth',
                    indices_file=folder_name + 'step' + str(Step_idx) + "/resnet" + str(0) + '/' + 'indices.txt')
                print("net " + str(i) + " test accuracy = " + str(max(net.test_accuracies)))
                test_acc.append(max(net.test_accuracies))
            except:
                print(folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(
                        i) + '_' + str(
                        199) + '.pth')
        print("mean = "+str(np.mean(test_acc))+" std = "+str(np.std(test_acc)))

def ablation_results(root, folder_names, Step_idx, net):
    for folder_name in folder_names:
        print(folder_name)
        f = root + folder_name + '/'
        test_acc = []
        for j in range(1,11):
            path = f+'ablation/'+str(j)+'netssupportCorrection/closest/step3/resnet0/resnet_0_199.pth'
            try:
                net.load_parameters(
                    path = path,
                    indices_file=folder_name + 'step' + str(Step_idx) + "/resnet" + str(0) + '/' + 'indices.txt')
                test_acc.append(max(net.test_accuracies))
            except:
                print(path)
        print(test_acc)
        try:
            plt.title(folder_name)
            plt.plot(range(1,11), test_acc)
            plt.show()
        except:
            print("failed")

def daniel_noise_function(save_dir, trainset, subset, noise_size):
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass

    trainsubset = copy.deepcopy(trainset)
    trainsubset.data = [trainsubset.data[index] for index in subset]
    trainsubset.targets = [trainsubset.targets[index] for index in subset]
    num_of_classes = np.unique(trainset.targets).size
    noise_size_percents = (noise_size/len(trainset.targets))

    random_state_generator = np.random.RandomState(0)
    indices_to_shuffle = random_state_generator.choice(a=subset,size=noise_size,replace = False)
    shuffled_indices = random_state_generator.permutation(indices_to_shuffle)
    random_state_generator = np.random.RandomState(0)
    noise = random_state_generator.randint(1, len(trainsubset.classes),shuffled_indices.size)
    clean_indices = np.ones(len(trainset))
    clean_indices[indices_to_shuffle] = 0
    for i, idx in enumerate(indices_to_shuffle):
        trainsubset.targets[idx] = (trainsubset.targets[idx] + noise[i]) % len(trainsubset.classes)
    np.save(save_dir+'clean_indices.npy', clean_indices)
    np.save(save_dir+'noisy_labels.npy', trainsubset.targets)

def create_oracle_filter(path, clean_indices):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass
    np.save(path+'final_remove.npy', np.where(clean_indices == 0)[0])

def create_random_filter(path, noise_amount, total_data = 50000):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass
    noise_indices = random.sample(range(total_data), noise_amount)
    np.save(path+'final_remove.npy', noise_indices)

def calc_f1(folder_name, epochs_num = 200, number_of_nets = 10, Step_idx = 1):
    print(folder_name)
    clean_indices = np.load(folder_name + 'step1/clean_indices.npy')
    noisy_labels = np.load(folder_name + 'step1/noisy_labels.npy')
    prev_score = np.zeros(len(noisy_labels))
    all_pred = []
    for epoch in range(epochs_num):
        all_pred.append([])
        # loading the current epoch
        nets_prediction = []
        nets_binary_prediction = []
        for i in range(number_of_nets):
            pred = np.load(
                folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(i) + '_' + str(
                    epoch) + '.npy')
            if len(pred) < len(noisy_labels):
                raise ValueError("Net " + str(i) + " in epoch " + str(
                    epoch) + " doesn't have all the predictions for the dataset")
            else:
                all_pred[epoch].append(pred == noisy_labels)
    all_pred = np.asarray(all_pred)
    score = np.mean(all_pred, axis = (0,1))
    bmm = BetaMixture1D()
    bmm.fit(score)
    clean_prediction = bmm.predict(score)
    clean_label = True
    data_filter = clean_prediction == clean_label
    intersection = np.intersect1d(np.where(data_filter == 0), np.where(clean_indices == 0)).size
    recall = intersection / len(np.where(clean_indices == 0)[0])
    precision = intersection / len(np.where(data_filter == 0)[0])
    #print("filter precision = " + str(precision) + " and recall = " + str(recall))
    print("f1score = " + str(precision * recall * 2 / (precision + recall)))

def curriculm_filtration(path, noise_level, dataset = 'cifar10', vanilla = False):
    if dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        clean_train_dataset = datasets.CIFAR10('data/cifar', train=True, download=True, transform=train_transform)
    if dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        clean_train_dataset = datasets.CIFAR100('data/cifar', train=True, download=True, transform=train_transform)
    noisy_labels = np.load(path+'noisy_labels_'+dataset+'_noise_'+str(noise_level)+'.npy')
    clean_indices = noisy_labels == np.asarray(clean_train_dataset.targets)
    if vanilla:
        vanilla_score = np.load(path+'elp_vanilla_'+dataset+'_noise_'+str(noise_level)+'.npy')
        plt.hist(vanilla_score[np.where(clean_indices == 1)[0]], bins=100, color='blue', alpha=0.5, label="clean data")
        plt.hist(vanilla_score[np.where(clean_indices == 0)[0]], bins=100, color='orange', alpha=0.5,
                 label="noisy data")
        plt.title("vanilla elp distribution")
        plt.legend()
        plt.savefig(path + dataset + '_noise_' + str(noise_level) + 'vanilla_elp_dist.jpg')
        plt.show()
        bmm = BetaMixture1D()
        bmm.fit(vanilla_score)
        x = np.linspace(0, 1, 100)
        plt.plot(x, bmm.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, bmm.weighted_likelihood(x, 1), label='positive')
        plt.legend()
        plt.savefig(path + dataset + '_noise_' + str(noise_level) + 'vanilla_bmm_fit.jpg')
        plt.show()

        clean_prediction = bmm.predict(vanilla_score)
        clean_label = True

        vanilla_original_data_filter = clean_prediction == clean_label
        noise_est = 1 - sum(vanilla_original_data_filter) / len(vanilla_original_data_filter)
        intersection = np.intersect1d(np.where(vanilla_original_data_filter == 0), np.where(clean_indices == 0)).size
        recall = intersection / len(np.where(clean_indices == 0)[0])
        precision = intersection / len(np.where(vanilla_original_data_filter == 0)[0])
        print("vanilla original noise level according to the algorithm: " + str(noise_est))
        print("vanilla original filter precision = " + str(precision) + " and recall = " + str(recall))
        print("vanilla original f1score = " + str(precision * recall * 2 / (precision + recall)))

        f = open(path + dataset + '_noise_' + str(noise_level) + 'results.txt', 'a')
        f.write("vanilla original noise level according to the algorithm: " + str(noise_est))
        f.write("\nvanilla original filter precision = " + str(precision) + " and recall = " + str(recall))
        f.write("\nvanilla original f1score = " + str(precision * recall * 2 / (precision + recall)))
        f.close()

        vanilla_new_data_filter = np.ones(len(noisy_labels))
        vanilla_new_data_filter[np.argsort(vanilla_score)[:int(len(noisy_labels) * noise_est)]] = 0
        intersection = np.intersect1d(np.where(vanilla_new_data_filter == 0), np.where(clean_indices == 0)).size
        recall = intersection / len(np.where(clean_indices == 0)[0])
        precision = intersection / len(np.where(vanilla_new_data_filter == 0)[0])
        print("vanilla new filter precision = " + str(precision) + " and recall = " + str(recall))
        print("vanilla new f1score = " + str(precision * recall * 2 / (precision + recall)))

        f = open(path + dataset + '_noise_' + str(noise_level) + 'results.txt', 'a')
        f.write("\nvanilla new filter precision = " + str(precision) + " and recall = " + str(recall))
        f.write("\nvanilla new f1score = " + str(precision * recall * 2 / (precision + recall)))
        f.close()

    curriculum_score = np.load(path+'elp_curriculum_'+dataset+'_noise_'+str(noise_level)+'.npy')
    plt.hist(curriculum_score[np.where(clean_indices == 1)[0]], bins=100, color='blue', alpha=0.5, label = "clean data")
    plt.hist(curriculum_score[np.where(clean_indices == 0)[0]], bins=100, color='orange', alpha=0.5, label = "noisy data")
    plt.title("curriculum elp distribution")
    plt.legend()
    plt.savefig(path+dataset+'_noise_'+str(noise_level)+'curriculum_elp_dist.jpg')
    plt.show()

    bmm = BetaMixture1D()
    bmm.fit(curriculum_score)
    x = np.linspace(0, 1, 100)
    plt.plot(x, bmm.weighted_likelihood(x, 0), label='negative')
    plt.plot(x, bmm.weighted_likelihood(x, 1), label='positive')
    plt.legend()
    plt.savefig(path +dataset+'_noise_'+ str(noise_level) + 'curriculum_bmm_fit.jpg')
    plt.show()

    clean_prediction = bmm.predict(curriculum_score)
    clean_label = True

    curriculum_original_data_filter = clean_prediction == clean_label
    noise_est = 1 - sum(curriculum_original_data_filter) / len(curriculum_original_data_filter)
    intersection = np.intersect1d(np.where(curriculum_original_data_filter == 0), np.where(clean_indices == 0)).size
    recall = intersection / len(np.where(clean_indices == 0)[0])
    precision = intersection / len(np.where(curriculum_original_data_filter == 0)[0])
    print("curriculum original noise level according to the algorithm: " + str(noise_est))
    print("curriculum original filter precision = " + str(precision) + " and recall = " + str(recall))
    print("curriculum original f1score = " + str(precision * recall * 2 / (precision + recall)))
    f = open(path +dataset+'_noise_'+ str(noise_level) + 'results.txt', 'a')
    f.write("\ncurriculum original noise level according to the algorithm: " + str(noise_est))
    f.write("\ncurriculum original filter precision = " + str(precision) + " and recall = " + str(recall))
    f.write("\ncurriculum original f1score = " + str(precision * recall * 2 / (precision + recall)))
    f.close()

    curriculum_new_data_filter = np.ones(len(noisy_labels))
    curriculum_new_data_filter[np.argsort(curriculum_score)[:int(len(noisy_labels)*noise_est)]] = 0
    intersection = np.intersect1d(np.where(curriculum_new_data_filter == 0), np.where(clean_indices == 0)).size
    recall = intersection / len(np.where(clean_indices == 0)[0])
    precision = intersection / len(np.where(curriculum_new_data_filter == 0)[0])
    print("curriculum new filter precision = " + str(precision) + " and recall = " + str(recall))
    print("curriculum new f1score = " + str(precision * recall * 2 / (precision + recall)))
    np.save(path +dataset+'_noise_'+ str(noise_level) +'_filtration.npy', curriculum_new_data_filter)
    f = open(path +dataset+'_noise_'+ str(noise_level) + 'results.txt', 'a')
    f.write("\ncurriculum new filter precision = " + str(precision) + " and recall = " + str(recall))
    f.write("\ncurriculum new f1score = " + str(precision * recall * 2 / (precision + recall)))
    f.close()

def main():
    transform = transforms.Compose(
        [transforms.RandomCrop(64, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    clean_train_dataset = TinyImageNet(root='', split='train', transform=transform)
    noise_size = int(0.2*len(clean_train_dataset.targets))
    subset = np.arange(len(clean_train_dataset.targets))
    trainsubset = copy.deepcopy(clean_train_dataset)
    trainsubset.data = [trainsubset.data[index] for index in subset]
    trainsubset.targets = [trainsubset.targets[index] for index in subset]
    num_of_classes = np.unique(clean_train_dataset.targets).size
    noise_size_percents = (noise_size / len(clean_train_dataset.targets))

    random_state_generator = np.random.RandomState(0)
    indices_to_shuffle = random_state_generator.choice(a=subset, size=noise_size, replace=False)
    shuffled_indices = random_state_generator.permutation(indices_to_shuffle)
    random_state_generator = np.random.RandomState(0)
    noise = random_state_generator.randint(1, len(trainsubset.classes), shuffled_indices.size)
    clean_indices = np.ones(len(clean_train_dataset))
    clean_indices[indices_to_shuffle] = 0
    for i, idx in enumerate(indices_to_shuffle):
        trainsubset.targets[idx] = (trainsubset.targets[idx] + noise[i]) % len(trainsubset.classes)
    noisy_labels = np.asarray(trainsubset.targets)
    all_pred = np.empty((3,200,100000))
    for i in tqdm(range(3)):
        all_pred[i] = np.argmax(np.load('/cs/snapless/daphna/daniels44/TinyImagenet20net'+str(i)+'.npy'),axis = 2)
    score = np.mean(all_pred == np.asarray(trainsubset.targets), axis = (0,1))
    bmm = BetaMixture1D()
    bmm.fit(score)
    clean_prediction = bmm.predict(score)
    clean_label = True
    data_filter = clean_prediction == clean_label
    print("noise level according to the algorithm: " + str(1 - sum(data_filter) / len(data_filter)))
    intersection = np.intersect1d(np.where(data_filter == 0), np.where(clean_indices == 0)).size
    recall = intersection / len(np.where(clean_indices == 0)[0])
    precision = intersection / len(np.where(data_filter == 0)[0])
    print("filter precision = " + str(precision) + " and recall = " + str(recall))
    print("f1score = " + str(precision * recall * 2 / (precision + recall)))

    curriculm_filtration(path = '/cs/labs/daphna/uri1234/guy/', noise_level = 0.6, dataset='cifar100')
    curriculm_filtration(path = '/cs/labs/daphna/uri1234/guy/', noise_level = 0.4, dataset='cifar100')
    curriculm_filtration(path = '/cs/labs/daphna/uri1234/guy/', noise_level = 0.2, dataset='cifar100')
    calc_f1('/cs/labs/daphna/uri1234/Asymmetricnoise/CIFAR100/40lb/')
    calc_f1('/cs/labs/daphna/uri1234/symmetricnoise/Dense/CIFAR100/20lb/')
    calc_f1('/cs/labs/daphna/uri1234/symmetricnoise/Dense/CIFAR100/40lb/')
    calc_f1('/cs/labs/daphna/uri1234/symmetricnoise/Dense/CIFAR100/60lb/')
    calc_f1('/cs/labs/daphna/uri1234/Asymmetricnoise/CIFAR100/10lb/')
    calc_f1('/cs/labs/daphna/uri1234/Asymmetricnoise/CIFAR100/20lb/')
    calc_f1('/cs/labs/daphna/uri1234/symmetricnoise/Dense/CIFAR10/20lb/')
    calc_f1('/cs/labs/daphna/uri1234/symmetricnoise/Dense/CIFAR10/40lb/')
    calc_f1('/cs/labs/daphna/uri1234/symmetricnoise/Dense/CIFAR10/60lb/')
    calc_f1('/cs/labs/daphna/uri1234/Asymmetricnoise/CIFAR10/10lb/')
    calc_f1('/cs/labs/daphna/uri1234/Asymmetricnoise/CIFAR10/20lb/')
    calc_f1('/cs/labs/daphna/uri1234/Asymmetricnoise/CIFAR10/40lb/')
    sys.exit(0)
    parser = argparse.ArgumentParser(description='training nets')
    parser.add_argument('--dataset', default = 'Imagenet100', help='dataset')
    parser.add_argument('--folder', default = '/cs/labs/daphna/uri1234/symmetricnoise/postprocess/Imagenet100/clean/', help='folder to save the net')
    parser.add_argument('--Step', type=int, default =1, help='folder to save the net')
    parser.add_argument('--filter', default = 'BMM', help='test used for filter')
    parser.add_argument('--correction', default = 'support', help='test used for filter')
    parser.add_argument('--zeros', default = 'True', help='test used for filter')
    parser.add_argument('--nets_for_correction', type=int, default = 3, help='folder to save the net')

    args = parser.parse_args()
    dataset = args.dataset
    folder_name = args.folder
    Step_idx = args.Step
    filter = args.filter
    score_methods = [args.correction]
    zeros = [args.zeros == 'True']
    total_number_nets_correction = args.nets_for_correction


    epochs_num = 200
    #filter = 'AccPerLT'

    if dataset == 'animal':
        classes_num = 10
    if dataset == 'CIFAR10':
        classes_num = 10
    if dataset == 'CIFAR100':
        classes_num = 100
    if dataset == 'Imagenet100':
        classes_num = 100
    if dataset == 'Imagenet50':
        classes_num = 50
        epochs_num = 200
    if dataset == 'TinyImagenet':
        classes_num = 200
    if dataset == 'webvision':
        classes_num = 50
    if dataset == 'clothing':
        classes_num = 100

    correction = 'first_label'
    correct_everything = False
    score_method = 'support'   
    if correct_everything:
        s = 'ce'
    else: s = 'cn'
    growth = 32
    number_of_nets = 5
    #labels_path = folder_name + 'step1/indices.txt'
    labels_path = folder_name +'step'+str(Step_idx)
    net = DenseNet(symmetric=True, noiselevel=0.3, num_classes=classes_num, growth = growth, dataset=dataset)

    corrections = ['first_label']
    correct_everythings = [True]

    #print_results(f, folder_names, Step_idx, net)
    """
    folder_names = ['/cs/labs/daphna/uri1234/asymmetricnoise/conv9/CIFAR100/40lb/', '/cs/labs/daphna/uri1234/asymmetricnoise/conv9/CIFAR100/20lb/', '/cs/labs/daphna/uri1234/symmetricnoise/conv9/CIFAR100/60lb/', '/cs/labs/daphna/uri1234/symmetricnoise/conv9/CIFAR100/20lb/']
    for folder_name in folder_names:
        net = DenseNet(symmetric=True, noiselevel=0.3, num_classes=100, growth=growth, dataset='CIFAR100', conv9 = True)
        print(folder_name)
        try:
            print_results(folder_name, [''], 1, net)
        except:
            print("failed")
    """
    print(folder_name)
    # loading data
    train_transform = transforms.Compose([
        #utilv3.Cutout(num_cutouts=2, size=8, p=0.8),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                         ])
    test_acc = []
    """
    for i in range(40,50):
        try:
            net.load_parameters(
                path=folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(
                    i) + '_' + str(
                    199) + '.pth',
                indices_file=folder_name + 'step' + str(Step_idx) + "/resnet" + str(0) + '/' + 'indices.txt')
            print("net " + str(i) + " test accuracy = " + str(max(net.test_accuracies)))
            test_acc.append(max(net.test_accuracies))
        except:
            print("net "+str(i)+" load failed")
    print("mean = " + str(np.mean(test_acc)) + " std = " + str(np.std(test_acc)))
    """
    if dataset == 'CIFAR10':
        clean_train_dataset = datasets.CIFAR10('data/cifar', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10('data/cifar', train=False, download=True, transform=test_transform)
    if dataset == 'CIFAR100':
        clean_train_dataset = datasets.CIFAR100('data/cifar', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100('data/cifar', train=False, download=True, transform=test_transform)
    if dataset == 'TinyImagenet':
        transform = transforms.Compose(
            [transforms.RandomCrop(64, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        clean_train_dataset = TinyImageNet(root='', split='train', transform=transform)
        test_dataset = TinyImageNet(root='', split='val', transform=test_transform)
    if dataset == 'Imagenet50':
        transform = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        test_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        clean_train_dataset = MyImageNet(split='train', num_classes=50, transform=transform)
        test_dataset = MyImageNet(split='val', num_classes=50, transform=test_transform)

    if dataset == 'Imagenet100':
        transform = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        test_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        clean_train_dataset = MyImageNet(split='train', num_classes=100, transform=transform)
        test_dataset = MyImageNet(split='val', num_classes=100, transform=test_transform)
    if dataset == 'clothing':
        num_classes = 100
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        clean_train_dataset = myClothing1m.clothing_dataset(root='/cs/labs/daphna/data/clothing1m', transform=transform)

    if dataset == 'webvision':
        batch_size = 32
        classes_num = 50
        loader = webvision_dataloader.webvision_dataloader(batch_size=batch_size, num_workers=4,
                                                           root_dir='/cs/labs/daphna/uri1234/mywebvision/',
                                                           num_class=50)
        trainloader = loader.run('train')
        testloader = loader.run('test')
        test_dataset = testloader.dataset
        clean_train_dataset = trainloader.dataset


    #calc_test_dist(folder_name, number_of_nets, epochs_num, classes_num, test_dataset.targets, Step_idx)

    """
    ###############################
    all_binary_pred = []
    logits_array = []
    for i in tqdm(range(number_of_nets)):
        all_binary_pred.append([])
        logits_array.append([])

        for epoch in range(epochs_num):
            pred = np.load(
                folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'resnet_' + str(i) + '_' + str(
                    epoch) + '.npy')
            if len(pred) < len(noisy_labels):
                raise ValueError("Net " + str(i) + " in epoch " + str(
                    epoch) + " doesn't have all the predictions for the dataset")
            all_binary_pred[i].append(pred == noisy_labels)

            logits = np.load(
                folder_name + 'step' + str(Step_idx) + "/resnet" + str(i) + "/" + 'clogits_' + str(i) + '_' + str(
                    epoch) + '.npy')
            if len(pred) < len(noisy_labels):
                raise ValueError("Net " + str(i) + " in epoch " + str(
                    epoch) + " doesn't have all the logits for the dataset")
            logits_array[i].append(logits)

    all_binary_pred = np.swapaxes(np.asarray(all_binary_pred), 0, 2)
    logits_array = np.swapaxes(np.asarray(logits_array), 0, 2)
    equation_vals = createMyGraph(all_binary_pred, logits_array, clean_indices)
    ###############################
    
    folder_names = [

                    '/cs/labs/daphna/uri1234/Asymmetricnoise/CIFAR10/10lb/',
                    '/cs/labs/daphna/uri1234/Asymmetricnoise/CIFAR10/20lb/',
                    '/cs/labs/daphna/uri1234/Asymmetricnoise/CIFAR10/40lb/']
    for folder_name in folder_names:
        #filters = ['BMM']
        filters = []
        for filter in filters:
            print(filter)
            data_filter, all_pred = filter_noisy_dataset(filter, folder_name, number_of_nets, epochs_num, noisy_labels,
                                                         Step_idx, classes_num, clean_indices)
        calc_test_dist(folder_name, number_of_nets, epochs_num, classes_num, test_dataset.targets, Step_idx)
    """
    labels_path = folder_name + 'step' + str(Step_idx)
    noisy_labels, clean_indices = get_original_noisy_dataset(labels_path)
    if dataset == 'animal':
        filenames = os.listdir('/cs/labs/daphna/uri1234/animal/training/')
        noisy_labels = []
        for filename in filenames:
            noisy_labels.append(int(filename[0]))
    if dataset == 'webvision' or dataset == 'clothing':
        noisy_labels = clean_train_dataset.targets
        clean_indices = np.ones(len(noisy_labels))
        clean_indices[:int(len(clean_indices)/2)] = 0
    noisy_labels = np.asarray(noisy_labels)
    data_filter, all_pred = filter_noisy_dataset(filter, folder_name, number_of_nets, epochs_num, noisy_labels,
                                                 Step_idx, classes_num, clean_indices)
    #try:
    #    calc_test_dist(folder_name, number_of_nets, epochs_num, classes_num, test_dataset.targets, Step_idx)
    #except:
    #    pass
    noisy_idx = np.where(data_filter == 0)[0]
    for zero in zeros:
        if zero:
            save_new_dataset(folder_name + 'step' + str(
                Step_idx+1) + "/", noisy_labels, noisy_idx, clean_train_dataset.targets)

        else:
            for correction in corrections:
                for score_method in score_methods:
                    for correct_everything in correct_everythings:
                        for number_nets_correction in range(1,total_number_nets_correction+1): #use for correction ablation, delete later
                            if correct_everything:
                                s = 'ce'
                            else:
                                s = 'cn'
                            alternative_labels, alternative_labels_scores = generate_alternative_labels(all_pred,
                                                                                                        classes_num,  correction = score_method, number_of_nets= number_nets_correction, folder_name = folder_name, Step_idx=Step_idx, noisy_labels=noisy_labels)
                            correct_correction = alternative_labels == np.asarray(clean_train_dataset.targets)
                            if correct_everything:
                                correct_nl_scores = alternative_labels_scores[np.nonzero(correct_correction)]
                                incorrect_nl_scores = alternative_labels_scores[np.where(correct_correction==0)[0]]
                                plt.hist(alternative_labels_scores, bins=100, color = 'blue')
                                plt.hist(correct_nl_scores, bins=100, color = 'green')
                                plt.hist(incorrect_nl_scores, bins=100, color= 'red')
                                plt.title("correct by: "+correction +" everything: "+str(correct_everything) +" score method: "+score_method)
                            else:
                                correct_nl_scores = alternative_labels_scores[np.intersect1d(noisy_idx,np.nonzero(correct_correction))]
                                incorrect_nl_scores = alternative_labels_scores[np.intersect1d(noisy_idx,np.where(correct_correction == 0)[0])]
                                plt.hist(alternative_labels_scores[noisy_idx], bins=100, color='blue')
                                plt.hist(correct_nl_scores, bins=100, color='green')
                                plt.hist(incorrect_nl_scores, bins=100, color='red')
                                plt.title(
                                    "correct by: " + correction + " everything: " + str(correct_everything) + " score method: " + score_method)

                            if correct_everything:
                                al_scores_bins, _ = np.histogram(alternative_labels_scores, bins = 100)
                            else:
                                al_scores_bins, _ = np.histogram(alternative_labels_scores[noisy_idx], bins = 100)
                            if score_method == 'support':
                                correction_thresholds, correction_thresholds_names = calc_correction_thresholds(al_scores_bins, Step_idx, score_method, correct_everything)
                                #bmm_correction = bmm_correction_threshold(alternative_labels_scores, correct_correction)
                                #correction_thresholds.append(bmm_correction)
                                #correction_thresholds_names.append('bmm_intersection')
                                correct_nl_scores = alternative_labels_scores[np.nonzero(correct_correction)]
                                incorrect_nl_scores = alternative_labels_scores[np.where(correct_correction == 0)[0]]
                                plt.hist(alternative_labels_scores, bins=100, color='blue')
                                plt.hist(correct_nl_scores, bins=100, color='green')
                                plt.hist(incorrect_nl_scores, bins=100, color='red')
                                plt.title("correct by: " + correction + " everything: " + str(
                                    correct_everything) + " score method: " + score_method)
                                for t in correction_thresholds:
                                    plt.plot(t, al_scores_bins[min(int(t*100), len(al_scores_bins) - 1)], color = 'black', marker = 'o')
                                plt.show()
                                plt.savefig(folder_name+'step2/'+score_method+'Correction.png')
                            if score_method == 'logits':
                                bins, _ = np.histogram(alternative_labels_scores, bins=100)
                                fp = findpeaks(method='topology', interpolate=None, limit=None, verbose=0)
                                results = fp.fit(bins)
                                bin_score = np.array(results['df'].score)
                                peaks = np.array(results['df'].x)
                                strongest_peaks = np.argsort(bin_score)[::-1]
                                try:
                                    indices1, indices2 = peaks[strongest_peaks[0]], peaks[strongest_peaks[1]]
                                except:
                                    indices1, indices2 = 0, len(bins) - 1
                                peak_left, peak_right = min(indices1, indices2), max(indices1, indices2)
                                smallest_bin = np.argmin(bins[peak_left:peak_right]) + peak_left
                                correction_threshold = smallest_bin
                                plt.plot(correction_threshold, al_scores_bins[correction_threshold], color = 'black', marker = 'o')
                                plt.show()
                                plt.savefig(folder_name+'step2/'+score_method+'Correction.png')
                                correction_thresholds = [correction_threshold]
                                correction_thresholds_names = ['bimodal']
                            for correction_threshold_name, correction_threshold in zip(correction_thresholds_names, correction_thresholds):
                                final_labels, data_remove, correct_new_labels_indices, new_labels_filter = generate_new_labels(correction_threshold, noisy_labels, data_filter, alternative_labels, alternative_labels_scores)
                                s1 = np.setdiff1d(correct_new_labels_indices, np.nonzero(data_filter)[0])
                                intersection = np.intersect1d(np.where(new_labels_filter == 0), np.where(correct_correction == 0)).size
                                recall = intersection / len(np.where(correct_correction == 0)[0])
                                precision = intersection / len(np.where(new_labels_filter == 0)[0])
                                print("threshold: " + correction_threshold_name + " value: " + str(correction_threshold))
                                print("correction precision = " + str(precision) + " and recall = " + str(recall))
                                print("f1score = " + str(precision * recall * 2 / (precision + recall)))
                                print("Additional data from correction = " + str(
                                    len(s1))+" correct = "+str(len(np.intersect1d(s1, np.where(correct_correction == 1))))+" incorrect = "+str(len(np.intersect1d(s1, np.where(correct_correction == 0)))))
                                save_new_dataset(folder_name+score_method+'Correction/'+correction_threshold_name+'/step3/', final_labels, data_remove, clean_train_dataset.targets)

if __name__ == "__main__":
    main()





