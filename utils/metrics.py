from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import numpy as np
from sklearn import metrics


# for decompensation, in-hospital mortality

def print_metrics_binary(y_true, predictions, verbose=1, cut_off = None):
    #predictions = np.array(predictions).reshape(predictions.shape[0])
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))
    #print(predictions)
    
    false_pos_rate, true_pos_rate, proba = metrics.roc_curve(y_true, predictions[:, 1])
    if (cut_off == None):
        optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
    else:
        optimal_proba_cutoff = cut_off
    #print('the threshold')
    #print(optimal_proba_cutoff)
    roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in predictions[:, 1]]
    
    #print("Accuracy Score Before and After Thresholding: {}, {}".format(metrics.accuracy_score(y_true, predictions.argmax(axis=1)), metrics.accuracy_score(y_true, roc_predictions)))
    #print("Precision Score Before and After Thresholding: {}, {}".format(metrics.precision_score(y_true, predictions.argmax(axis=1)), metrics.precision_score(y_true, roc_predictions)))
    #print("Recall Score Before and After Thresholding: {}, {}".format(metrics.recall_score(y_true, predictions.argmax(axis=1)), metrics.recall_score(y_true, roc_predictions)))
    #print("F1 Score Before and After Thresholding: {}, {}".format(metrics.f1_score(y_true, predictions.argmax(axis=1)), metrics.f1_score(y_true, roc_predictions)))
    #print('\n')
    
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])

    #plt.plot(recalls,precisions)
    #plt.xlabel('recall')
    #plt.ylabel('precision')
    #plt.title('precision-recall curve')
    #plt.savefig("D:\StageNet\Jingyuan\prcurve_1") #the PR curve
    # plt.savefig('/home/Jingyuan/StageNet/prcurve_1')
    
    #plt.clf()

    (fpr, tpr, thresholds2) = metrics.roc_curve(y_true, predictions[:, 1])

    #plt.plot(fpr,tpr)
    #plt.xlabel('false positive rate')
    #plt.ylabel('true positive rate')
    #plt.title('ROC curve')
    #plt.savefig("D:\StageNet\Jingyuan\curve_1") # the ROC curve
    # plt.savefig('/home/Jingyuan/StageNet/roccurve_1')


    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    '''
    #before thresholding
    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))         # choose the class according to the largest probability (threshold 0.5 for binary classification)
    if verbose:
        print("confusion matrix before thresholding:")
        print(cf)
        print('\n')
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])
    '''
    #after thresholding
    cf = metrics.confusion_matrix(y_true, roc_predictions)                    # roc_predictions is set according to the pre-set threshold
    if verbose:
        print("confusion matrix after thresholding:")
        print(cf)
        print('\n')
    cf = cf.astype(np.float32)

    acc_t = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0_t = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1_t = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0_t = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1_t = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])
    
    
    if verbose:
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))
        print('\n')
        '''
        print("before thresholding:")
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print('\n')
        '''
        print("after thresholding:")
        print("accuracy = {}".format(acc_t))
        print("precision class 0 = {}".format(prec0_t))
        print("precision class 1 = {}".format(prec1_t))
        print("recall class 0 = {}".format(rec0_t))
        print("recall class 1 = {}".format(rec1_t))
        print('\n')

    return {"acc": acc_t,                                                     # return with the metrics after thresholding
            "prec0": prec0_t,
            "prec1": prec1_t,
            "rec0": rec0_t,
            "rec1": rec1_t,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse}, optimal_proba_cutoff


# for phenotyping

def print_metrics_multilabel(y_true, predictions, verbose=1):
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return {"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted}


# for length of stay

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100


def print_metrics_regression(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    predictions = np.maximum(predictions, 0).flatten()
    y_true = np.array(y_true)

    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Custom bins confusion matrix:")
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins,
                                      weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    if verbose:
        print("Mean absolute deviation (MAD) = {}".format(mad))
        print("Mean squared error (MSE) = {}".format(mse))
        print("Mean absolute percentage error (MAPE) = {}".format(mape))
        print("Cohen kappa score = {}".format(kappa))

    return {"mad": mad,
            "mse": mse,
            "mape": mape,
            "kappa": kappa}


class LogBins:
    nbins = 10
    means = [0.611848, 2.587614, 6.977417, 16.465430, 37.053745,
             81.816438, 182.303159, 393.334856, 810.964040, 1715.702848]


def get_bin_log(x, nbins, one_hot=False):
    binid = int(np.log(x + 1) / 8.0 * nbins)
    if binid < 0:
        binid = 0
    if binid >= nbins:
        binid = nbins - 1

    if one_hot:
        ret = np.zeros((LogBins.nbins,))
        ret[binid] = 1
        return ret
    return binid


def get_estimate_log(prediction, nbins):
    bin_id = np.argmax(prediction)
    return LogBins.means[bin_id]


def print_metrics_log_bins(y_true, predictions, verbose=1):
    y_true_bins = [get_bin_log(x, LogBins.nbins) for x in y_true]
    prediction_bins = [get_bin_log(x, LogBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("LogBins confusion matrix:")
        print(cf)
    return print_metrics_regression(y_true, predictions, verbose)


class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)
    means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
             131.579534, 155.643957, 179.660558, 254.306624, 585.325890]


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None


def get_estimate_custom(prediction, nbins):
    bin_id = np.argmax(prediction)
    assert 0 <= bin_id < nbins
    return CustomBins.means[bin_id]


def print_metrics_custom_bins(y_true, predictions, verbose=1):
    return print_metrics_regression(y_true, predictions, verbose)
