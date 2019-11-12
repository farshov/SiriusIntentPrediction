import numpy as np
import math


def get_accuracy_arr(true_answers, pred_answers):
    """
    params:
    true 0-1 vectors
    predicted 0-1 vectors
    """

    acc = []
    for (correct_vector, pred_vector) in zip(true_answers, pred_answers):
        correct_list = [i for i in range(len(correct_vector)) if correct_vector[i] == 1]
        pred_list = [i for i in range(len(pred_vector)) if pred_vector[i] == 1]
        acc_ = len(set(correct_list) & set(pred_list)) / len(set(correct_list) | set(pred_list))
        acc.append(acc_)
    acc = np.array(list(map(lambda x: 0 if math.isnan(x) else x, acc)))
    return acc


def get_accuracy(true_answers, pred_answers):

    """
    params:
    true 0-1 vectors
    predicted 0-1 vectors
    """
    
    acc = 0
    for (correct_vector, pred_vector) in zip(true_answers, pred_answers):
        correct_list = [i for i in range(len(correct_vector)) if correct_vector[i] == 1]
        pred_list = [i for i in range(len(pred_vector)) if pred_vector[i] == 1]
        acc += len(set(correct_list) & set(pred_list)) / len(set(correct_list) | set(pred_list))

    return acc / len(true_answers)


def get_f1(true_answers, pred_answers):
    """
    params:
    true 0-1 vectors
    predicted 0-1 vectors
    """

    cor_preds = 0
    all_cors = 0
    all_preds = 0

    for (correct_vector, pred_vector) in zip(true_answers, pred_answers):
        correct_list = [i for i in range(len(correct_vector)) if correct_vector[i] == 1]
        pred_list = [i for i in range(len(pred_vector)) if pred_vector[i] == 1]
        cor_preds += len(set(correct_list) & set(pred_list))
        all_cors += len(correct_list)
        all_preds += len(pred_list)

    precision = cor_preds / all_preds
    recall = cor_preds / all_cors
    f1 = 0
    if cor_preds:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def get_f1_arr(true_answers, pred_answers):
    cor_preds = []
    all_cors = []
    all_preds = []

    for (correct_vector, pred_vector) in zip(true_answers, pred_answers):
        correct_list = [i for i in range(len(correct_vector)) if correct_vector[i] == 1]
        pred_list = [i for i in range(len(pred_vector)) if pred_vector[i] == 1]
        cor_preds.append(len(set(correct_list) & set(pred_list)))
        all_cors.append(len(correct_list))
        all_preds.append(len(pred_list))

    precision = np.array(cor_preds) / np.array(all_preds)
    recall = np.array(cor_preds) / np.array(all_cors)
    f1 = []
    if cor_preds:
        f1 = 2 * precision * recall / (precision + recall)
    precision = np.array(list(map(lambda x: 0 if math.isnan(x) else x, precision)))
    recall = np.array(list(map(lambda x: 0 if math.isnan(x) else x, recall)))
    f1 = np.array(list(map(lambda x: 0 if math.isnan(x) else x, f1)))
    return precision, recall, f1
