import numpy as np

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
