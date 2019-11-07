def get_accuracy(true_answers, pred_answers):

    """
    params:
    true encoded labels
    predicted encoded labels
    """

    acc = 0
    for (correct_list, pred_list) in zip(true_answers, pred_answers):
        acc += len(set(correct_list) & set(pred_answers)) / len(set(correct_list) | set(pred_answers))

    return acc / len(true_answers)


def get_f1(true_answers, pred_answers):
    """
    params:
    true encoded labels
    predicted encoded labels
    """

    cor_preds = 0
    all_cors = 0
    all_preds = 0

    for (correct_list, pred_list) in zip(true_answers, pred_answers):
        cor_preds += len(set(correct_list) & set(pred_list))
        all_cors += len(correct_list)
        all_preds += len(pred_list)

    precision = cor_preds / all_preds
    recall = cor_preds / all_cors
    f1 = 0
    if cor_preds:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1