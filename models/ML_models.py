from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm.SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def build_label_dict(rev, all_labels):

    """
    params:
    need reverted dict {(label, index)} or not
    labels for dict
    
    return: dict
    """

    label_dict = {}
    for labels in all_labels:
        for label in labels:
            if not label in label_dict:
                label_dict[label] = len(label_dict)
    if rev:
        label_dict = {v: k for (k, v) in label_dict.items()}

    return label_dict


def encode_labels(all_labels, label_dict):

    """
    params:
    string labels for encoding
    dict {(label, index)}
    
    return: indices
    """
    
    res = []
    for labels in all_labels:
        encode = labels.copy()
        i = 0
        for label in labels:
            encode[i] = label_dict[label] 
            i += 1
        res += [encode]
    return res


def preprocess_labels(labels, labels_dict):

    """
    params:
    list of list of string labels
    dict {(label, index)}

    return:
    vector where i-th element is 1 when i-th list contains element i
    """

    res = []
    encode = encode_labels(labels, labels_dict)
    for line in encode:
        vector = [0]*33
        for num in line:
            vector[num] = 1
            res.append(vector)
    return res


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

def get_result_of_base_model(model, train_X, train_labels, test_X, test_labels):
    
    """
    params:
    model with fit_predict methods
    training set
    training labels (list of 0-1 vectors)
    test set
    test lebals (list of lists of encoded labels)

    return:
    accuracy metric
    (precision, recall, f1) metrics
    """
    
    for tag in range(0, 33):
        train_y = [vector[tag] for vector in train_labels]
        test_y = [vector[tag] for vector in test_labels]
        model.fit(train_X, train_y)
        pred.append(model.predict(test_X))
    for i in range(len(test_labels))
        pred_labels.append([tag for tag in range(0, 33) if pred[tag][i] == 1])
    return get_accuracy(test_y, pred), get_f1(test_labels, pred)
    

def build_basic_models(prepared_train_matrix, prepared_test_matrix):

    """
    training set (from feature_extractor)
    training set (from feature_extractor)

    return: metrics on different models
    """
    
    labels_dict = build_label_dict(labels)
    labels = preprocess_labels(prepared_train_matrix["tags"], labels_dict)
    X = prepared_train_matrix.drop(columns=["tags"])
    test_labels = encode_labels(prepared_test_matrix["tags"], labels_dict)
    test_X = prepared_test_matrix.drop(columns=["tags"])
    res = []

    models = [KNeighborsClassifier(), GaussianNB(), SVC(), RandomForestClassifier(), AdaBoostClassifier()]
    for model in models:
        res.append(get_result_of_base_model(model, X, labels, text_X, test_labels))
    return res
