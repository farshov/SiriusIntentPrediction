from scipy.stats import wilcoxon
from quality_metrics.quality_metrics import get_accuracy_arr, get_f1_arr
import numpy as np


def rank_sum(our_acc, theirs_acc):
    our_acc = list(zip(our_acc, ["o"] * len(our_acc)))
    theirs_acc = list(zip(theirs_acc, ["t"] * len(theirs_acc)))
    together = our_acc + theirs_acc
    together = sorted(together)
    together = list(enumerate(together))
    sum_rank_our = 0
    sum_rank_theirs = 0
    for el in together:
        if el[1][1] == "o":
            sum_rank_our += el[0] + 1
        else:
            sum_rank_theirs += el[0] + 1
    return sum_rank_our, sum_rank_theirs


def measure_metric(our, theirs, metric, alpha):
    stats, pval = wilcoxon(our, theirs, alternative="two-sided")
    sum_rank_our, sum_rank_theirs = rank_sum(our, theirs)
    our_mean = np.mean(our)
    theirs_mean = np.mean(theirs)
    if pval < alpha:
        if our_mean < theirs_mean:
            print("{0} - we lose.".format(metric), end=" ")
        else:
            print("{0} - we win.".format(metric), end=" ")
    else:
        print("{0} doesn't differ.".format(metric), end=" ")
    print("P-value: {0: .3f}%. Stats: {1}. Our rank sum: {2}. Their rank sum: {3}"
          .format(pval * 100, str(stats), str(sum_rank_our), str(sum_rank_theirs)))
    print("Our: {0: .2f}".format(our_mean))
    print("Their: {0: .2f}".format(theirs_mean))


def perform_stat_tests(our_preds, theirs_preds, true_preds, alpha=0.05):
    our_acc = get_accuracy_arr(true_preds, our_preds)
    theirs_acc = get_accuracy_arr(true_preds, theirs_preds)
    our_precision, our_recall, our_f1 = get_f1_arr(true_preds, our_preds)
    theirs_precision, theirs_recall, theirs_f1 = get_f1_arr(true_preds, theirs_preds)

    measure_metric(our_acc, theirs_acc, "accuracy", alpha)
    measure_metric(our_precision, theirs_precision, "precision", alpha)
    measure_metric(our_recall, theirs_recall, "recall", alpha)
    measure_metric(our_f1, theirs_f1, "f1", alpha)

