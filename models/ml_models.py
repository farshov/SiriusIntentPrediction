from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import quality_metrics.quality_metrics as qm


class MLModels:
    @staticmethod
    def build_basic_models(x_train, y_train):
        """
        training set (from feature_extractor)
        training set (from feature_extractor)

        return: metrics on different models
        """
        models = [KNeighborsClassifier(), GaussianNB(), SVC(), RandomForestClassifier(), AdaBoostClassifier()]
        for model in models:
            model.fit(x_train, y_train)
        return models

    @staticmethod
    def evaluate_basic_models(models,  x_test, y_test):
        """

        :param models:
        :param x_test:
        :param y_test:
        :return:
        """
        models_acc = []
        models_f1 = []
        for model in models:
            y_pred = model.predict(x_test)
            acc = qm.get_accuracy(y_test, y_pred)
            f1 = qm.get_f1(y_test, y_pred)
            models_acc.append(acc)
            models_f1.append(f1)
        return models_acc, models_f1
