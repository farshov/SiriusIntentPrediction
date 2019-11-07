from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import quality_metrics.quality_metrics as qm


class MLModels:

    @staticmethod
    def fit_models(x_train, y_train):
        """
        training set (from feature_extractor)
        training set (from feature_extractor) list of lists of label codes

        return: metrics on different models
        """
        models = [KNeighborsClassifier(), GaussianNB(), SVC(), RandomForestClassifier(), AdaBoostClassifier()]
        for model in models:
            model.fit(x_train, y_train)
        return models

    @staticmethod
    def predict_model(model, x_test, y_test):
        """

        :param model:
        :param x_test:
        :param y_test:
        :return:
        """
        y_pred = model.predict(x_test)
        acc = qm.get_accuracy(y_test, y_pred)
        f1 = qm.get_f1(y_test, y_pred)
        return y_pred, acc, f1

    @staticmethod
    def test_basic_model_multilabel(self, x_train, y_train, x_test, y_test):
        models = self.build_basic_models(x_train, y_train[:,0])
        models_x_test = []
        models_acc = []
        models_f1 = []
        for i in range(len(models)):
            models_x_test.append(x_test)

        for label_idx in y_test.shape[1]:
            for i, model in enumerate(models):
                y_pred = model.predict(models_x_test[i], y_test[:, label_idx])
                models_x_test[i]["label_"+str(label_idx)] = y_pred
            if label_idx + 1 < y_test.shape[1]:
                x_train["label_"+str(label_idx + 1)] = y_train[:, label_idx]
                models = self.build_basic_models(x_train, y_train[:, label_idx + 1])

        for model_pred in models_x_test:
            predictions = model_pred[:, len(model_pred.columns)-len(y_test.columns)]
            acc = qm.get_accuracy(y_test, predictions)
            f1 = qm.get_f1(y_test, predictions)
            models_acc.append(acc)
            models_f1.append(f1)
        return models_acc, models_f1
