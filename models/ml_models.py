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

    def test_basic_model_multilabel(self, x_train, y_train, x_test, y_test):
        models = self.fit_models(x_train, np.array(y_train)[:,0])
        models_acc = []
        models_f1 = []
        preds = np.zeros(shape=(len(models), len(y_test[0]), len(y_test)))

        for label_idx in range(len(y_test[0])):
            for i, model in enumerate(models):
                y_pred = model.predict(x_test)
                preds[i][label_idx] = y_pred
            if label_idx + 1 < len(y_test[0]):
                x_train["label_"+str(label_idx + 1)] = np.array(y_train)[:, label_idx]
                x_test["label_"+str(label_idx + 1)] = y_pred
                models = self.fit_models(x_train, np.array(y_train)[:, label_idx + 1])

        for i, model in enumerate(models):
            predictions = [preds[i][:,j] for j in range(len(preds[i][0]))]
            acc = qm.get_accuracy(y_test, predictions)
            f1 = qm.get_f1(y_test, predictions)
            models_acc.append(acc)
            models_f1.append(f1)
        return models_acc, models_f1
