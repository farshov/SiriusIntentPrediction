from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import quality_metrics.quality_metrics as qm
import numpy as np


class MLModels:

    @staticmethod
    def fit_models(x_train, y_train):
        """
        training set (from feature_extractor)
        training set (from feature_extractor) list of lists of label codes

        return: metrics on different models
        """
        models = [KNeighborsClassifier(), GaussianNB(), SVC(gamma="scale"),
                  RandomForestClassifier(n_estimators=100), AdaBoostClassifier()]
        for model in models:
            model.fit(x_train, y_train)
        return models

    def test_basic_model_multilabel(self, x_train, y_train, x_test, y_test):
        models_acc = []
        models_f1 = []
        models_predictions = self.get_basic_model_predictions(x_train, y_train, x_test)
        for predictions in models_predictions:
            acc = qm.get_accuracy(y_test, predictions)
            f1 = qm.get_f1(y_test, predictions)
            models_acc.append(acc)
            models_f1.append(f1)
        return models_acc, models_f1

    def get_basic_model_predictions(self, x_train, y_train, x_test):
        models = self.fit_models(x_train, np.array(y_train)[:,0])
        predictions = []
        x_test_models = []
        for i in range(len(models)):
            x_test_models.append(x_test.copy())
        for label_idx in range(len(y_train[0])):
            for i, model in enumerate(models):
                y_pred = model.predict(x_test_models[i])
                x_test_models[i].loc[:, "label_"+str(label_idx + 1)] = y_pred
            x_train.loc[:, "label_"+str(label_idx + 1)] = np.array(y_train)[:, label_idx]
            if label_idx + 1 < len(y_train[0]):
                models = self.fit_models(x_train, np.array(y_train)[:, label_idx + 1])

        for i, model in enumerate(models):
            pred = np.array(x_test_models[i].loc[:, "label_1":])
            predictions.append(pred)
        return predictions

    def best_params(self, train_x, test_x, y_train, y_test):
        k_neighbours_range = [1,2,3,4,5]
        best_knn_acc = 0
        best_knn_param = 0
        for n_n in k_neighbours_range:
            x_test = test_x.copy()
            x_train = train_x.copy()
            model = KNeighborsClassifier(n_neighbors=n_n)
            model.fit(x_train, np.array(y_train)[:,0])
            for label_idx in range(len(y_test[0])):
                y_pred = model.predict(x_test)
                x_test.loc[:, "label_" + str(label_idx + 1)] = y_pred
                x_train.loc[:, "label_" + str(label_idx + 1)] = np.array(y_train)[:, label_idx]
                if label_idx + 1 < len(y_test[0]):
                    model = KNeighborsClassifier(n_neighbors=n_n)
                    model.fit(x_train, np.array(y_train)[:, label_idx + 1])

            predictions = np.array(x_test.loc[:, "label_1":])
            acc = qm.get_accuracy(y_test, predictions)
            print("KNN acc: {}".format(acc))
            print("num_neighbours: {}".format(n_n))
            if acc > best_knn_acc:
                best_knn_acc = acc
                best_knn_param = n_n

        print("KNN best acc: {}".format(best_knn_acc))
        print("num_neighbours: {}".format(best_knn_param))

        c_range = [2000, 3000]
        best_svc_acc = 0
        best_svc_param = 0
        for n_n in c_range:
            x_test = test_x.copy()
            x_train = train_x.copy()
            model = SVC(C=n_n, gamma="scale")
            model.fit(x_train, np.array(y_train)[:,0])
            for label_idx in range(len(y_test[0])):
                y_pred = model.predict(x_test)
                x_test.loc[:, "label_" + str(label_idx + 1)] = y_pred
                x_train.loc[:, "label_" + str(label_idx + 1)] = np.array(y_train)[:, label_idx]
                if label_idx + 1 < len(y_test[0]):
                    model = SVC(C=n_n, gamma="scale")
                    model.fit(x_train, np.array(y_train)[:, label_idx + 1])

            predictions = np.array(x_test.loc[:, "label_1":])
            acc = qm.get_accuracy(y_test, predictions)
            print("SVC acc: {}".format(acc))
            print("c: {}".format(n_n))
            if acc > best_svc_acc:
                best_svc_acc = acc
                best_svc_param = n_n

        print("SVC best acc: {}".format(best_svc_acc))
        print("c: {}".format(best_svc_param))

        num_est_range = [100, 70, 50, 30]
        max_depth_range = [15, 25, 50]
        best_rf_acc = 0
        best_num_est = 0
        best_max_d = 0
        for n_n in num_est_range:
            for max_d in max_depth_range:
                x_test = test_x.copy()
                x_train = train_x.copy()
                model = RandomForestClassifier(n_estimators=n_n, max_depth=max_d)
                model.fit(x_train, np.array(y_train)[:,0])
                for label_idx in range(len(y_test[0])):
                    y_pred = model.predict(x_test)
                    x_test.loc[:, "label_" + str(label_idx + 1)] = y_pred
                    x_train.loc[:, "label_" + str(label_idx + 1)] = np.array(y_train)[:, label_idx]
                    if label_idx + 1 < len(y_test[0]):
                        model = RandomForestClassifier(n_estimators=n_n, max_depth=max_d)
                        model.fit(x_train, np.array(y_train)[:, label_idx + 1])

                predictions = np.array(x_test.loc[:, "label_1":])
                acc = qm.get_accuracy(y_test, predictions)
                print("Forest acc: {}".format(acc))
                print("num_estimators: {}".format(n_n))
                print("max_depth: {}".format(max_d))
                if acc > best_rf_acc:
                    best_rf_acc = acc
                    best_num_est = n_n
                    best_max_d = max_d

        print("Random Forest best acc: {}".format(best_rf_acc))
        print("num_estimators: {}".format(best_num_est))
        print("max_depth: {}".format(best_max_d))


        num_est_range = [100, 200, 300, 500, 50]
        best_adab_acc = 0
        best_adab_param = 0
        for n_n in num_est_range:
            x_test = test_x.copy()
            x_train = train_x.copy()
            model = AdaBoostClassifier(n_estimators=n_n)
            model.fit(x_train, np.array(y_train)[:,0])
            for label_idx in range(len(y_test[0])):
                y_pred = model.predict(x_test)
                x_test.loc[:, "label_" + str(label_idx + 1)] = y_pred
                x_train.loc[:, "label_" + str(label_idx + 1)] = np.array(y_train)[:, label_idx]
                if label_idx + 1 < len(y_test[0]):
                    model = AdaBoostClassifier(n_estimators=n_n)
                    model.fit(x_train, np.array(y_train)[:, label_idx + 1])

            predictions = np.array(x_test.loc[:, "label_1":])
            acc = qm.get_accuracy(y_test, predictions)
            print("AdaBoost acc: {}".format(acc))
            print("num_estimators: {}".format(n_n))
            if acc > best_adab_acc:
                best_adab_acc = acc
                best_adab_param = n_n

        print("KNN best acc: {}".format(best_knn_acc))
        print("num_neighbours: {}".format(best_knn_param))

        print("SVC best acc: {}".format(best_svc_acc))
        print("c: {}".format(best_svc_param))

        print("Random Forest best acc: {}".format(best_rf_acc))
        print("num_estimators: {}".format(best_num_est))
        print("max_depth: {}".format(best_max_d))

        print("Adaboost best acc: {}".format(best_adab_acc))
        print("num_estimators: {}".format(best_adab_param))
