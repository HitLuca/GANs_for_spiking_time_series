from keras import Input, Model
from keras.callbacks import *
from keras.layers import Dense

from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics

import sys
sys.path.append("..")
from generative_models import utils
import metrics_utils

class Evaluator:
    def __init__(self, models_list, real_data_filepath, split, elements, timesteps, regression_targets):
        self._generated_data = None
        self._histories_regression = {}
        self._histories_classification = {}
        self._models_list = models_list
        self._real_data = utils.load_splitted_dataset(real_data_filepath, timesteps)[:elements]
        self._split = split
        self._elements = elements
        self._timesteps = timesteps
        self._regression_targets = regression_targets
        self._base_folder = ''

    def _build_nn_classifier(self):
        model_inputs = Input((self._timesteps,))
        classified = model_inputs

        classified = Dense(45, activation='relu')(classified)
        classified = Dense(15, activation='relu')(classified)
        classified = Dense(1, activation='sigmoid')(classified)

        classifier = Model(model_inputs, classified, 'classifier')
        classifier.compile(loss='binary_crossentropy', optimizer='adam')
        return classifier

    def _build_nn_regressor(self):
        model_inputs = Input((self._timesteps - self._regression_targets,))
        regressed = model_inputs

        regressed = Dense(15, activation='relu')(regressed)
        regressed = Dense(5, activation='relu')(regressed)
        regressed = Dense(self._regression_targets, activation='tanh')(regressed)

        regressor = Model(model_inputs, regressed, 'regressor')
        regressor.compile(loss='mse', optimizer='adam')
        return regressor

    def _build_classifiers(self):
        if 'nn' in self._models_list:
            assert self._models_list[0] == 'nn'

        classifiers = []
        for classifier in self._models_list:
            if classifier == 'nn':
                classifiers.append(self._build_nn_classifier())
            if classifier == 'svm':
                classifiers.append(SVC())
            if classifier == 'rf':
                classifiers.append(RandomForestClassifier())
            if classifier == 'dt':
                classifiers.append(DecisionTreeClassifier())
        return classifiers

    def _build_regressors(self):
        if 'nn' in self._models_list:
            assert self._models_list[0] == 'nn'

        regressors = []
        for regressor in self._models_list:
            if regressor == 'nn':
                regressors.append(self._build_nn_regressor())
            if regressor == 'svm':
                regressors.append(MultiOutputRegressor(SVR()))
            if regressor == 'rf':
                regressors.append(RandomForestRegressor())
            if regressor == 'dt':
                regressors.append(DecisionTreeRegressor())
        return regressors

    def _postprocess_dataset(self):
        step = 0.1
        self._generated_data = np.around(self._generated_data, 4)
        zero_value = metrics_utils.get_mode(self._real_data)
        self._generated_data[np.logical_and(self._generated_data < (zero_value + step),
                                            self._generated_data > (zero_value - step))] = zero_value

    def _evaluate_data_classification(self):
        self._postprocess_dataset()
        (x_train, y_train), (x_test, y_test) = metrics_utils.combine_data(self._real_data, self._generated_data,
                                                                          self._split)

        classifiers = self._build_classifiers()

        if 'nn' in self._models_list:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3)
            classifiers[0].fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=0,
                               callbacks=[early_stopping])

        for i, classifier in enumerate(classifiers[1:]):
            classifier.fit(x_train, y_train)

        histories = {}
        for i, classifier in enumerate(classifiers):
            classifier_name = self._models_list[i]
            y_pred = classifier.predict(x_test)

            f1_score = metrics.f1_score(y_test, np.rint(y_pred))
            accuracy = metrics.accuracy_score(y_test, np.rint(y_pred))
            histories[classifier_name] = {
                'f1_score': f1_score,
                'accuracy': accuracy
            }

        return histories

    def _evaluate_data_regression(self):
        self._postprocess_dataset()
        (x_train, y_train), (x_test, y_test) = metrics_utils.combine_data_regression(self._real_data,
                                                                                     self._generated_data, self._split,
                                                                                     self._regression_targets)

        regressors = self._build_regressors()

        if 'nn' in self._models_list:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3)
            regressors[0].fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=0,
                              callbacks=[early_stopping])

        for i, regressor in enumerate(regressors[1:]):
            regressor.fit(x_train, y_train)

        histories = {}
        for i, regressor in enumerate(regressors):
            regressor_name = self._models_list[i]
            y_pred = regressor.predict(x_test)

            mse = metrics.mean_squared_error(y_test, y_pred)
            r2 = metrics.r2_score(y_test, y_pred)
            histories[regressor_name] = {
                'mse': mse,
                'r2': r2
            }

        return histories

    def run_comparison(self, generated_data_filepaths, labels):
        for index, filepath in enumerate(generated_data_filepaths):
            print(labels[index])
            self._generated_data = np.load(filepath)[:self._elements]
            history_classification = self._evaluate_data_classification()
            history_regression = self._evaluate_data_regression()
            self._histories_classification[labels[index]] = history_classification
            self._histories_regression[labels[index]] = history_regression

    def save_histories(self, filename_tag):
        metrics_utils.save_to_json(self._base_folder + '/' + 'classification_scores_' + filename_tag + '.json', self._histories_classification)
        metrics_utils.save_to_json(self._base_folder + '/' + 'regression_scores_' + filename_tag + '.json', self._histories_regression)

    def save_metrics_plot(self, labels, title):
        metrics_utils.plot_metrics(self._histories_classification, labels, title, True, self._base_folder + '/' + title + '_classification')
        metrics_utils.plot_metrics(self._histories_regression, labels, title, True, self._base_folder + '/' + title + '_regression')

    def set_base_folder(self, base_folder):
        self._base_folder = base_folder

    def set_models_list(self, models_list):
        self._models_list = models_list
