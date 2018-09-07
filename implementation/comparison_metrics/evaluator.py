import sys

from keras import Input, Model
from keras.callbacks import *
from keras.layers import Dense, Conv1D, LeakyReLU, MaxPooling1D, Flatten, Lambda
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.append("..")
from generative_models import utils
import metrics_utils


class Evaluator:
    def __init__(self, models_list, real_data_filepath, split, elements, timesteps, regression_targets,
                 flattening_range, iteration):
        self._generated_data = None
        self._histories_regression = {}
        self._histories_classification = {}
        self._models_list = models_list
        _, self._real_data, _ = utils.load_splitted_dataset(split=split, timesteps=timesteps,
                                                            dataset_filepath=real_data_filepath)
        self._real_data = self._real_data[:elements]
        self._split = split
        self._elements = elements
        self._timesteps = timesteps
        self._regression_targets = regression_targets
        self._base_folder = ''
        self._flattening_range = flattening_range
        self._iteration = iteration

    def _build_nn_classifier(self):
        model_inputs = Input((self._timesteps,))
        classified = Lambda(lambda x: K.expand_dims(x, -1))(model_inputs)

        classified = Conv1D(32, 3, padding='same')(classified)
        classified = LeakyReLU(0.2)(classified)
        classified = MaxPooling1D(2, padding='same')(classified)

        classified = Conv1D(32, 3, padding='same')(classified)
        classified = LeakyReLU(0.2)(classified)
        classified = MaxPooling1D(2, padding='same')(classified)

        classified = Conv1D(32, 3, padding='same')(classified)
        classified = LeakyReLU(0.2)(classified)
        classified = MaxPooling1D(2, padding='same')(classified)

        classified = Conv1D(32, 3, padding='same')(classified)
        classified = LeakyReLU(0.2)(classified)
        classified = Flatten()(classified)

        classified = Dense(1, activation='sigmoid')(classified)

        classifier = Model(model_inputs, classified, 'classifier')
        classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return classifier

    def _build_nn_regressor(self):
        model_inputs = Input((self._timesteps - self._regression_targets,))
        regressed = Lambda(lambda x: K.expand_dims(x, -1))(model_inputs)

        regressed = Conv1D(32, 3, padding='same')(regressed)
        regressed = LeakyReLU(0.2)(regressed)
        regressed = MaxPooling1D(2, padding='same')(regressed)

        regressed = Conv1D(32, 3, padding='same')(regressed)
        regressed = LeakyReLU(0.2)(regressed)
        regressed = MaxPooling1D(2, padding='same')(regressed)

        regressed = Conv1D(32, 3, padding='same')(regressed)
        regressed = LeakyReLU(0.2)(regressed)
        regressed = MaxPooling1D(2, padding='same')(regressed)

        regressed = Conv1D(32, 3, padding='same')(regressed)
        regressed = LeakyReLU(0.2)(regressed)

        regressed = Flatten()(regressed)
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
        self._generated_data = np.around(self._generated_data, 4)
        zero_value = metrics_utils.get_mode(self._real_data)
        self._generated_data[np.logical_and(self._generated_data < (zero_value + self._flattening_range),
                                            self._generated_data > (zero_value - self._flattening_range))] = zero_value

    def _evaluate_data_classification(self, postprocess):
        if postprocess:
            self._postprocess_dataset()

        (x_train, y_train), (x_test, y_test) = metrics_utils.combine_data(self._real_data, self._generated_data,
                                                                          self._split)

        classifiers = self._build_classifiers()

        if 'nn' in self._models_list:
            print('nn')
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3)
            classifiers[0].fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=0,
                               callbacks=[early_stopping])

        for i, classifier in enumerate(classifiers[1:]):
            print(self._models_list[i + 1])
            classifier.fit(x_train, y_train)

        histories = {}
        for i, classifier in enumerate(classifiers):
            classifier_name = self._models_list[i]
            y_pred = classifier.predict(x_test)

            accuracy = metrics.accuracy_score(y_test, np.rint(y_pred))
            f1_score = metrics.f1_score(y_test, np.rint(y_pred))

            print(classifier_name)
            print('accuracy:', accuracy)
            print('f1_score:', f1_score)
            histories[classifier_name] = {
                'accuracy': accuracy,
                'f1_score': f1_score
            }

        return histories

    def _evaluate_data_regression(self, postprocess):
        if postprocess:
            self._postprocess_dataset()

        (x_train, y_train), (x_test, y_test) = metrics_utils.combine_data_regression(self._real_data,
                                                                                     self._generated_data, self._split,
                                                                                     self._regression_targets)

        regressors = self._build_regressors()

        if 'nn' in self._models_list:
            print('nn')
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3)
            regressors[0].fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=0,
                              callbacks=[early_stopping])

        for i, regressor in enumerate(regressors[1:]):
            print(self._models_list[i + 1])
            regressor.fit(x_train, y_train)

        histories = {}
        for i, regressor in enumerate(regressors):
            regressor_name = self._models_list[i]
            y_pred = regressor.predict(x_test)

            mse = metrics.mean_squared_error(y_test, y_pred)
            r2 = metrics.r2_score(y_test, y_pred)
            print(regressor_name)
            print('mse:', mse)
            print('r2:', r2)

            histories[regressor_name] = {
                'mse': mse,
                'r2': r2
            }

        return histories

    def run_comparison_classification(self, generated_data_filepaths, labels, title):
        for index, filepath in enumerate(generated_data_filepaths):
            postprocess = True
            if labels[index] == 'handcrafted':
                postprocess = False

            print(labels[index])
            self._generated_data = np.load(filepath)[:self._elements]

            print('classification')
            history_classification = self._evaluate_data_classification(postprocess)
            self._histories_classification[labels[index]] = history_classification

        metrics_utils.save_to_json(self._base_folder + '/' + str(self._iteration) + '_classification_scores_' + title + '_' + str(self._flattening_range) + '.json',
                                   self._histories_classification)

        metrics_utils.plot_metrics(self._histories_classification, labels, title, True,
                                   self._base_folder + '/' + str(self._iteration) + '_' + title + '_classification' + '_' + str(self._flattening_range))

    def run_comparison_regression(self, generated_data_filepaths, labels, title):
        for index, filepath in enumerate(generated_data_filepaths):
            postprocess = True
            if labels[index] == 'handcrafted':
                postprocess = False

            print(labels[index])

            self._generated_data = np.load(filepath)[:self._elements]

            print('regression')
            history_regression = self._evaluate_data_regression(postprocess)
            self._histories_regression[labels[index]] = history_regression

        metrics_utils.save_to_json(self._base_folder + '/' + str(self._iteration) + '_regression_scores_' + title + '_' + str(self._flattening_range) + '.json',
                                   self._histories_regression)

        metrics_utils.plot_metrics(self._histories_regression, labels, title, True,
                                   self._base_folder + '/' + str(self._iteration) + '_' + title + '_regression' + '_' + str(self._flattening_range))

    def set_base_folder(self, base_folder):
        self._base_folder = base_folder

    def set_models_list(self, models_list):
        self._models_list = models_list
