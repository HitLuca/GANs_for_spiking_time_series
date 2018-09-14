import sys

from keras import Input, Model
from keras.callbacks import *
from keras.layers import Dense, Conv1D, LeakyReLU, MaxPooling1D, Flatten, Lambda
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append("..")
from generative_models import utils
import metrics_utils


class EvaluationFramework:
    def __init__(self, models_list, real_data_filepath, split, elements, timesteps,
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

    def _build_classifiers(self):
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

        for model_name, classifier in zip(self._models_list, classifiers):
            print(model_name)
            if model_name == 'nn':
                split = int((x_train.shape[0]) * 0.2)

                x_val_nn = x_train[:split]
                x_train_nn = x_train[split:]

                y_val_nn = y_train[:split]
                y_train_nn = y_train[split:]

                early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3)
                classifier.fit(x_train_nn, y_train_nn, validation_data=(x_val_nn, y_val_nn), epochs=100, verbose=0,
                               callbacks=[early_stopping])
            else:
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

        metrics_utils.save_to_json(
            self._base_folder + '/' + str(self._iteration) + '_classification_scores_' + title + '_' + str(
                self._flattening_range) + '.json',
            self._histories_classification)

        metrics_utils.plot_metrics(self._histories_classification, labels, title, True,
                                   self._base_folder + '/' + str(
                                       self._iteration) + '_' + title + '_classification' + '_' + str(
                                       self._flattening_range))

    def set_base_folder(self, base_folder):
        self._base_folder = base_folder

    def set_models_list(self, models_list):
        self._models_list = models_list
