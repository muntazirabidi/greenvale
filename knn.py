# -*- coding: utf-8 -*-
"""This module contains the methods necessary to classify the bandsize
of a group of potatoes given their weight and vice versa
"""

import copy
from itertools import groupby
import constants
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def load_batch_data(variety_name):
    '''Loads data from variety name csv file'''
    with open("./training_data/{}_TuberData.csv".format(variety_name), 'r') as csvfile:
        batch_data = pd.read_csv(csvfile)
    return pd.DataFrame(batch_data)

def get_training_data(variety):
    '''Retrieves all data from csv for a given variety'''
    dataframe = load_batch_data(variety)
    train_data = dataframe.fillna(0)

    bands_train_data = train_data.iloc[:, 1].values
    weight_train_data = train_data.iloc[:, 5].values*constants.SCALING_INTEGER
    weight_train_data_scaled = weight_train_data.astype(int)

    return bands_train_data, weight_train_data_scaled

def get_individual_data(data):
    '''Given a spreadsheet of data, returns an object with an
    array of tuples of average weight and counts for each band

    Returns object;
    {
        32.5: [(1, 0.75), (2, 0.73), ...],
        37.5: ....,
        ....
    }
    '''
    midgrade_to_weight = {}
    for index, row in data.iterrows():
        if index != 0:
            midgrade, counts = row[1:3]
            weight = row[5]/constants.SCALING_INTEGER
            if midgrade in midgrade_to_weight:
                midgrade_to_weight[midgrade].append((counts, weight))
            else:
                midgrade_to_weight[midgrade] = [(counts, weight)]
    return midgrade_to_weight

def get_total_counts_and_weights(data):
    '''Given a varieties spreadsheet data, returns counts and
    average weight for each band
    '''
    potato_counts = {}
    potato_weight = {}
    midgrade_to_weight = get_individual_data(data)
    for band, values in midgrade_to_weight.items():
        total_counts = 0
        mean_weight = 0
        number_of_items = len(values)
        for (counts, weight) in values:
            total_counts += counts
            mean_weight += weight/number_of_items

        potato_counts[band] = total_counts
        potato_weight[band] = mean_weight
    return potato_counts, potato_weight

class KNNClassifier(object):
    '''KNN classifier for both the size given weight data and vice versa
    Initialised with the variety and sample data to be applied.

    Note: if weight sample data is given only the bandsize classifier
    will be available and vice versa. Create a new instance of the classifier
    to use the other method with alternative data.
    '''
    def __init__(self, test_data):
        self.variety = test_data[0]['variety']
        self.k_value = constants.OPTIMIZED_K[self.variety]
        self.raw_test_data = test_data
        self.test_data = pd.DataFrame(test_data)
        self.midsizebands_train_data, self.tuberweight_train_data = get_training_data(self.variety)

    # def weight_classifier(self):
    #     '''From a list of given bandsizes will return a calculated weight
    #     Requires test_data to have a column named "bandsize"
    #     '''
    #     # Training and predicting for tuber weight
    #     classifier_weight = KNeighborsClassifier(n_neighbors=self.k_value)
    #     classifier_weight.fit(
    #         self.midsizebands_train_data.reshape(-1, 1),
    #         self.tuberweight_train_data
    #     )

    #     weight_prediction = classifier_weight.predict(
    #         self.test_data["bandsize"].as_matrix().reshape(-1, 1)
    #     )

    def bandsize_classifier(self):
        '''From a list of given bandsizes will return a calculated weight
        Requires test_data to have a column named "tuber_weight"
        '''
        # Training and predicting for tuber band size, inverse of above
        self.test_data.tuber_weight = (self.test_data.tuber_weight*1000).astype(int)
        tuber_weight_test_data = self.test_data["tuber_weight"].values.reshape(-1, 1)
        classifier_bandsize = KNeighborsClassifier(n_neighbors=5)
        classifier_bandsize.fit(
            self.tuberweight_train_data.reshape(-1, 1),
            self.midsizebands_train_data*10
        )

        size_prediction = classifier_bandsize.predict(
            tuber_weight_test_data
        )
        # Remove scaling factor applied above
        size_prediction *= 0.1

        # Add size_band to each relevant item
        # raw_test_data and size_predictions can NOT be
        # sorted prior to this combination
        for i, _ in enumerate(size_prediction):
            self.raw_test_data[i]['size_band'] = size_prediction[i]

        # Avoid mutating of object from affecting ungrouped results
        ungrouped_results = copy.deepcopy(self.raw_test_data)

        # Group by sample_id
        key = lambda x: x['sample_id']
        self.raw_test_data.sort(key=key)

        grouped_results = []
        for k, values in groupby(self.raw_test_data, key=key):
            # Remove sample_id from tuber_details
            items = list(values)
            for item in items:
                item.pop('sample_id', None)

            grouped_results.append({
                'sample_id': k,
                'tuber_details': items
            })
        return ungrouped_results, grouped_results
