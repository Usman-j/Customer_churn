'''
Script for testing all the functions of churn_library and corresponding logging.
'''

import os
import logging
import numpy as np
import churn_library as cls
from constants import (DATA_path, MODELS_path, EDA_path, RESULTS_path,
                       categorical_feats_lst, final_feats_lst)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - Checks if the data is loaded propery
    '''
    try:
        df = import_data(DATA_path)
        logging.info("Testing import_data: SUCCESS")
        return df
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, df):
    '''
    test perform eda function - Checks if the eda images have been saved in the respective folder
    '''
    try:
        perform_eda(df)
        assert os.path.isfile(EDA_path + 'Churn_histogram.png')
        assert os.path.isfile(EDA_path + 'Customer_Age_histogram.png')
        assert os.path.isfile(EDA_path + 'Marital_status_bar_plot.png')
        assert os.path.isfile(EDA_path + 'Total_Trans_Ct_dist_plot.png')
        assert os.path.isfile(EDA_path + 'Correlation_heatmap.png')
        logging.info('Testing perform_eda: SUCCESS: All images were saved')
    except AssertionError as err:
        logging.error('Testing perform_eda: ERROR: Image(s) were not found')
        raise err


def test_encoder_helper(encoder_helper, df, cat_lst):
    '''
    test encoder helper - Checks if the categorical features have been encoded with
    respect to response proportion and added as new columns
    '''
    try:
        df_encoded = encoder_helper(df, cat_lst)
        logging.info(
            'Testing encoder_helper: SUCCESS: All categorical features were processed')
    except KeyError as err:
        logging.error(
            f'Testing encoder_helper: ERROR: Column {err} not found in dataframe for\
            encoding')
        raise KeyError from err

    try:
        for col in cat_lst:
            assert df_encoded[col + '_Churn'].dtype == np.float64
        logging.info(
            'Testing encoder_helper: SUCCESS: All categorical have been encoded')
        return df_encoded
    except AssertionError as err:
        logging.error(
            'Testing encoder_helper: ERROR: Encoded features are not numeric')
        raise err


def test_perform_feature_engineering(
        perform_feature_engineering, df, feat_lst):
    '''
    test perform_feature_engineering - Checks if data has been split into train and test
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, feat_lst)
        assert len(X_train) + len(X_test) == len(df)
        logging.info(
            "Testing perform_feature_engineering: SUCCESS: Data split into train and test")
        return X_train, X_test, y_train, y_test
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: ERROR: Data splitting not accurate")
        raise err


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models - Checks if trained models, classification report, ROC curves and
    Feature importance plot have been saved in the respective folders
    '''
    try:
        train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile(MODELS_path + 'rfc_model.pkl')
        assert os.path.isfile(MODELS_path + 'logistic_model.pkl')
        assert os.path.isfile(RESULTS_path +
                              'Random_Forest_Classification_report.png')
        assert os.path.isfile(RESULTS_path +
                              'Logistic_Regression_Classification_report.png')
        assert os.path.isfile(RESULTS_path + 'ROC_curves.png')
        assert os.path.isfile(
            RESULTS_path +
            'RandomForest_FeatureImportance.png')
        logging.info(
            "Testing train_models: SUCCESS: All files related to model training have been saved")
    except AssertionError as err:
        logging.error(
            "Testing train_models: ERROR: Some file(s) for model training not found \
            in the respective folder")
        raise err


if __name__ == "__main__":
    df_data = test_import(cls.import_data)
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    test_eda(cls.perform_eda, df_data)
    df_Encoded = test_encoder_helper(
        cls.encoder_helper, df_data, categorical_feats_lst)
    X_Train, X_Test, y_Train, y_Test = test_perform_feature_engineering(
        cls.perform_feature_engineering, df_Encoded, final_feats_lst)
    test_train_models(cls.train_models, X_Train, X_Test, y_Train, y_Test)
