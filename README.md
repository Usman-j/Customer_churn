# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
- This is a library for predicting Customer Churn based on demographics and transactions data. It provides functions for exploratory data analysis, handling categorical features, random forest and logistic regression model training as well as saving associated results.
- Unit tests have also been written for each function in the library which simultaneoulsy log relevant errors and information for each step.


## Running Files
- [requirements](requirements.txt) file has been provided for environment setup. Please use the command ``pip install -r requirements.txt`` to install all dependencies. Specifically note the versions of the following dependencies which are different from the default Udacity workspace:
    - scikit-learn==0.22
    - numpy==1.19.5
    - seaborn==0.11.2
    - shap==0.40.0
    - pylint==2.13.5
    - autopep8==1.6.0
- The paths and features list used by functions are present in [constants](constants.py). The details are as follows:
    - DATA_path: Path of data CSV file. Default: './data/bank_data.csv'
    - MODELS_path: Path to save pickled models. Default: './models/'
    - EDA_path: Path to save EDA plots. Default: './images/eda/'
    - RESULTS_path: Path to save classification reports, ROC curves and feature importance plot. Default: './images/results/'
    - categorical_feats_lst: List of categorical features in the data. Required by <encoder_helper> function in [library](churn_library.py).
    - final_feats_lst: List of final features to use for model training. Required by <perform_feature_engineering> function in [library](churn_library.py).
- [library](churn_library.py) can be run by using the following command ``ipython churn_library.py``. The steps performed in the main block are as follows:
    - <import_data(DATA_path)>; Reads CSV data and returns a DataFrame. 
    - <perform_eda(df_data)>; Performs EDA such as histogram, bar and correlation plots. Saves figures in 'EDA_path'.
    - <encoder_helper(df_data, categorical_feats_lst)>; Converts each categorical feature into numeric values by considering the proportion of churn for each category in such features. 'categorical_feats_lst'. Returns an encoded DataFrame.
    - <perform_feature_engineering(df_Encoded, final_feats_lst)>; Selects a subset of features ('final_feats_lst') from encoded DataFrame and splits the data into train and test.
    - <train_models(X_Train, X_Test, y_Train, y_Test)>; Trains Random Forest and Logistic Regression models. Uses GridSearch to optimize hyperparamters for Random Forest. Saves models as pickle files in 'MODELS_path'. Generates and saves classification reports as well as Random Forest based feature importances in 'RESULTS_path'. 
- [tests](churn_script_logging_and_tests.py) can be run by using the following command ``ipython churn_script_logging_and_tests.py`` in terminal. The models and results will be stored as per the paths in [constants](constants.py). The .log file containing results from running all tests will be stored in the logging path set in [tests](churn_script_logging_and_tests.py). 


