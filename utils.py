import pandas as pd
import matplotlib.pyplot as plt
import re

# Helper function to plot missing values in dataframe
def plot_missing(df):
    data = [(col, df[col].isnull().sum() / len(df)) 
            for col in df.columns if df[col].isnull().sum() > 0]
    col_names = ['column', 'percent_missing']
    missing_df = pd.DataFrame(data, columns=col_names).sort_values('percent_missing')
    missing_df.plot(kind='barh', x='column', y='percent_missing'); 
    plt.title('Percent of missing values in colummns');
    
# plot number of unique values in categorical columns
def plot_distinct_categorical(df):
    categorical_columns = df.columns[df.dtypes == 'O'].tolist()
    data = [(col, df[col].nunique()) for col in df.columns if col in categorical_columns]
    col_names = ['column', 'number_unique_values']
    number_unique_df = pd.DataFrame(data, columns=col_names).sort_values('number_unique_values')
    number_unique_df.plot(kind='barh', x='column', y='number_unique_values'); 
    plt.title('Number of unique values in categorical columns');
    
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# Customer categorical transformer to ohe and label encode categorical columns, depending on number of unique values
class CustomEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, unique_value_threshold=2):
        self.unique_value_threshold = unique_value_threshold
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Find indices of one-hot columns with number of distinct values below threshold
        self.ohe_columns = [col_idx for col_idx in range(X.shape[1])
                            if len(np.unique(X[:, col_idx])) >= self.unique_value_threshold]
        # Apply OHE
        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        ohe_result = one_hot_encoder.fit_transform(X[:, self.ohe_columns])
        self.new_ohe_columns = one_hot_encoder.get_feature_names()
        
        # Pass if all columns were ohe (need a similar check for all le cols and exceptions for invalid threshold values)
        if len(self.ohe_columns) == X.shape[1]:
            output = ohe_result
        else:
            # Find label encoding columns, not in ohe
            self.label_enc_columns = [col_idx for col_idx in range(X.shape[1])
                                      if col_idx not in self.ohe_columns]
            # Apply column encoder for each column and stack result
            label_encoder = LabelEncoder()
            le_result = [LabelEncoder().fit_transform(X[:, col_idx]) 
                         for col_idx in self.label_enc_columns]
            le_result = np.stack(le_result).T
            
            # Apply scalers
            
            # Combine results of both encoders
            output = np.hstack((ohe_result, le_result))
        
        return output
        

# Recover ohe names from preprocessor pipeline (new_ohe_cols and ohe_col_idxs need to be adjusted for different pipelines)
# This probably needs to be a part of custom_encoder above and return proper ohe column attribute, instead of this lousy accessing
def recover_ohe_columns(preprocessor):    
    new_ohe_cols = preprocessor.transformers_[1][1].named_steps['custom_encoder'].new_ohe_columns    
    ohe_col_idxs = preprocessor.transformers_[1][1].named_steps['custom_encoder'].ohe_columns
    
    categorical_columns = preprocessor.transformers_[1][2]
    
    old_ohe_cols = [categorical_columns[idx] for idx in ohe_col_idxs]
    recovered_ohe_cols = []

    for new_ohe_col in new_ohe_cols:
        for idx, old_ohe_col in enumerate(old_ohe_cols):
            pattern = 'x' + str(idx) + '_'
            replacement = old_ohe_cols[idx] + '_'
            match = re.search(pattern, new_ohe_col)
            if match is not None:
                result = re.sub(pattern, replacement, new_ohe_col)
                recovered_ohe_cols.append(result)
            
    return recovered_ohe_cols
