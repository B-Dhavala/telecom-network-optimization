#data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
logger = logging.getLogger(__name__)
# Function to handle missing values
def handle_missing_values(df, numerical_columns, categorical_columns,geographical_columns):
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    for column in geographical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    return df

# One-Hot Encoding for categorical columns
def one_hot_encode(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Feature Engineering
def feature_engineering(df):
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Replace 'Date' with the actual datetime column name
        df.set_index('Timestamp', inplace=True)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Weekday'] = df.index.weekday
    return df

# Label Encoding for categorical variables
def label_encode(df):
    label_encoder_caller = LabelEncoder()
    label_encoder_receiver = LabelEncoder()

    df['Caller_Location'] = label_encoder_caller.fit_transform(df['Caller_Location'])
    df['Receiver_Location'] = label_encoder_receiver.fit_transform(df['Receiver_Location'])
    df['Combined_Label'] = df['Caller_Location'].astype(str) + ' ' + df['Receiver_Location'].astype(str)

    label_encoder_combined = LabelEncoder()
    df['Combined_Label_Encoded'] = label_encoder_combined.fit_transform(df['Combined_Label'])
    
    return df, label_encoder_caller, label_encoder_receiver, label_encoder_combined
