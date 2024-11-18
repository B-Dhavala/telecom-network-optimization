#retraining.py
from sklearn.model_selection import train_test_split
from model_training import train_xgboost
from kmeans_clustering import kmeans_clustering
from model_saving import save_model
from data_preprocessing import handle_missing_values, feature_engineering, one_hot_encode, label_encode
import pandas as pd
def retrain_model_and_save():
    # Load and preprocess the data
    df = pd.read_csv('caller.csv')
    df = handle_missing_values(df, numerical_columns=['Call_Duration_Seconds', 'Data_Transferred', 'Signal_Strength', 'Time_to_Connect_Seconds', 'Allocated_Bandwidth'], 
                               categorical_columns=['Service_Type', 'Network_Tower_ID', 'Customer_Plan', 'Call_Direction', 'Roaming_Status', 'Network_Congestion_Level'],
                               geographical_columns=['Caller_Location', 'Receiver_Location'])

    df = feature_engineering(df)
    df, le_caller, le_receiver, le_combined = label_encode(df)
    df = one_hot_encode(df, categorical_columns=['Service_Type', 'Network_Tower_ID', 'Customer_Plan', 'Call_Direction', 'Roaming_Status', 'Network_Congestion_Level'])

    # Perform KMeans Clustering
    kmeans = kmeans_clustering(df)

    # Train XGBoost model
    X = df.drop(columns=['Allocated_Bandwidth', 'Combined_Label', 'Combined_Label_Encoded'])
    y = df['Allocated_Bandwidth']
    
    # Train-test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xg_model, mse, r2 = train_xgboost(X_train, y_train, X_test, y_test)

    # Save models
    save_model(le_caller, 'models/label_encoder_caller.pkl')
    save_model(le_receiver, 'models/label_encoder_receiver.pkl')
    save_model(le_combined, 'models/label_encoder_combined.pkl')
    save_model(kmeans, 'models/kmeans_model.pkl')
    save_model(xg_model, 'models/xgboost_model.pkl')
