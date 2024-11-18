import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from eda import visualize_outliers, detect_outliers_iqr, adf_test, count_plots, monthly_call_volume_plot, seasonal_decompose_call_volume
from data_preprocessing import handle_missing_values, one_hot_encode, feature_engineering, label_encode
from model_training import train_xgboost, cross_validate
from kmeans_clustering import kmeans_clustering
from model_saving import save_model, load_model

# Setup logging
logging.getLogger('matplotlib').setLevel(logging.WARNING) 
logging.getLogger('PIL').setLevel(logging.WARNING) 
logging.getLogger('seaborn').setLevel(logging.WARNING)
log_filename = 'log/data.log'

# Configure the logging system
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,  # Log all levels from DEBUG and above
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format the log messages
    filemode='w'  # Use 'w' to overwrite the log file each time you run the program
)

# Ensure we have a logger instance
logger = logging.getLogger()

# Log the start of the execution
logger.info("Starting the script execution.")

# Load data
df = pd.read_csv('caller.csv')

# EDA
numerical_columns = ['Call_Duration_Seconds', 'Data_Transferred', 'Signal_Strength', 'Time_to_Connect_Seconds', 'Allocated_Bandwidth']
categorical_columns = ['Service_Type', 'Network_Tower_ID', 'Customer_Plan', 'Call_Direction', 'Roaming_Status', 'Network_Congestion_Level']
geographical_columns = ['Caller_Location', 'Receiver_Location']

# Visualize outliers
visualize_outliers(df, numerical_columns)

# Detect and log outliers in numerical columns
for column in numerical_columns:
    outliers = detect_outliers_iqr(df, column)
    if not outliers.empty:
        logger.info(f"Outliers in column {column}:\n{outliers}")

# Handle missing values
df = handle_missing_values(df, numerical_columns, categorical_columns, geographical_columns)

# Count plots for categorical and geographical variables
count_plots(df, categorical_columns)
count_plots(df, geographical_columns)

# Check if 'Timestamp' exists in the dataframe
if 'Timestamp' in df.columns:
    # Convert 'Timestamp' to datetime and set as index for time series analysis
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')  # 'coerce' to handle invalid parsing
else:
    logger.error("The column 'Timestamp' is missing from the dataset.")
    raise KeyError("The column 'Timestamp' is missing from the dataset.")

# Set 'Timestamp' column as the index
df.set_index('Timestamp', inplace=True)

# Resample the data by month and calculate the mean for numerical columns
monthly_data = df[numerical_columns].resample('M').mean()
monthly_calls = df.resample('M').size()
# Monthly Call Volume Plot
monthly_call_volume_plot(monthly_calls)

# Perform ADF Test on 'Allocated_Bandwidth'
adf_statistic, p_value = adf_test(monthly_data['Allocated_Bandwidth']) 

# Check stationarity
if p_value < 0.05:
    logging.info("The data is stationary.")
else:
    logging.info("The data is non-stationary.")

# Seasonal decomposition of call volume
seasonal_decompose_call_volume(monthly_data)

# Feature Engineering and Encoding
df = feature_engineering(df)
df, le_caller, le_receiver, le_combined = label_encode(df)
df = one_hot_encode(df, categorical_columns)

# KMeans Clustering
kmeans = kmeans_clustering(df)

# Train-Test Split
X = df.drop(columns=['Allocated_Bandwidth', 'Combined_Label', 'Combined_Label_Encoded'])
y = df['Allocated_Bandwidth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
xg_model, mse, r2 = train_xgboost(X_train, y_train, X_test, y_test)

# Cross-validation
avg_mse, avg_r2 = cross_validate(X, y)

# Save models
save_model(le_caller, 'models/label_encoder_caller.pkl')
save_model(le_receiver, 'models/label_encoder_receiver.pkl')
save_model(le_combined, 'models/label_encoder_combined.pkl')
save_model(kmeans, 'models/kmeans_model.pkl')
save_model(xg_model, 'models/xgboost_model.pkl')

# Load models for future use
le_caller = load_model('models/label_encoder_caller.pkl')
le_receiver = load_model('models/label_encoder_receiver.pkl')
le_combined = load_model('models/label_encoder_combined.pkl')
kmeans = load_model('models/kmeans_model.pkl')
xg_model = load_model('models/xgboost_model.pkl')
