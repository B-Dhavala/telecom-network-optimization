#eda.py

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import logging

# Setup logging
logger = logging.getLogger(__name__)
# Create a directory for saving plots
os.makedirs('plots', exist_ok=True)

# Function to visualize outliers using box plots
def visualize_outliers(df, columns):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(columns):
        plt.subplot(3, 2, i + 1)
        sns.boxplot(y=df[column])
        plt.title(f'Box Plot of {column}')
    plt.tight_layout()
    plot_filename = os.path.join('plots', 'outliers_boxplot.png')
    plt.savefig(plot_filename)
    logger.info(f"Outlier Boxplot saved to {plot_filename}")
    plt.close()

# Function to detect outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# ADF Test for stationarity
def adf_test(data):
    adf_result = adfuller(data)

    # Log the complete result of the ADF test
    logger.info(f"ADF Statistic: {adf_result[0]}")
    logger.info(f"p-value: {adf_result[1]}")

    
    # Return the ADF Statistic and p-value
    return adf_result[0], adf_result[1]

# Count Plot for categorical variables
def count_plots(df, categorical_columns):
    for column in categorical_columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x=column, order=df[column].value_counts().index)
        plt.title(f'Count Plot of {column}')
        plt.xticks(rotation=45)
        plot_filename = os.path.join('plots', f'{column}_countplot.png')
        plt.savefig(plot_filename)
        logger.info(f"Count Plot for {column} saved to {plot_filename}")
        plt.close()

# Monthly Call Volume Plot
def monthly_call_volume_plot(col):

    # Plotting the monthly call volume
    plt.figure(figsize=(12, 6))
    col.plot(title='Monthly Call Volume', color='b')
    plt.xlabel('Months')
    plt.ylabel('Number of Calls')
    
    # Save the plot
    plot_filename = os.path.join('plots', 'monthly_call_volume.png')
    plt.savefig(plot_filename)
    
    logger.info(f"Monthly Call Volume plot saved to {plot_filename}")
    plt.close()

# Seasonal Decomposition of Call Duration
def seasonal_decompose_call_volume(df, column='Call_Duration_Seconds'):
    # Resample the data to monthly frequency
    df_resampled = df.resample('M').mean()  # Resample to monthly mean
    decomposition = seasonal_decompose(df_resampled[column], model='additive', period=12)  # 'period=12' for monthly data

    # Plot the decomposition
    plt.figure(figsize=(12, 8))
    decomposition.plot()
    plt.suptitle(f'Seasonal Decomposition of {column}', fontsize=16)
    decomposition_filename = os.path.join('plots', f'seasonal_decomposition_{column}.png')
    plt.savefig(decomposition_filename)
    logger.info(f"Seasonal Decomposition plot for {column} saved to {decomposition_filename}")
    plt.close()
