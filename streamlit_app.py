#streamlit_app.py
import streamlit as st
import pandas as pd
from data_preprocessing import feature_engineering
from model_saving import load_model

xg_model_ = load_model('models/xgboost_model.pkl')
kmeans = load_model('models/kmeans_model.pkl')
label_encoder_caller = load_model('models/label_encoder_caller.pkl')
label_encoder_receiver = load_model('models/label_encoder_receiver.pkl')
label_encoder_combined = load_model('models/label_encoder_combined.pkl')

st.title('Telecommunication Network Optimization')

with st.sidebar:
    st.header('Input Data')
    sample_data = {
        'Timestamp': st.text_input('Timestamp (MM/DD/YYYY HH MM)', '5/9/2023 9:29'),
        'Caller_ID': st.number_input('Caller ID', min_value=1, max_value=10000, step=1),
        'Call_Duration_Seconds': st.number_input('Call Duration (Seconds)', min_value=0, step=1),
        'Data_Transferred': st.number_input('Data Transferred', min_value=0.0, step=0.1),
        'Signal_Strength': st.number_input('Signal Strength (dBm)', min_value=-100.0, step=0.1),
        'Time_to_Connect_Seconds': st.number_input('Time to Connect (Seconds)', min_value=0.0, step=0.1),
        'Caller_Location': st.selectbox('Caller Location', ['Surat', 'Mumbai', 'Delhi', 'Chennai', 'Jaipur', 'Nagpur', 'Kolkata', 'Ahmedabad', 'Bengaluru', 'Hyderabad', 'Lucknow', 'Pune']),
        'Receiver_Location': st.selectbox('Receiver Location', ['Surat', 'Mumbai', 'Delhi', 'Chennai', 'Jaipur', 'Nagpur', 'Kolkata', 'Ahmedabad', 'Bengaluru', 'Hyderabad', 'Lucknow', 'Pune']),
        'Service_Type': st.selectbox('Service Type', ['Voice', 'Video', 'Data']),
        'Network_Tower_ID': st.selectbox('Network Tower ID', ['Tower_1', 'Tower_2', 'Tower_3', 'Tower_4', 'Tower_5', 'Tower_6', 'Tower_7', 'Tower_8', 'Tower_9', 'Tower_10', 'Tower_11', 'Tower_12', 'Tower_13', 'Tower_14', 'Tower_15', 'Tower_16', 'Tower_17', 'Tower_18', 'Tower_19']),
        'Customer_Plan': st.selectbox('Customer Plan', ['Prepaid', 'Postpaid']),
        'Roaming_Status': st.selectbox('Roaming Status', ['Yes', 'No']),
        'Network_Congestion_Level': st.selectbox('Network Congestion Level', ['Low', 'Medium', 'High'])
    }

sample_df = pd.DataFrame(sample_data, index=[0])
sample_df = feature_engineering(sample_df)  

categorical_columns = ['Service_Type', 'Network_Tower_ID', 'Customer_Plan', 'Roaming_Status', 'Network_Congestion_Level']
sample_df = pd.get_dummies(sample_df, columns=categorical_columns)

sample_df['Caller_Location'] = label_encoder_caller.transform(sample_df['Caller_Location'])
sample_df['Receiver_Location'] = label_encoder_receiver.transform(sample_df['Receiver_Location'])

sample_df['Combined_Label'] = sample_df['Caller_Location'].astype(str) + ' ' + sample_df['Receiver_Location'].astype(str)
sample_df['Combined_Label_Encoded'] = label_encoder_combined.transform(sample_df['Combined_Label'])

sample_cluster = kmeans.predict(sample_df[['Combined_Label_Encoded']])
sample_df['Cluster'] = sample_cluster[0]

train_columns = xg_model_.get_booster().feature_names

missing_cols = set(train_columns) - set(sample_df.columns)
for col in missing_cols:
    sample_df[col] = 0  

sample_df = sample_df[train_columns] 

predicted_bandwidth = xg_model_.predict(sample_df)

with st.expander('Input Data'):
    st.write(sample_df)

st.subheader('Predicted Allocated Bandwidth:')

predicted_value_html = f"""
    <p style="font-size: 20px; color: blue; font-weight: bold;">
        {predicted_bandwidth[0]:.2f}
    </p>
"""
st.markdown(predicted_value_html, unsafe_allow_html=True)
