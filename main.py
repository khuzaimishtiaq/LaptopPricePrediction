
import streamlit as st
import joblib
import numpy as np

model = joblib.load("predicter.pkl")

st.title("PC & Laptop Price Prediction App")
feature_values = []
ohe_dict1 = {'Dell': 0, 'Lenovo': 1, 'HP': 2, 'Asus': 3, 'Acer': 4, 'Toshiba': 5, 'MSI': 6, 'Apple': 7, 'Samsung': 8, 'Mediacom': 9}
ohe_dict2 = {'256': 0, '1TB': 1, '128': 2, '512': 3, '500': 4, '32G': 5, '2TB': 6, '64G': 7, '16G': 8, '1GB': 9}
dict1 = {'Netbook': 0, 'Notebook': 1, '2 in 1 Convertible': 2, 'Ultrabook': 3, 'Gaming': 4, 'Workstation': 5}
dict2 = {'Intel Core i3': 0, 'Other Intel Processor': 1, 'AMD Processor': 2, 'Intel Core i5': 3, 'Intel Core i7': 4}
dict3 = {'Android': 0, 'Chrome OS': 1, 'Linux': 2, 'Windows': 3, 'Mac OS': 4}
dict4 = {10.1: 0, 14.1: 1, 11.6: 2, 17.0: 3, 15.6: 4, 14.0: 5, 11.3: 6, 12.0: 7, 13.3: 8, 12.3: 9, 13.0: 10, 13.5: 11, 17.3: 12, 12.5: 13, 13.9: 14, 15.0: 15, 15.4: 16, 18.4: 17}
dict5 = {2: 0, 4: 1, 6: 2, 8: 3, 12: 4, 16: 5, 24: 6, 32: 7}
ohe1 = ohe_dict1[st.selectbox('Pick the Manufacturer', ['Dell', 'Lenovo', 'HP', 'Asus', 'Acer', 'Toshiba', 'MSI', 'Apple', 'Samsung', 'Mediacom'])]
value = dict1[st.selectbox('Pick the Category', ['Netbook', 'Notebook', '2 in 1 Convertible', 'Ultrabook', 'Gaming', 'Workstation'])]
feature_values.append(value)
value = dict4[st.selectbox('Pick the screen Size', [10.1, 14.1, 11.6, 17.0, 15.6, 14.0, 11.3, 12.0, 13.3, 12.3, 13.0, 13.5, 17.3, 12.5, 13.9, 15.0, 15.4, 18.4])]
feature_values.append(value)
value = dict2[st.selectbox('Pick up CPU', ['Intel Core i3', 'Other Intel Processor', 'AMD Processor', 'Intel Core i5', 'Intel Core i7'])]
feature_values.append(value)
value = dict5[st.selectbox('Pick the RAM', [2, 4, 6, 8, 12, 16, 24, 32])]
feature_values.append(value)
ohe2 = ohe_dict2[st.selectbox('Pick the Storage', ['256', '1TB', '128', '512', '500', '32G', '2TB', '64G', '16G', '1GB'])]
value = dict3[st.selectbox('Pick the OS', ['Android', 'Chrome OS', 'Linux', 'Windows', 'Mac OS'])]
feature_values.append(value)
if st.button("Predict"):
    input_data = np.array(feature_values).reshape(1, -1)
    manufacturer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    storage = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    manufacturer[ohe1] = 1
    storage[ohe2] = 1
    arr1 = np.array(manufacturer).reshape(1, -1)
    arr2 = np.array(storage).reshape(1, -1)
    input_features = np.concatenate((input_data, arr1, arr2), axis=1)
    prediction = model.predict(input_features)
    print(input_data)
    st.write(f"Predicted  Price for PC: {prediction[0]:.2f}")
