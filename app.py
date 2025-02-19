import streamlit as st
import pandas as pd
import pickle

# Load your pre-trained models
try:
    with open('model.pkl', 'rb') as model_file:
        purchase_model = pickle.load(model_file)
    with open('product_model.pkl', 'rb') as product_model_file:  # Load the new product prediction model
        product_model = pickle.load(product_model_file)
    st.write("Models loaded successfully!")
except Exception as e:
    purchase_model = None
    product_model = None
    st.error(f"Error loading models: {e}")

# Ensure models are loaded before using them
if purchase_model is None or product_model is None:
    st.stop()  # Stop the app if the models are not loaded

# Load data
dis = pd.read_csv("Discount_Coupon.csv")
markt = pd.read_csv("Marketing_Spend.csv")
online_sales = pd.read_csv("Online_Sales.csv")
custmer = pd.read_excel("CustomersData.xlsx")
tax = pd.read_excel("Tax_amount.xlsx")

# Merge datasets
online_sales_custmer = pd.merge(online_sales, custmer, on="CustomerID", how="left")
o_c_t = pd.merge(online_sales_custmer, tax, on="Product_Category", how="left")
o_c_t.Transaction_Date = pd.to_datetime(o_c_t.Transaction_Date, format="%Y%m%d")

# Calculate the date difference
o_c_t = o_c_t.sort_values(by=['CustomerID', 'Transaction_Date'])  # Sort by CustomerID and Transaction_Date
o_c_t['Days_Since_Last_Transaction'] = o_c_t.groupby('CustomerID')['Transaction_Date'].diff().dt.days

# Title
st.title("Customer Segmentation and Next Purchase Prediction")

# Sidebar for user input
st.sidebar.header("User Input Features")

# User input for customer features
customer_id_input = st.sidebar.text_input("Enter Customer ID (or select from the dropdown)")

if customer_id_input:
    try:
        customer_id = int(customer_id_input)
    except ValueError:
        st.error("Cannot make predictions as no data is available for the entered Customer ID")
        st.stop()
else:
    customer_id = st.sidebar.selectbox("Select Customer ID", o_c_t['CustomerID'].unique())

customer_data = o_c_t[o_c_t['CustomerID'] == customer_id]

# Display customer data
if not customer_data.empty:
    st.write("Customer Data:")
    st.write(customer_data)
else:
    st.warning("No data found for the entered Customer ID.")

# Features for prediction
# Assuming these are the features used to train the product_model
expected_product_features = ['Quantity', 'Avg_Price', 'Delivery_Charges']  # Adjust this list based on your model

# Filter X_input to only include the expected features
X_input = customer_data[expected_product_features]

# Button to perform purchase prediction
if st.sidebar.button("Predict Next Purchase"):
    if purchase_model is not None:
        if not customer_data.empty:
            # Predict the next purchase using the pre-trained model
            next_purchase_prediction = purchase_model.predict(X_input)

            # Check the sum of 'Days_Since_Last_Transaction'
            total_days_since_last_transaction = customer_data['Days_Since_Last_Transaction'].sum()

            # Logic to determine likelihood based on days since last transaction
            if total_days_since_last_transaction > 1:
                st.write("Prediction: The customer is likely to make a next purchase (based on transaction history).")
            else:
                # Display the prediction from the model
                if next_purchase_prediction[0] == 1:
                    st.write("Prediction: The customer is likely to make a next purchase (based on model prediction).")
                else:
                    st.write("Prediction: The customer is unlikely to make a next purchase.")
        else:
            st.warning("Cannot make predictions as no data is available for the entered Customer ID.")
    else:
        st.error("Purchase model is not loaded, unable to make predictions.")

# Button to predict next product
if st.sidebar.button("Predict Next Product"):
    if product_model is not None:
        if not customer_data.empty:
            # Predict the next product using the pre-trained product model
            next_product_prediction = product_model.predict(X_input)

            # Display the prediction
            st.write(f"Prediction: The customer is likely to buy {next_product_prediction[0]}.")
        else:
            st.warning("Cannot make predictions as no data is available for the entered Customer ID.")
    else:
        st.error("Product model is not loaded, unable to make predictions.")