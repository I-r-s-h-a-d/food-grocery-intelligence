
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("food_delivery_random_forest.pkl")

st.set_page_config(
    page_title="Food Delivery Intelligence",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Food & Grocery Delivery Intelligence")
st.subheader("AI-Powered Delivery Delay Prediction")

st.write(
    "Enter order details to predict the delivery status."
)

# Sidebar
st.sidebar.header("Order Information")

city = st.sidebar.selectbox(
    "Select City",
    ["Bengaluru", "Pune", "Chennai",
     "Mumbai", "Delhi NCR", "Hyderabad"]
)

order_hour = st.sidebar.slider(
    "Order Hour",
    0, 23, 13
)

order_day = st.sidebar.selectbox(
    "Order Day",
    ["Monday", "Tuesday", "Wednesday",
     "Thursday", "Friday", "Saturday", "Sunday"]
)

quantity = st.sidebar.number_input(
    "Total Quantity",
    min_value=1,
    value=3
)

order_value = st.sidebar.number_input(
    "Total Order Value",
    min_value=0.0,
    value=500.0
)

discount = st.sidebar.number_input(
    "Total Discount",
    min_value=0.0,
    value=20.0
)

promised_eta = st.sidebar.number_input(
    "Promised ETA (Minutes)",
    min_value=1,
    value=40
)

if st.button("Predict Delivery Status"):

    input_data = pd.DataFrame([{
        "city": city,
        "order_hour": order_hour,
        "order_day": order_day,
        "total_quantity": quantity,
        "total_order_value": order_value,
        "total_discount": discount,
        "promised_eta": promised_eta
    }])

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]

    late_probability = probabilities[1] * 100

    st.divider()

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Late Delivery Predicted")
        else:
            st.success("✅ On-Time Delivery Predicted")

    with col2:
        st.metric(
            "Predicted Late Delivery Probability",
            f"{late_probability:.2f}%"
        )

    st.progress(float(probabilities[1]))

    st.caption(
        "This is a model prediction, not a guaranteed delivery outcome."
    )

st.divider()

st.subheader("About This Project")

st.write("""
This application uses a trained Random Forest classification
pipeline to predict food and grocery delivery delays.

Technologies:
- Python
- Pandas
- Scikit-learn
- Random Forest
- Streamlit
""")
