
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from keras.saving import load_model

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="AML Detection System",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Intelligent Anti-Money Laundering Detection System")

st.write(
    "Enter transaction details below to identify whether "
    "a financial transaction is legitimate or potentially suspicious."
)

st.divider()

# -----------------------------------
# Load Trained Model and Preprocessor
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_assets():
    model = load_model(BASE_DIR / "AML.keras")
    preprocessor = joblib.load(BASE_DIR / "preprocessor")
    return model, preprocessor

try:
    model, preprocessor = load_assets()
except Exception as e:
    st.error(f"Unable to load model or preprocessor: {e}")
    st.stop()

# -----------------------------------
# Input Features
# -----------------------------------

currency_options = [
    'UK pounds', 'Indian rupee', 'Albanian lek',
    'Swiss franc', 'Pakistani rupee', 'Naira',
    'Yen', 'Euro', 'Dirham', 'Mexican Peso',
    'Turkish lira', 'US dollar', 'Moroccan dirham'
]

country_options = [
    'UK', 'Albania', 'Nigeria', 'Japan', 'Spain',
    'Switzerland', 'UAE', 'Italy', 'France',
    'Pakistan', 'Mexico', 'Austria', 'Germany',
    'Turkey', 'India', 'Morocco', 'USA', 'Netherlands'
]

payment_options = [
    'Cash Deposit', 'Cross-border', 'Cheque',
    'ACH', 'Credit card', 'Debit card',
    'Cash Withdrawal'
]

# -----------------------------------
# User Input Form
# -----------------------------------

with st.form("transaction_form"):

    st.subheader("Transaction Details")

    AMOUNT = st.number_input(
        "Transaction Amount",
        min_value=1.0,
        value=1000.0,
        step=100.0
    )

    col1, col2 = st.columns(2)

    with col1:
        Payment_Currency = st.selectbox(
            "Payment Currency",
            currency_options
        )

        Sender_bank_location = st.selectbox(
            "Sender Bank Location",
            country_options
        )

        Payment_type = st.selectbox(
            "Payment Type",
            payment_options
        )

    with col2:
        Received_Currency = st.selectbox(
            "Received Currency",
            currency_options
        )

        Receiver_bank_location = st.selectbox(
            "Receiver Bank Location",
            country_options
        )

    st.divider()

    submit = st.form_submit_button(
        "🔍 Analyze Transaction",
        use_container_width=True
    )

# -----------------------------------
# Prediction
# -----------------------------------

if submit:

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Amount': [AMOUNT],
        'Payment_currency': [Payment_Currency],
        'Received_currency': [Received_Currency],
        'Sender_bank_location': [Sender_bank_location],
        'Receiver_bank_location': [Receiver_bank_location],
        'Payment_type': [Payment_type]
    })

    st.subheader("Transaction Summary")
    st.dataframe(input_data, use_container_width=True)

    try:
        # Apply saved preprocessing
        processed_data = preprocessor.transform(input_data)

        # ANN prediction
        prediction = model.predict(
            processed_data,
            verbose=0
        )

        probability = float(prediction[0][0])

        # Classification
        if probability >= 0.5:

            st.error(
                "🚨 Potentially Suspicious Transaction Detected!"
            )

            st.write(
                "The ANN model has classified this transaction "
                "as potentially suspicious."
            )

            st.metric(
                "Suspicious Transaction Probability",
                f"{probability * 100:.2f}%"
            )

        else:

            st.success(
                "✅ Legitimate Transaction"
            )

            st.write(
                "The ANN model has classified this transaction "
                "as legitimate."
            )

            st.metric(
                "Suspicious Transaction Probability",
                f"{probability * 100:.2f}%"
            )

        st.caption(
            "Note: This is an ANN-based prediction, not a final "
            "determination of money laundering. Suspicious "
            "transactions should undergo further investigation."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------------------
# Footer
# -----------------------------------

st.divider()

st.caption(
    "Intelligent Anti-Money Laundering Detection System | "
    "Powered by Artificial Neural Networks"
)
