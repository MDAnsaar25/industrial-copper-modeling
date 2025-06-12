import streamlit as st
import numpy as np
import pandas as pd
import pickle
from streamlit_option_menu import option_menu

# === Load dataset for dropdown values ===
df = pd.read_csv("data/Copper_Set.csv")

# === Mappings ===
item_type_mapping = {'W': 1, 'WI': 2, 'S': 3, 'Others': 4, 'PL': 5, 'IPL': 6, 'SLAWR': 7}
status_mapping = {'Lost': 0, 'Won': 1}
status_reverse_mapping = {0: 'Lost', 1: 'Won'}

# === Load models and scalers ===
with open("models\Regression_Model.pkl", "rb") as f:
    regression_model = pickle.load(f)
with open("models\Classification_Model.pkl", "rb") as f:
    classification_model = pickle.load(f)
with open("models\scaler_reg.pkl", "rb") as f:
    scaler_reg = pickle.load(f)
with open("models\scaler_clf.pkl", "rb") as f:
    scaler_clf = pickle.load(f)

# === Set Streamlit page config ===
st.set_page_config(page_title="Industrial Copper Modeling", page_icon="🏭", layout="wide")
st.title("🏭 Industrial Copper Modeling")

# === Option menu ===
selected = option_menu(
    menu_title="Model Selection",
    options=["Predict Selling Price", "Predict Status"],
    icons=["cash-coin", "check-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# === Common input fields ===
st.subheader("📥 Input Features")
col1, col2, col3 = st.columns(3)

with col1:
    quantity = st.text_input("Quantity (Tons)", "")
    thickness = st.text_input("Thickness", "")
    width = st.text_input("Width", "")

with col2:
    customer = st.text_input("Customer ID", "")
    country = st.selectbox("Country", sorted(df["country"].dropna().unique()))
    application = st.selectbox("Application", sorted(df["application"].dropna().unique()))

with col3:
    item_type = st.selectbox("Item Type", sorted(df["item type"].dropna().unique()))
    product_ref = st.text_input("Product Reference", "")
    if selected == "Predict Status":
        selling_price = st.text_input("Selling Price", "")

# === Fireworks animation function ===
def show_fireworks():
    st.balloons()  # You can also use st.snow()

# === REGRESSION ===
if selected == "Predict Selling Price":
    if st.button("Predict Selling Price"):
        try:
            all_inputs = [quantity, thickness, width, customer, product_ref]
            if not all(all_inputs):
                st.warning("⚠️ Please fill all the required fields.")
            else:
                inputs = np.array([[float(quantity),
                                    status_mapping["Won"],  # Dummy status
                                    item_type_mapping.get(item_type, 0),
                                    float(application),
                                    np.log(float(thickness) + 1e-6),
                                    float(width),
                                    float(country),
                                    float(customer),
                                    float(product_ref)]])
                scaled = scaler_reg.transform(inputs)
                pred_log = regression_model.predict(scaled)[0]
                pred_price = np.exp(pred_log)
                st.success(f"💰 Predicted Selling Price: ${round(pred_price, 2)}")
                show_fireworks()
        except Exception as e:
            st.error(f"❌ Error: {e}")

# === CLASSIFICATION ===
if selected == "Predict Status":
    if st.button("Predict Status"):
        try:
            all_inputs = [quantity, thickness, width, customer, product_ref, selling_price]
            if not all(all_inputs):
                st.warning("⚠️ Please fill all the required fields.")
            else:
                inputs = np.array([[np.log(float(quantity) + 1e-6),
                                    np.log(float(selling_price) + 1e-6),
                                    item_type_mapping.get(item_type, 0),
                                    float(application),
                                    np.log(float(thickness) + 1e-6),
                                    float(width),
                                    float(country),
                                    float(customer),
                                    float(product_ref)]])
                scaled = scaler_clf.transform(inputs)
                pred = classification_model.predict(scaled)[0]
                st.success(f"📝 Predicted Status: {status_reverse_mapping[pred]}")
                show_fireworks()
        except Exception as e:
            st.error(f"❌ Error: {e}")
