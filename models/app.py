import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Carbon Emission Predictor", layout="centered")


@st.cache_resource
def load_artifacts():
    """Load trained model and preprocessing artifacts."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        model = joblib.load(os.path.join(script_dir, "best_model_new.pkl"))
        consumption_scaler = joblib.load(os.path.join(script_dir, "consumption_scaler.pkl"))
        travel_scaler = joblib.load(os.path.join(script_dir, "travel_scaler.pkl"))
        recycling_items = joblib.load(os.path.join(script_dir, "recycling_items.pkl"))
        cooking_items = joblib.load(os.path.join(script_dir, "cooking_items.pkl"))
    except Exception as e:
        st.error(f"Could not load required files: {e}")
        st.stop()

    return model, consumption_scaler, travel_scaler, recycling_items, cooking_items


model, consumption_scaler, travel_scaler, recycling_items, cooking_items = load_artifacts()

# mappings used during feature engineering
AIR_MAP = {
    "never": 0,
    "rarely": 1,
    "frequently": 2,
    "very frequently": 3
}

WASTE_SIZE_MAP = {
    "small": 1,
    "medium": 2,
    "large": 3,
    "extra large": 4
}

st.title("Carbon Emission Predictor")
st.write("Enter lifestyle details below to estimate an individual's carbon emission.")

with st.form("prediction_form"):
    body_type = st.selectbox("Body Type", ["underweight", "normal", "overweight", "obese"])
    sex = st.selectbox("Sex", ["male", "female"])
    diet = st.selectbox("Diet", ["omnivore", "vegetarian", "vegan", "pescatarian"])

    how_often_shower = st.selectbox(
        "How Often Shower",
        ["less frequently", "daily", "more frequently", "twice a day"]
    )

    heating_energy_source = st.selectbox(
        "Heating Energy Source",
        ["electricity", "natural gas", "wood", "coal", "oil"]
    )

    transport = st.selectbox(
        "Transport",
        ["private", "public", "walk/bicycle"]
    )

    if transport == "private":
        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["lpg", "petrol", "diesel", "hybrid", "electric"]
        )
    else:
        vehicle_type = None
        st.text_input("Vehicle Type", value="", disabled=True)

    social_activity = st.selectbox(
        "Social Activity",
        ["never", "sometimes", "often"]
    )

    monthly_grocery_bill = st.number_input(
        "Monthly Grocery Bill",
        min_value=0.0,
        step=10.0
    )

    frequency_air = st.selectbox(
        "Frequency of Traveling by Air",
        ["never", "rarely", "frequently", "very frequently"]
    )

    vehicle_monthly_distance = st.number_input(
        "Vehicle Monthly Distance Km",
        min_value=0.0,
        step=10.0
    )

    waste_bag_size = st.selectbox(
        "Waste Bag Size",
        ["small", "medium", "large", "extra large"]
    )

    waste_bag_weekly_count = st.number_input(
        "Waste Bag Weekly Count",
        min_value=0,
        step=1
    )

    tv_pc_hours = st.number_input(
        "How Long TV PC Daily Hour",
        min_value=0.0,
        max_value=24.0,
        step=0.5
    )

    new_clothes_monthly = st.number_input(
        "How Many New Clothes Monthly",
        min_value=0,
        step=1
    )

    internet_hours = st.number_input(
        "How Long Internet Daily Hour",
        min_value=0.0,
        max_value=24.0,
        step=0.5
    )

    energy_efficiency = st.selectbox(
        "Energy efficiency",
        ["No", "Sometimes", "Yes"]
    )

    recycling = st.multiselect(
        "Recycling",
        options=recycling_items,
        default=[]
    )

    cooking_with = st.multiselect(
        "Cooking With",
        options=cooking_items,
        default=[]
    )

    submitted = st.form_submit_button("Predict Carbon Emission")


if submitted:
    # Create base input row
    input_data = pd.DataFrame([{
        "Body Type": body_type,
        "Sex": sex,
        "Diet": diet,
        "How Often Shower": how_often_shower,
        "Heating Energy Source": heating_energy_source,
        "Transport": transport,
        "Vehicle Type": vehicle_type,
        "Social Activity": social_activity,
        "Monthly Grocery Bill": monthly_grocery_bill,
        "Frequency of Traveling by Air": frequency_air,
        "Vehicle Monthly Distance Km": vehicle_monthly_distance,
        "Waste Bag Size": waste_bag_size,
        "Waste Bag Weekly Count": waste_bag_weekly_count,
        "How Long TV PC Daily Hour": tv_pc_hours,
        "How Many New Clothes Monthly": new_clothes_monthly,
        "How Long Internet Daily Hour": internet_hours,
        "Energy efficiency": energy_efficiency,
    }])

    # Add binary columns for recycling items
    for item in recycling_items:
        input_data[item] = 1 if item in recycling else 0

    # Add binary columns for cooking items
    for item in cooking_items:
        input_data[item] = 1 if item in cooking_with else 0

    # Feature 1: Total screen time
    input_data["TotalScreenTime"] = (
        input_data["How Long TV PC Daily Hour"] +
        input_data["How Long Internet Daily Hour"]
    )

    # Air travel score
    input_data["AirTravelScore"] = input_data["Frequency of Traveling by Air"].map(AIR_MAP)

    # Scale consumption columns
    consumption_values = input_data[["Monthly Grocery Bill", "How Many New Clothes Monthly"]].copy()
    scaled_consumption = consumption_scaler.transform(consumption_values)

    input_data["Monthly Grocery Bill"] = scaled_consumption[:, 0]
    input_data["How Many New Clothes Monthly"] = scaled_consumption[:, 1]

    input_data["ConsumptionScore"] = (
        input_data["Monthly Grocery Bill"] +
        input_data["How Many New Clothes Monthly"]
    )

    # Scale travel columns
    travel_values = input_data[["Vehicle Monthly Distance Km", "AirTravelScore"]].copy()
    scaled_travel = travel_scaler.transform(travel_values)

    input_data["Vehicle Monthly Distance Km"] = scaled_travel[:, 0]
    input_data["AirTravelScore"] = scaled_travel[:, 1]

    input_data["TravelIntensity"] = (
        input_data["Vehicle Monthly Distance Km"] +
        input_data["AirTravelScore"]
    )

    # Waste score
    input_data["WasteSizeScore"] = input_data["Waste Bag Size"].map(WASTE_SIZE_MAP)
    input_data["WasteScore"] = (
        input_data["Waste Bag Weekly Count"] * input_data["WasteSizeScore"]
    )

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.success(f"Estimated Carbon Emission: {prediction:.2f} kg CO₂e/month")

        with st.expander("See input data sent to model"):
            st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Model loaded from best_model_new.pkl")