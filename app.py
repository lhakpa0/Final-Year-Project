import os
import sys
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore", message="Found unknown categories")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from utils import apply_basic_features, apply_scaled_features, DEMOGRAPHIC_FEATURES

# Average from the training dataset, used to compare against predictions
DATASET_AVERAGE_KG_MONTH = 2269

# Real world reference figures for context (approximate monthly equivalents)
UK_AVERAGE_KG_MONTH = 917       # ~11 tonnes/year
GLOBAL_AVERAGE_KG_MONTH = 458   # ~5.5 tonnes/year

# Actionable tips mapped to feature names shown in SHAP output
ACTIONABLE_TIPS = {
    "Vehicle Monthly Distance Km": "Reduce monthly driving by using public transport, cycling, or combining trips.",
    "Transport": "Switching from a private car to public transport or cycling significantly cuts transport emissions.",
    "Frequency of Traveling by Air": "Cutting air travel, especially long haul flights, is one of the highest impact changes you can make.",
    "Diet": "Shifting toward a plant-based diet reduces emissions from food production.",
    "Monthly Grocery Bill": "Buying less, wasting less food, and choosing local produce reduces your consumption footprint.",
    "Heating Energy Source": "Switching to electric heating powered by renewables, or improving home insulation, reduces heating emissions.",
    "Energy efficiency": "Choosing energy-efficient appliances reduces electricity consumption over time.",
    "How Many New Clothes Monthly": "Buying fewer new clothes and choosing second-hand items reduces fast fashion emissions.",
    "Waste Bag Weekly Count": "Reducing waste through less packaging, composting food scraps, and recycling more lowers your waste score.",
    "WasteScore": "Reducing the size and number of waste bags you produce each week lowers waste emissions.",
    "ConsumptionScore": "Reducing grocery spending and clothing purchases together lowers your overall consumption footprint.",
    "Social Activity": "Being mindful of the carbon cost of frequent dining out and events can make a small difference.",
    "How Long TV PC Daily Hour": "Reducing screen time slightly lowers electricity use from devices.",
    "How Long Internet Daily Hour": "Reducing internet usage slightly lowers energy consumed by devices and data centres.",
}

# Rough min/max from training data
TRAINING_RANGES = {
    "Monthly Grocery Bill": (0, 500),
    "Vehicle Monthly Distance Km": (0, 5000),
    "How Long TV PC Daily Hour": (0, 12),
    "How Long Internet Daily Hour": (0, 12),
    "Waste Bag Weekly Count": (0, 8),
    "How Many New Clothes Monthly": (0, 75),
}

def validate_inputs(inputs):
    # Warn the user if any input is unusual or outside training range
    issues = []

    # Check if numeric value exceeds training max
    for field, (lo, hi) in TRAINING_RANGES.items():
        value = inputs.get(field)
        if value is not None and value > hi:
            issues.append((
                "warning",
                f"**{field}** ({value}) is above the maximum the model was trained on "
                f"({hi}). The prediction may be less reliable for this value."
            ))

    # Transport type vs vehicle distance mismatch
    transport = inputs.get("Transport", "")
    vehicle_distance = inputs.get("Vehicle Monthly Distance Km", 0)

    if transport == "private" and vehicle_distance == 0:
        issues.append((
            "info",
            "You selected private transport but distance is 0 km. "
            "Entering your actual monthly distance will improve accuracy."
        ))

    # Diet vs grocery bill mismatch
    diet = inputs.get("Diet", "")
    grocery_bill = inputs.get("Monthly Grocery Bill", 0)

    if diet in ("vegan", "vegetarian") and grocery_bill > 400:
        issues.append((
            "info",
            f"A **{diet}** diet with a grocery bill of **${grocery_bill:.0f}/month** "
            f"is unusual in the training data. The model may associate high grocery "
            f"spending with higher-emission diets."
        ))

    if grocery_bill == 0:
        issues.append((
            "info",
            "A grocery bill of **$0** is unusual. Entering an estimate "
            "will produce a more meaningful prediction."
        ))

    # Nothing selected for recycling or cooking
    if inputs.get("_recycling_count", 0) == 0 and inputs.get("_cooking_count", 0) == 0:
        issues.append((
            "info",
            "No recycling or cooking methods selected. If that's not accurate, "
            "selecting your actual habits will improve the prediction."
        ))

    # Both screen time fields at zero
    if inputs.get("How Long TV PC Daily Hour", 0) == 0 and inputs.get("How Long Internet Daily Hour", 0) == 0:
        issues.append((
            "info",
            "Both screen time values are **0 hours**. Even a small value "
            "will give a more realistic prediction if you use screens at all."
        ))

    return issues


def is_demographic_feature(feature_name):
    # Check if a feature is demographic so it can be hidden from SHAP charts
    return any(
        feature_name == d or feature_name.startswith(d + "_")
        for d in DEMOGRAPHIC_FEATURES
    )

@st.cache_resource
def load_artifacts():
    # Load the trained model and scalers from disk, cached so it only runs once
    try:
        model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        consumption_scaler = joblib.load(os.path.join(MODELS_DIR, "consumption_scaler.pkl"))
        recycling_items = joblib.load(os.path.join(MODELS_DIR, "recycling_items.pkl"))
        cooking_items = joblib.load(os.path.join(MODELS_DIR, "cooking_items.pkl"))
    except Exception as e:
        st.error(f"Could not load required files: {e}")
        st.stop()

    return model, consumption_scaler, recycling_items, cooking_items

def main():
    st.set_page_config(page_title="Carbon Emission Predictor", layout="centered")
    model, consumption_scaler, recycling_items, cooking_items = load_artifacts()

    st.title("Carbon Emission Predictor")
    st.markdown(
        "Answer a few questions about your lifestyle and we'll estimate your "
        "**monthly carbon footprint** in kg CO₂e. The more accurate your answers, "
        "the better the estimate."
    )

    with st.form("prediction_form"):
        st.subheader("About You")
        st.caption("Basic personal details that influence metabolism and consumption patterns.")

        col1, col2, col3 = st.columns(3)
        with col1:
            body_type = st.selectbox(
                "Body Type",
                ["underweight", "normal", "overweight", "obese"],
                help="Your body type category."
            )
        with col2:
            sex = st.selectbox(
                "Sex",
                ["male", "female"],
                help="Your biological sex."
            )
        with col3:
            diet = st.selectbox(
                "Diet",
                ["omnivore", "vegetarian", "vegan", "pescatarian"],
                help="Your primary diet. Pescatarian = vegetarian + fish."
            )

        st.markdown("")

        st.subheader("Home & Energy")
        st.caption("How you heat your home, use water, and your stance on energy efficiency.")

        col1, col2 = st.columns(2)
        with col1:
            how_often_shower = st.selectbox(
                "How often do you shower?",
                ["less frequently", "daily", "more frequently", "twice a day"],
                help="Your frequency of showering."
            )
        with col2:
            heating_energy_source = st.selectbox(
                "Heating Energy Source",
                ["electricity", "natural gas", "wood", "coal"],
                help="Your residential heating energy source. If you use multiple, pick the main one."
            )

        energy_efficiency = st.selectbox(
            "Do you prioritise energy-efficient appliances?",
            ["No", "Sometimes", "Yes"],
            help="Whether or not you care about purchasing energy-efficient devices."
        )

        st.markdown("")
        st.subheader("Transport & Travel")
        st.caption("How you get around day-to-day and how often you fly.")

        col1, col2 = st.columns(2)
        with col1:
            transport = st.selectbox(
                "Primary mode of transport",
                ["private", "public", "walk/bicycle"],
                help="Your transportation preference. 'Private' = car/motorcycle. 'Public' = bus, train, metro."
            )
        with col2:
            if transport == "private":
                vehicle_type = st.selectbox(
                    "Vehicle fuel type",
                    ["lpg", "petrol", "diesel", "hybrid", "electric"],
                    help="Your vehicle's fuel type."
                )
            else:
                vehicle_type = None
                st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            vehicle_monthly_distance = st.number_input(
                "Monthly driving distance (all sources)",
                min_value=0.0, max_value=20000.0, step=10.0,
                help="Monthly km of driving from all sources (personal car, rentals, carpools, borrowed vehicles, etc.)"
            )
        with col2:
            frequency_air = st.selectbox(
                "How often do you fly?",
                ["never", "rarely", "frequently", "very frequently"],
                help="Frequency of travelling by air. Rarely = 1-2 flights/year. Frequently = monthly."
            )

        st.markdown("")

        st.subheader("Consumption & Shopping")
        st.caption("Your spending and purchasing habits.")

        col1, col2 = st.columns(2)
        with col1:
            monthly_grocery_bill = st.number_input(
                "Monthly grocery bill ($)",
                min_value=0.0, max_value=2000.0, step=10.0, format="%.2f",
                help="Monthly amount spent on groceries. The dataset uses USD values."
            )
        with col2:
            new_clothes_monthly = st.number_input(
                "New clothes purchased per month",
                min_value=0, max_value=200, step=1,
                help="Number of individual clothing items purchased monthly."
            )

        social_activity = st.selectbox(
            "How often do you participate in social activities?",
            ["never", "sometimes", "often"],
            help="Frequency of participating in social activities (dining out, events, entertainment)."
        )

        st.markdown("")

        st.subheader("Screen Time")
        st.caption("Daily time spent on screens.")

        col1, col2 = st.columns(2)
        with col1:
            tv_pc_hours = st.number_input(
                "Hours on TV / PC per day",
                min_value=0.0, max_value=24.0, step=0.5,
                help="Daily time spent in front of TV or PC."
            )
        with col2:
            internet_hours = st.number_input(
                "Hours on the Internet per day",
                min_value=0.0, max_value=24.0, step=0.5,
                help="Time spent on the Internet daily. Can overlap with TV/PC time, that's fine."
            )

        st.markdown("")
        st.subheader("Waste")
        st.caption("How much waste you produce and what you recycle.")

        col1, col2 = st.columns(2)
        with col1:
            waste_bag_size = st.selectbox(
                "Waste bag size",
                ["small", "medium", "large", "extra large"],
                help="Size of your garbage bag. Small = ~20L, Medium = ~40L, Large = ~60L, Extra large = 100L+."
            )
        with col2:
            waste_bag_weekly_count = st.number_input(
                "Bags of waste per week",
                min_value=0, max_value=30, step=1,
                help="The amount of garbage thrown away in the last week."
            )
        recycling = st.multiselect(
            "What do you recycle?",
            options=sorted(recycling_items), default=[],
            help="The wastes you recycle. Select all that apply."
        )

        st.markdown("")

        st.subheader("Cooking")
        st.caption("What you cook with affects your energy consumption.")

        cooking_with = st.multiselect(
            "What do you cook with?",
            options=sorted(cooking_items), default=[],
            help="Devices used in cooking. Select all that apply."
        )
        st.markdown("")
        st.markdown("---")

        submitted = st.form_submit_button("Calculate My Carbon Footprint", use_container_width=True)

    if submitted:
        validation_inputs = {
            "Transport": transport,
            "Vehicle Monthly Distance Km": vehicle_monthly_distance,
            "Diet": diet,
            "Monthly Grocery Bill": monthly_grocery_bill,
            "How Long TV PC Daily Hour": tv_pc_hours,
            "How Long Internet Daily Hour": internet_hours,
            "Waste Bag Weekly Count": waste_bag_weekly_count,
            "How Many New Clothes Monthly": new_clothes_monthly,
            "_recycling_count": len(recycling),
            "_cooking_count": len(cooking_with),
        }

        issues = validate_inputs(validation_inputs)

        if issues:
            with st.expander("Input check - review before trusting this prediction", expanded=True):
                for level, message in issues:
                    if level == "warning":
                        st.warning(message)
                    else:
                        st.info(message)

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

        for item in recycling_items:
            input_data[item] = 1 if item in recycling else 0
        for item in cooking_items:
            input_data[item] = 1 if item in cooking_with else 0

        input_data = apply_basic_features(input_data)
        input_data = apply_scaled_features(input_data, consumption_scaler)

        try:
            prediction = max(model.predict(input_data)[0], 0)

            st.markdown("---")
            st.subheader("Your Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Your Carbon Footprint", value=f"{prediction:.0f} kg")
            with col2:
                st.metric(
                    label="Dataset Average",
                    value=f"{DATASET_AVERAGE_KG_MONTH} kg",
                    help="Approximate average monthly carbon footprint per person in this dataset."
                )
            with col3:
                diff = prediction - DATASET_AVERAGE_KG_MONTH
                st.metric(
                    label="Difference",
                    value=f"{abs(diff):.0f} kg",
                    delta=f"{'above' if diff > 0 else 'below'} average",
                    delta_color="inverse"
                )

            # Annual equivalent and real world context
            annual_kg = prediction * 12
            st.markdown(f"**Annual equivalent:** approximately **{annual_kg / 1000:.1f} tonnes** CO\u2082e per year.")
            st.caption(
                f"Real world reference: UK average \u2248 {UK_AVERAGE_KG_MONTH * 12 / 1000:.0f} tonnes/year · "
                f"Global average \u2248 {GLOBAL_AVERAGE_KG_MONTH * 12 / 1000:.0f} tonnes/year "
                f"(note: these figures are based on real data while this model uses a synthetic dataset)"
            )

            pct_diff = ((prediction - DATASET_AVERAGE_KG_MONTH) / DATASET_AVERAGE_KG_MONTH) * 100
            if prediction > DATASET_AVERAGE_KG_MONTH * 1.5:
                st.error(
                    f"Your footprint is **{pct_diff:.0f}% above** average. "
                    "The biggest areas to look at are usually transport, diet, and energy source."
                )
            elif prediction > DATASET_AVERAGE_KG_MONTH:
                st.warning(
                    f"Your footprint is **{pct_diff:.0f}% above** average. "
                    "Small changes in a few areas could bring it down."
                )
            else:
                st.success(
                    f"Your footprint is **{abs(pct_diff):.0f}% below** average. "
                    "Great job, keep it up!"
                )

            if issues:
                warning_count = sum(1 for level, _ in issues if level == "warning")
                if warning_count >= 2:
                    st.caption(
                        "**Low confidence** - several inputs fall outside the model's "
                        "training range. Treat this estimate with caution."
                    )
                elif warning_count == 1:
                    st.caption(
                        "**Moderate confidence** - one input is outside the model's "
                        "training range. The estimate is likely reasonable but may be less precise."
                    )

            try:
                import shap
                from sklearn.pipeline import Pipeline as SKPipeline

                preprocessor = SKPipeline(model.steps[:-1])
                X_transformed = preprocessor.transform(input_data)

                feature_names = model.named_steps["preprocessor"].get_feature_names_out()
                clean_names = [
                    n.replace("num__", "").replace("nom__", "").replace("ord__", "")
                    for n in feature_names
                ]

                shap_values = shap.TreeExplainer(model.named_steps["model"]).shap_values(X_transformed)
                shap_df = pd.DataFrame({"Feature": clean_names, "SHAP Value": shap_values[0]})

                shap_df = shap_df[~shap_df["Feature"].apply(is_demographic_feature)]
                shap_df["Abs"] = shap_df["SHAP Value"].abs()
                shap_df = shap_df.sort_values("Abs", ascending=False).head(10).drop(columns="Abs")

                colors = ["#d62728" if v > 0 else "#1f77b4" for v in shap_df["SHAP Value"]]

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(shap_df["Feature"][::-1], shap_df["SHAP Value"][::-1], color=colors[::-1])
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP value (kg CO2e/month impact)")
                ax.set_title("What's driving this prediction?")
                plt.tight_layout()

                with st.expander("What's pushing your number up or down?", expanded=True):
                    st.pyplot(fig)
                    st.caption("Red = pushes emission higher. Blue = brings it down. Only actionable factors shown.")

                plt.close(fig)

                # Top 3 actionable recommendations based on highest positive SHAP contributors
                top_actions = shap_df[shap_df["SHAP Value"] > 0].head(3)
                if not top_actions.empty:
                    st.subheader("Your top areas to reduce")
                    for _, action_row in top_actions.iterrows():
                        feature = action_row["Feature"]
                        tip = next(
                            (v for k, v in ACTIONABLE_TIPS.items() if k.lower() in feature.lower() or feature.lower() in k.lower()),
                            f"Consider reducing your impact in: {feature}"
                        )
                        st.info(f"**{feature}** is adding **+{action_row['SHAP Value']:.0f} kg/month** to your footprint.  \n{tip}")

            except ImportError:
                st.info("Install shap to see which factors are driving this prediction.")
            except Exception as e:
                st.warning(f"Could not compute SHAP values: {e}")

            feat_imp_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
            if os.path.exists(feat_imp_path):
                feat_df = pd.read_csv(feat_imp_path).copy()
                feat_df["Feature"] = feat_df["Feature"].str.replace(r"^(num|nom|ord)__", "", regex=True)

                feat_df = feat_df[~feat_df["Feature"].apply(is_demographic_feature)]
                feat_df = feat_df.head(10).sort_values("Importance")

                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.barh(feat_df["Feature"], feat_df["Importance"], color="#2ca02c")
                ax2.set_xlabel("Importance")
                ax2.set_title("Which lifestyle factors matter most?")
                plt.tight_layout()

                with st.expander("Which factors matter most? (across all predictions)"):
                    st.pyplot(fig2)

                plt.close(fig2)

            export_data = {
                "Predicted CO2e (kg/month)": round(prediction, 2),
                "vs Average (kg/month)": DATASET_AVERAGE_KG_MONTH,
                "Difference (kg/month)": round(prediction - DATASET_AVERAGE_KG_MONTH, 2),
                "Body Type": body_type,
                "Sex": sex,
                "Diet": diet,
                "How Often Shower": how_often_shower,
                "Heating Energy Source": heating_energy_source,
                "Energy efficiency": energy_efficiency,
                "Transport": transport,
                "Vehicle Type": vehicle_type or "N/A",
                "Vehicle Monthly Distance Km": vehicle_monthly_distance,
                "Frequency of Traveling by Air": frequency_air,
                "Monthly Grocery Bill": round(monthly_grocery_bill, 2),
                "How Many New Clothes Monthly": new_clothes_monthly,
                "Social Activity": social_activity,
                "How Long TV PC Daily Hour": tv_pc_hours,
                "How Long Internet Daily Hour": internet_hours,
                "Waste Bag Size": waste_bag_size,
                "Waste Bag Weekly Count": waste_bag_weekly_count,
                "Recycling": ", ".join(recycling) if recycling else "None",
                "Cooking With": ", ".join(cooking_with) if cooking_with else "None",
            }

            if issues:
                export_data["Validation Notes"] = "; ".join(
                    msg.replace("**", "") for _, msg in issues
                )

            csv = pd.DataFrame([export_data]).to_csv(index=False)
            st.download_button(
                "Download Result as CSV",
                data=csv,
                file_name="carbon_prediction.csv",
                mime="text/csv"
            )

            with st.expander("See the data that was sent to the model"):
                st.dataframe(input_data)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.caption(
        "Predictions are made using a machine learning model trained on synthetic lifestyle data. "
        "Results are estimates only and should not be treated as precise measurements. "
        "Real-world carbon footprints depend on many additional factors not captured here."
    )

main()
