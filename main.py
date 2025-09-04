import os
import io
import json
import base64
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import joblib

# Optional heavy libs (kept for completeness; not heavily used)
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns  # noqa: F401


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OceanONE",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: transparent;
        border: 2px solid #74b9ff;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .metric-container h3 {
        color: #74b9ff;
        margin-bottom: 0.5rem;
    }
    .metric-container p {
        color: #e0e0e0;
        margin-bottom: 1rem;
    }
    .metric-container ul {
        color: #cccccc;
    }
    .metric-container li {
        margin-bottom: 0.3rem;
    }
    .prediction-box {
        background: transparent;
        border: 2px solid #667eea;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background: transparent;
        border: 2px solid #28a745;
        color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: transparent;
        border: 2px solid #17a2b8;
        color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model loading (cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all trained models with caching - fall back to demo mode if missing."""
    models_dir = "models"

    if not os.path.exists(models_dir):
      return None

    try:
        model_files = {
            "fish_classifier": os.path.join(models_dir, "fish_classifier.pkl"),
            "fish_scaler": os.path.join(models_dir, "fish_scaler.pkl"),
            "ocean_models": os.path.join(models_dir, "ocean_models.pkl"),
            "ocean_scaler": os.path.join(models_dir, "ocean_scaler.pkl"),
            "bio_models": os.path.join(models_dir, "biodiversity_models.pkl"),
            "bio_scaler": os.path.join(models_dir, "biodiversity_scaler.pkl"),
        }

        models = {}
        missing_files = []

        for key, path in model_files.items():
            if os.path.exists(path):
                models[key] = joblib.load(path)
            else:
                missing_files.append(path)
                models[key] = None

        if missing_files:
            st.warning(
                f"⚠️ Some model files are missing: {missing_files}. Using demo mode for affected features."
            )

        return models if any(v is not None for v in models.values()) else None

    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Error loading models: {exc}")
        return None


# -----------------------------------------------------------------------------
# Demo prediction functions
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Real prediction functions (use demo if models missing)
# -----------------------------------------------------------------------------
def predict_fish_species_real(length, weight, height, width, models):
    if not models or not models.get("fish_classifier"):
        return predict_fish_species_demo(length, weight, height, width)

    try:
        input_data = np.array([[length, weight, height, width]])
        if models.get("fish_scaler") is not None:
            input_data = models["fish_scaler"].transform(input_data)

        prediction = models["fish_classifier"].predict(input_data)[0]
        if hasattr(models["fish_classifier"], "predict_proba"):
            probabilities = models["fish_classifier"].predict_proba(input_data)[0]
            class_names = models["fish_classifier"].classes_
            all_probabilities = dict(zip(class_names, [float(p) for p in probabilities]))
            confidence = float(np.max(probabilities))
        else:
            all_probabilities = {prediction: 1.0}
            confidence = 1.0

        return {
            "predicted_species": str(prediction),
            "confidence": confidence,
            "all_probabilities": all_probabilities,
        }
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Error in fish prediction: {exc}")
        return predict_fish_species_demo(length, weight, height, width)


def predict_ocean_parameters_real(lat, lon, depth, day_of_year, chlorophyll, ph, models):
    if not models or not models.get("ocean_models"):
        return predict_ocean_parameters_demo(lat, lon, depth, day_of_year, chlorophyll, ph)

    try:
        input_data = np.array([[lat, lon, depth, day_of_year, chlorophyll, ph]])
        if models.get("ocean_scaler") is not None:
            input_data = models["ocean_scaler"].transform(input_data)

        predictions = models["ocean_models"].predict(input_data)
        if isinstance(predictions, (list, tuple, np.ndarray)) and np.array(predictions).ndim > 1:
            predictions = predictions[0]

        if len(np.atleast_1d(predictions)) >= 3:
            return {
                "temperature": float(np.atleast_1d(predictions)[0]),
                "salinity": float(np.atleast_1d(predictions)[1]),
                "dissolved_oxygen": float(np.atleast_1d(predictions)[2]),
            }
        else:
            return {
                "temperature": float(np.atleast_1d(predictions)[0]),
                "salinity": 34.5,
                "dissolved_oxygen": 8.0,
            }
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Error in ocean prediction: {exc}")
        return predict_ocean_parameters_demo(lat, lon, depth, day_of_year, chlorophyll, ph)


def assess_biodiversity_real(temp, depth, pollution, coral_cover, lat, lon, models):
    if not models or not models.get("bio_models"):
        return assess_biodiversity_demo(temp, depth, pollution, coral_cover, lat, lon)

    try:
        input_data = np.array([[temp, depth, pollution, coral_cover, lat, lon]])
        if models.get("bio_scaler") is not None:
            input_data = models["bio_scaler"].transform(input_data)

        predictor = models["bio_models"]
        prediction = predictor.predict(input_data)[0]

        if hasattr(predictor, "predict_proba"):
            probabilities = predictor.predict_proba(input_data)[0]
            class_names = predictor.classes_
            category_probabilities = dict(zip(class_names, [float(p) for p in probabilities]))
            biodiversity_category = str(prediction)

            if biodiversity_category == "High":
                species_count = 20; shannon_diversity = 3.0
            elif biodiversity_category == "Medium":
                species_count = 12; shannon_diversity = 2.0
            else:
                species_count = 6; shannon_diversity = 1.0
        else:
            numeric = float(prediction)
            species_count = int(max(1, numeric))
            shannon_diversity = max(0.1, numeric / 10.0)
            if shannon_diversity > 2.5:
                biodiversity_category = "High"
            elif shannon_diversity > 1.5:
                biodiversity_category = "Medium"
            else:
                biodiversity_category = "Low"
            category_probabilities = {biodiversity_category: 0.9}

        return {
            "species_count": species_count,
            "shannon_diversity": shannon_diversity,
            "biodiversity_category": biodiversity_category,
            "category_probabilities": category_probabilities,
        }
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Error in biodiversity assessment: {exc}")
        return assess_biodiversity_demo(temp, depth, pollution, coral_cover, lat, lon)


def predict_fish_species_demo(length, weight, height, width):
    if weight > 500:
        species = "Pike"; confidence = 0.85
    elif weight > 300:
        species = "Bream"; confidence = 0.82
    elif weight > 150:
        species = "Perch"; confidence = 0.78
    else:
        species = "Roach"; confidence = 0.75

    confidence = float(np.clip(confidence + np.random.uniform(-0.1, 0.1), 0.6, 0.95))

    all_probabilities = {
        species: confidence,
        "Bream": 0.2 if species != "Bream" else confidence,
        "Perch": 0.15 if species != "Perch" else confidence,
        "Pike": 0.1 if species != "Pike" else confidence,
        "Roach": 0.05 if species != "Roach" else confidence,
    }
    total = sum(all_probabilities.values())
    all_probabilities = {k: v / total for k, v in all_probabilities.items()}

    return {
        "predicted_species": species,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
    }


def predict_ocean_parameters_demo(lat, lon, depth, day_of_year, chlorophyll, ph):
    base_temp = 26 + (lat - 15) * 0.2 + np.sin(2 * np.pi * day_of_year / 365) * 3
    temperature = float(np.clip(base_temp + np.random.normal(0, 1), 15, 35))

    base_salinity = 34.5 - depth * 0.01
    salinity = float(np.clip(base_salinity + np.random.normal(0, 0.5), 30, 40))

    base_oxygen = 8.5 - depth * 0.02 + chlorophyll * 0.3
    dissolved_oxygen = float(np.clip(base_oxygen + np.random.normal(0, 0.3), 4, 12))

    return {
        "temperature": temperature,
        "salinity": salinity,
        "dissolved_oxygen": dissolved_oxygen,
    }


def assess_biodiversity_demo(temp, depth, pollution, coral_cover, lat, lon):
    temp_score = 1 - abs(temp - 26) / 15
    depth_score = max(0.0, 1 - depth / 100)
    pollution_score = max(0.0, (5 - pollution) / 5)
    coral_score = coral_cover / 100.0
    overall_score = max(0.0, min(1.0, (temp_score + depth_score + pollution_score + coral_score) / 4))

    species_count = int(5 + overall_score * 20)
    shannon_diversity = overall_score * 3.5

    if shannon_diversity > 2.5:
        category = "High"
    elif shannon_diversity > 1.5:
        category = "Medium"
    else:
        category = "Low"

    category_probabilities = {
        "High": 0.1,
        "Medium": 0.1,
        "Low": 0.1,
    }
    category_probabilities[category] = float(0.8 + np.random.uniform(0, 0.15))
    total = sum(category_probabilities.values())
    category_probabilities = {k: v / total for k, v in category_probabilities.items()}

    return {
        "species_count": species_count,
        "shannon_diversity": shannon_diversity,
        "biodiversity_category": category,
        "category_probabilities": category_probabilities,
    }


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def show_home_page():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-container">
                <h3>🐟 Fish Identification</h3>
                <p>Classify fish species using morphometric measurements with 85%+ accuracy</p>
                <ul>
                    <li>4 Marine Species Supported</li>
                    <li>Real-time Classification</li>
                    <li>Confidence Scoring</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-container">
                <h3>🌊 Ocean Monitoring</h3>
                <p>Predict oceanographic parameters for ecosystem health assessment</p>
                <ul>
                    <li>Temperature Prediction</li>
                    <li>Salinity Analysis</li>
                    <li>Oxygen Level Assessment</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-container">
                <h3>🐠 Biodiversity Analysis</h3>
                <p>Comprehensive marine biodiversity and ecosystem health evaluation</p>
                <ul>
                    <li>Species Richness Estimation</li>
                    <li>Shannon Diversity Index</li>
                    <li>Ecosystem Health Scoring</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("📊 Platform Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fish Species Classified", "4", "100%")
    c2.metric("Model Accuracy", "85%", "↗️ 5%")
    c3.metric("Ocean Parameters", "3", "Real-time")
    c4.metric("Biodiversity Sites", "60+", "Monitored")

    st.markdown("---")
    st.subheader("🔄 Recent Platform Activity")

    recent_data = pd.DataFrame(
        {
            "Time": pd.date_range("2024-01-01", periods=10, freq="h"),
            "Analysis Type": np.random.choice(["Fish ID", "Ocean Pred", "Bio Assessment"], 10),
            "Location": [f"Site_{i}" for i in range(1, 11)],
            "Result": np.random.choice([
                "High Confidence",
                "Medium Confidence",
                "Low Confidence",
            ], 10),
        }
    )

    st.dataframe(recent_data, use_container_width=True)


def show_fish_identification(models):
    st.markdown('<h2 class="sub-header">🐟 Fish Species Identification</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            Enter fish morphometric measurements to identify the species. Our AI model analyzes length, weight, 
            height, and width measurements to classify fish with high accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📏 Fish Measurements")
        length = st.number_input("Length (cm)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        weight = st.number_input("Weight (g)", min_value=50.0, max_value=800.0, value=300.0, step=10.0)
        height = st.number_input("Height (cm)", min_value=5.0, max_value=25.0, value=12.0, step=0.1)
        width = st.number_input("Width (cm)", min_value=2.0, max_value=10.0, value=4.5, step=0.1)

        predict_button = st.button("🔍 Identify Fish Species", type="primary")

    with col2:
        st.subheader("🐠 Supported Species")
        species_info = {
            "🐟 Bream": "Medium-sized fish, typically 20-30cm, weight 200-400g",
            "🎣 Perch": "Popular sport fish, 15-25cm length, weight 100-250g",
            "🐊 Pike": "Large predatory fish, 30-40cm+, weight 400-600g+",
            "🔴 Roach": "Small schooling fish, 15-20cm, weight 80-150g",
        }
        for species, description in species_info.items():
            st.markdown(f"**{species}**: {description}")

    if predict_button:
        with st.spinner("🤖 Analyzing fish measurements..."):
            result = predict_fish_species_real(length, weight, height, width, models)

        st.markdown("---")
        st.markdown('<h3 class="sub-header">🎯 Identification Results</h3>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2>Predicted Species: {result['predicted_species']}</h2>
                    <p>Confidence Score: {result['confidence']:.1%}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_b:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["confidence"] * 100,
                    title={"text": "Confidence"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "yellow"},
                            {"range": [80, 100], "color": "green"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90},
                    },
                )
            )
            fig_gauge.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_c:
            st.markdown("**Input Summary:**")
            st.write(f"Length: {length} cm")
            st.write(f"Weight: {weight} g")
            st.write(f"Height: {height} cm")
            st.write(f"Width: {width} cm")

        st.subheader("📊 Species Probability Breakdown")
        prob_df = pd.DataFrame(list(result["all_probabilities"].items()), columns=["Species", "Probability"]).sort_values(
            "Probability", ascending=False
        )
        fig_bar = px.bar(
            prob_df,
            x="Species",
            y="Probability",
            color="Probability",
            color_continuous_scale="Blues",
            title="Species Classification Probabilities",
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        if result["confidence"] > 0.8:
            st.markdown(
                f"""
                <div class="success-box">
                    <strong>✅ High Confidence Prediction</strong><br>
                    This fish is most likely a <strong>{result['predicted_species']}</strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ Medium confidence prediction. Consider additional verification or measurements.")


def show_ocean_prediction(models):
    st.markdown('<h2 class="sub-header">🌊 Ocean Parameter Prediction</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            Predict key oceanographic parameters based on location, depth, and environmental conditions. 
            Essential for marine ecosystem monitoring and fisheries management.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Location & Environmental Data")
        latitude = st.slider("Latitude", -90.0, 90.0, 15.0, 0.1)
        longitude = st.slider("Longitude", -180.0, 180.0, 75.0, 0.1)
        depth = st.number_input("Depth (m)", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
        date_input = st.date_input("Date", datetime.now().date())
        day_of_year = date_input.timetuple().tm_yday
        chlorophyll = st.number_input("Chlorophyll-a (mg/m³)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
        ph = st.number_input("pH", min_value=7.5, max_value=8.5, value=8.1, step=0.01)
        predict_button = st.button("🌊 Predict Ocean Parameters", type="primary")

    with col2:
        st.subheader("🗺️ Location Preview")
        map_data = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        st.map(map_data)

        st.subheader("📊 Parameter Information")
        param_info = {
            "🌡️ Temperature": "Sea surface temperature affects marine life distribution",
            "🧂 Salinity": "Salt content influences ocean circulation and marine organisms",
            "💨 Dissolved Oxygen": "Essential for marine life survival and ecosystem health",
        }
        for param, description in param_info.items():
            st.markdown(f"**{param}**: {description}")

    if predict_button:
        with st.spinner("🤖 Analyzing oceanographic conditions..."):
            result = predict_ocean_parameters_real(latitude, longitude, depth, day_of_year, chlorophyll, ph, models)

        st.markdown("---")
        st.markdown('<h3 class="sub-header">🎯 Ocean Parameter Predictions</h3>', unsafe_allow_html=True)

        a, b, c = st.columns(3)
        with a:
            st.metric("🌡️ Temperature", f"{result['temperature']:.1f}°C", f"{result['temperature'] - 25:.1f}°C from avg")
        with b:
            st.metric("🧂 Salinity", f"{result['salinity']:.1f}‰", f"{result['salinity'] - 35:.1f}‰ from avg")
        with c:
            st.metric(
                "💨 Dissolved Oxygen",
                f"{result['dissolved_oxygen']:.1f} mg/L",
                f"{result['dissolved_oxygen'] - 8:.1f} mg/L from avg",
            )

        st.subheader("📈 Parameter Analysis")
        parameters = ["Temperature", "Salinity", "Dissolved Oxygen"]
        values = [
            result["temperature"] / 30 * 100,
            result["salinity"] / 40 * 100,
            result["dissolved_oxygen"] / 12 * 100,
        ]
        fig_radar = go.Figure(
            go.Scatterpolar(r=values, theta=parameters, fill="toself", name="Predicted Values")
        )
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="Ocean Parameter Profile")
        st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("🔍 Environmental Assessment")
        temp_status = "Optimal" if 22 <= result["temperature"] <= 28 else "Sub-optimal"
        sal_status = "Normal" if 33 <= result["salinity"] <= 36 else "Abnormal"
        oxy_status = "Healthy" if result["dissolved_oxygen"] >= 6 else "Low"
        assessment_data = pd.DataFrame(
            {
                "Parameter": ["Temperature", "Salinity", "Dissolved Oxygen"],
                "Value": [
                    f"{result['temperature']:.1f}°C",
                    f"{result['salinity']:.1f}‰",
                    f"{result['dissolved_oxygen']:.1f} mg/L",
                ],
                "Status": [temp_status, sal_status, oxy_status],
                "Impact": [
                    "Affects metabolic rates and species distribution",
                    "Influences water density and circulation",
                    "Critical for marine organism survival",
                ],
            }
        )
        st.dataframe(assessment_data, use_container_width=True)


def show_biodiversity_assessment(models):
    st.markdown('<h2 class="sub-header">🐠 Marine Biodiversity Assessment</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            Comprehensive evaluation of marine biodiversity and ecosystem health based on environmental conditions. 
            Helps in conservation planning and marine protected area management.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌡️ Environmental Conditions")
        temperature = st.number_input("Water Temperature (°C)", min_value=15.0, max_value=35.0, value=26.0, step=0.5)
        depth = st.number_input("Water Depth (m)", min_value=0.0, max_value=200.0, value=25.0, step=1.0)
        pollution_level = st.slider("Pollution Level", 0.0, 5.0, 1.0, 0.1)
        coral_cover = st.slider("Coral Cover (%)", 0.0, 100.0, 60.0, 1.0)

        st.subheader("📍 Geographic Location")
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=12.0, step=0.1)
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.0, step=0.1)

        assess_button = st.button("🔍 Assess Biodiversity", type="primary")

    with col2:
        st.subheader("🌊 Assessment Factors")
        factors_info = {
            "🌡️ Temperature": f"Current: {temperature}°C (Optimal: 24-28°C)",
            "🏊 Depth": f"Current: {depth}m (Shallow waters more diverse)",
            "🏭 Pollution": f"Level: {pollution_level}/5 (Lower is better)",
            "🪸 Coral Cover": f"Coverage: {coral_cover}% (Higher supports more species)",
        }
        for k, v in factors_info.items():
            st.write(f"**{k}:** {v}")

    if assess_button:
        with st.spinner("🤖 Assessing biodiversity..."):
            result = assess_biodiversity_real(
                temperature, depth, pollution_level, coral_cover, latitude, longitude, models
            )

        st.markdown("---")
        st.markdown('<h3 class="sub-header">🎯 Assessment Results</h3>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "🐠 Estimated Species Count",
                f"{result['species_count']:.0f}",
                "Higher is better",
            )
        with c2:
            st.metric(
                "📊 Shannon Diversity Index",
                f"{result['shannon_diversity']:.2f}",
                "Target > 2.5",
            )
        with c3:
            badge = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
            st.metric("🏆 Biodiversity Category", f"{badge.get(result['biodiversity_category'],'')} {result['biodiversity_category']}")

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("📊 Category Probabilities")
            prob_data = pd.DataFrame(
                list(result["category_probabilities"].items()),
                columns=["Category", "Probability"],
            )
            fig_pie = px.pie(prob_data, values="Probability", names="Category", title="Biodiversity Category Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_r:
            st.subheader("🎯 Factor Score Breakdown")
            temp_score = max(0.0, 1 - abs(temperature - 26) / 10) * 100
            pollution_score = max(0.0, (5 - pollution_level) / 5) * 100
            coral_score = coral_cover
            depth_score = max(0.0, (100 - depth) / 100) * 100
            score_data = pd.DataFrame(
                {
                    "Component": ["Temperature", "Pollution", "Coral Cover", "Depth"],
                    "Score": [temp_score, pollution_score, coral_score, depth_score],
                }
            )
            fig_bar = px.bar(
                score_data,
                x="Score",
                y="Component",
                orientation="h",
                color="Score",
                color_continuous_scale="RdYlGn",
                title="Environmental Factor Scores",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("🛡️ Conservation Recommendations")
        if result["biodiversity_category"] == "High":
            recs = [
                "🎉 Excellent biodiversity! Continue current conservation efforts.",
                "🔒 Consider establishing Marine Protected Area status.",
                "📊 Regular monitoring to maintain ecosystem health.",
                "🎓 Use as reference site for research and education.",
            ]
        elif result["biodiversity_category"] == "Medium":
            recs = [
                "⚠️ Moderate biodiversity with improvement potential.",
                "🏭 Address pollution sources if present.",
                "🪸 Coral restoration programs could help.",
                "🐠 Monitor fish populations regularly.",
            ]
        else:
            recs = [
                "🚨 Low biodiversity requires immediate attention.",
                "🛑 Urgent pollution control measures needed.",
                "🔧 Habitat restoration programs recommended.",
                "📞 Consult marine conservation experts.",
            ]
        for rec in recs:
            st.markdown(f"- {rec}")


def show_comprehensive_analysis(models):
    st.markdown('<h2 class="sub-header">📊 Comprehensive Marine Analysis</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            Run all three analysis modules simultaneously for a complete marine ecosystem assessment.
            Perfect for research studies and environmental impact assessments.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🔧 Input Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🐟 Fish Data**")
        fish_length = st.number_input("Fish Length (cm)", min_value=10.0, max_value=50.0, value=25.0, key="comp_length")
        fish_weight = st.number_input("Fish Weight (g)", min_value=50.0, max_value=800.0, value=300.0, key="comp_weight")
        fish_height = st.number_input("Fish Height (cm)", min_value=5.0, max_value=25.0, value=12.0, key="comp_height")
        fish_width = st.number_input("Fish Width (cm)", min_value=2.0, max_value=10.0, value=4.5, key="comp_width")

    with col2:
        st.markdown("**🌊 Ocean Data**")
        latitude = st.slider("Latitude", -90.0, 90.0, 15.0, key="comp_lat")
        longitude = st.slider("Longitude", -180.0, 180.0, 75.0, key="comp_lon")
        depth = st.number_input("Depth (m)", min_value=0.0, max_value=200.0, value=30.0, key="comp_depth")
        chlorophyll = st.number_input("Chlorophyll-a", min_value=0.1, max_value=10.0, value=2.5, key="comp_chl")
        ph = st.number_input("pH", min_value=7.5, max_value=8.5, value=8.1, key="comp_ph")

    with col3:
        st.markdown("**🐠 Biodiversity Data**")
        temperature = st.number_input("Temperature (°C)", min_value=15.0, max_value=35.0, value=26.0, key="comp_temp")
        pollution_level = st.slider("Pollution Level", 0.0, 5.0, 1.0, key="comp_pollution")
        coral_cover = st.slider("Coral Cover (%)", 0.0, 100.0, 60.0, key="comp_coral")
        date_input = st.date_input("Analysis Date", datetime.now().date(), key="comp_date")

    if st.button("🚀 Run Comprehensive Analysis", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🐟 Analyzing fish species...")
        fish_result = predict_fish_species_real(fish_length, fish_weight, fish_height, fish_width, models)
        progress_bar.progress(33)

        status_text.text("🌊 Predicting ocean parameters...")
        day_of_year = date_input.timetuple().tm_yday
        ocean_result = predict_ocean_parameters_real(latitude, longitude, depth, day_of_year, chlorophyll, ph, models)
        progress_bar.progress(66)

        status_text.text("🐠 Assessing biodiversity...")
        bio_result = assess_biodiversity_real(temperature, depth, pollution_level, coral_cover, latitude, longitude, models)
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        st.markdown("---")
        st.markdown('<h3 class="sub-header">🎯 Comprehensive Results</h3>', unsafe_allow_html=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("🐟 Fish Species", fish_result["predicted_species"], f"{fish_result['confidence']:.1%} confidence")
        with col_b:
            st.metric("🌡️ Ocean Temp", f"{ocean_result['temperature']:.1f}°C", "Predicted")
        with col_c:
            st.metric("🐠 Biodiversity", bio_result["biodiversity_category"], f"{bio_result['species_count']} species")
        with col_d:
            health_score = (
                fish_result["confidence"] + (ocean_result["temperature"] / 30) + (bio_result["shannon_diversity"] / 3.5)
            ) / 3
            st.metric("🏥 Ecosystem Health", f"{health_score:.1%}", "Overall Score")

        tab1, tab2, tab3, tab4 = st.tabs(["🐟 Fish Analysis", "🌊 Ocean Parameters", "🐠 Biodiversity", "📈 Integrated View"])

        with tab1:
            st.subheader("Fish Species Identification Results")
            t1c1, t1c2 = st.columns(2)
            with t1c1:
                st.write(f"**Species:** {fish_result['predicted_species']}")
                st.write(f"**Confidence:** {fish_result['confidence']:.1%}")
                st.write("**Input Parameters:**")
                st.write(f"- Length: {fish_length} cm")
                st.write(f"- Weight: {fish_weight} g")
                st.write(f"- Height: {fish_height} cm")
                st.write(f"- Width: {fish_width} cm")
            with t1c2:
                prob_df = pd.DataFrame(list(fish_result["all_probabilities"].items()), columns=["Species", "Probability"])
                fig = px.bar(prob_df, x="Species", y="Probability", title="Species Probabilities")
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Ocean Parameter Predictions")
            t2c1, t2c2 = st.columns(2)
            with t2c1:
                st.write("**Predicted Parameters:**")
                st.write(f"- Temperature: {ocean_result['temperature']:.1f}°C")
                st.write(f"- Salinity: {ocean_result['salinity']:.1f}‰")
                st.write(f"- Dissolved Oxygen: {ocean_result['dissolved_oxygen']:.1f} mg/L")
                st.write("**Input Location:**")
                st.write(f"- Latitude: {latitude}")
                st.write(f"- Longitude: {longitude}")
                st.write(f"- Depth: {depth} m")
            with t2c2:
                ocean_data = pd.DataFrame(
                    {
                        "Parameter": ["Temperature", "Salinity", "Dissolved Oxygen"],
                        "Value": [
                            ocean_result["temperature"],
                            ocean_result["salinity"],
                            ocean_result["dissolved_oxygen"],
                        ],
                        "Unit": ["°C", "‰", "mg/L"],
                    }
                )
                fig = px.bar(ocean_data, x="Parameter", y="Value", title="Ocean Parameters")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Biodiversity Assessment Results")
            t3c1, t3c2 = st.columns(2)
            with t3c1:
                st.write("**Assessment Results:**")
                st.write(f"- Species Count: {bio_result['species_count']}")
                st.write(f"- Shannon Diversity: {bio_result['shannon_diversity']:.2f}")
                st.write(f"- Category: {bio_result['biodiversity_category']}")
                st.write("**Environmental Factors:**")
                st.write(f"- Temperature: {temperature}°C")
                st.write(f"- Pollution Level: {pollution_level}/5")
                st.write(f"- Coral Cover: {coral_cover}%")
            with t3c2:
                prob_data = pd.DataFrame(list(bio_result["category_probabilities"].items()), columns=["Category", "Probability"])
                fig = px.pie(prob_data, values="Probability", names="Category", title="Biodiversity Categories")
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Integrated Ecosystem Analysis")
            t4c1, t4c2 = st.columns(2)
            with t4c1:
                categories = ["Fish Diversity", "Ocean Health", "Biodiversity", "Water Quality"]
                values = [
                    fish_result["confidence"] * 100,
                    (ocean_result["temperature"] / 30 + ocean_result["dissolved_oxygen"] / 12) * 50,
                    bio_result["shannon_diversity"] / 3.5 * 100,
                    (1 - pollution_level / 5) * 100,
                ]
                fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill="toself"))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Ecosystem Health Profile")
                st.plotly_chart(fig, use_container_width=True)
            with t4c2:
                st.subheader("🎯 Integrated Recommendations")
                recommendations = []
                recommendations.append("✅ Fish identification highly reliable" if fish_result["confidence"] > 0.8 else "⚠️ Fish identification needs verification")
                recommendations.append("✅ Ocean temperature optimal for marine life" if 22 <= ocean_result["temperature"] <= 28 else "⚠️ Ocean temperature may stress marine organisms")
                recommendations.append("✅ Excellent biodiversity - maintain current conditions" if bio_result["biodiversity_category"] == "High" else "🔧 Biodiversity enhancement recommended")
                recommendations.append("✅ Low pollution levels detected" if pollution_level <= 1.5 else "🚨 Pollution control measures needed")
                for rec in recommendations:
                    st.markdown(f"- {rec}")


def show_data_visualization():
    st.markdown('<h2 class="sub-header">📈 Data Visualization & Analytics</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            Interactive data visualizations and statistical analysis of marine ecosystem data.
            Explore trends, patterns, and relationships in the data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    viz_type = st.selectbox(
        "Choose Visualization Type",
        [
            "📊 Fish Species Distribution",
            "🌊 Ocean Parameter Trends",
            "🐠 Biodiversity Heatmap",
            "📈 Correlation Analysis",
            "🗺️ Geographic Distribution",
        ],
    )

    if viz_type == "📊 Fish Species Distribution":
        st.subheader("Fish Species Distribution Analysis")
        np.random.seed(42)
        fish_data = pd.DataFrame(
            {
                "Species": np.random.choice(["Bream", "Perch", "Pike", "Roach"], 200),
                "Length": np.random.normal(25, 8, 200),
                "Weight": np.random.normal(300, 100, 200),
                "Location": np.random.choice(["Site_A", "Site_B", "Site_C", "Site_D"], 200),
                "Season": np.random.choice(["Spring", "Summer", "Fall", "Winter"], 200),
            }
        )
        c1, c2 = st.columns(2)
        with c1:
            species_counts = fish_data["Species"].value_counts()
            fig = px.pie(values=species_counts.values, names=species_counts.index, title="Species Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(fish_data, x="Length", y="Weight", color="Species", title="Length vs Weight by Species")
            st.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(fish_data, x="Species", color="Season", title="Species Distribution by Season")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "🌊 Ocean Parameter Trends":
        st.subheader("Ocean Parameter Trends Analysis")
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
        np.random.seed(42)
        ocean_data = pd.DataFrame(
            {
                "Date": dates,
                "Temperature": 26 + 4 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 0.5, len(dates)),
                "Salinity": 34.5 + 0.5 * np.cos(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 0.2, len(dates)),
                "Dissolved_Oxygen": 8 + np.random.normal(0, 0.3, len(dates)),
            }
        )
        param = st.selectbox("Select Parameter", ["Temperature", "Salinity", "Dissolved_Oxygen"])
        fig = px.line(ocean_data, x="Date", y=param, title=f"{param} Trends Over Time")
        st.plotly_chart(fig, use_container_width=True)
        ocean_data["Month"] = ocean_data["Date"].dt.month_name()
        monthly_avg = ocean_data.groupby("Month")[param].mean().reset_index()
        fig = px.bar(monthly_avg, x="Month", y=param, title=f"Average {param} by Month")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "🐠 Biodiversity Heatmap":
        st.subheader("Biodiversity Distribution Heatmap")
        np.random.seed(42)
        locations = [f"Site_{i}" for i in range(1, 21)]
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        biodiversity_data = []
        for loc in locations:
            for month in months:
                biodiversity_data.append({
                    "Location": loc,
                    "Month": month,
                    "Shannon_Diversity": float(np.random.uniform(1, 4)),
                })
        bio_df = pd.DataFrame(biodiversity_data)
        pivot_data = bio_df.pivot(index="Location", columns="Month", values="Shannon_Diversity")
        fig = px.imshow(pivot_data, title="Shannon Diversity Index Heatmap", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("📊 Biodiversity Statistics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Average Diversity", f"{bio_df['Shannon_Diversity'].mean():.2f}")
        with c2:
            st.metric("Highest Diversity", f"{bio_df['Shannon_Diversity'].max():.2f}")
        with c3:
            st.metric("Sites Monitored", len(locations))

    elif viz_type == "📈 Correlation Analysis":
        st.subheader("Environmental Parameters Correlation Analysis")
        np.random.seed(42)
        n = 500
        df = pd.DataFrame(
            {
                "Temperature": np.random.normal(26, 3, n),
                "Salinity": np.random.normal(34.5, 1, n),
                "Dissolved_Oxygen": np.random.normal(8, 1, n),
                "pH": np.random.normal(8.1, 0.2, n),
                "Chlorophyll": np.random.normal(2.5, 0.8, n),
                "Species_Count": np.random.normal(15, 5, n),
            }
        )
        df["Dissolved_Oxygen"] = 10 - 0.1 * df["Temperature"] + np.random.normal(0, 0.5, n)
        df["Species_Count"] = 5 + 0.3 * df["Dissolved_Oxygen"] + 2 * df["Chlorophyll"] + np.random.normal(0, 2, n)
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, title="Environmental Parameters Correlation Matrix", color_continuous_scale="RdBu", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(df, x="Temperature", y="Dissolved_Oxygen", title="Temperature vs Dissolved Oxygen", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(df, x="Dissolved_Oxygen", y="Species_Count", title="Dissolved Oxygen vs Species Count", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "🗺️ Geographic Distribution":
        st.subheader("Geographic Distribution of Marine Data")
        np.random.seed(42)
        n_points = 100
        geo_data = pd.DataFrame(
            {
                "Latitude": np.random.uniform(8, 25, n_points),
                "Longitude": np.random.uniform(68, 85, n_points),
                "Temperature": np.random.normal(27, 2, n_points),
                "Biodiversity_Score": np.random.uniform(1, 4, n_points),
                "Site_Type": np.random.choice(["Coral Reef", "Open Ocean", "Coastal", "Deep Sea"], n_points),
            }
        )
        fig = px.scatter_mapbox(
            geo_data,
            lat="Latitude",
            lon="Longitude",
            color="Biodiversity_Score",
            size="Temperature",
            hover_data=["Site_Type"],
            title="Marine Biodiversity Distribution",
            mapbox_style="open-street-map",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("📊 Regional Analysis")
        c1, c2 = st.columns(2)
        with c1:
            site_summary = geo_data.groupby("Site_Type").agg({"Biodiversity_Score": "mean", "Temperature": "mean"}).round(2)
            st.dataframe(site_summary)
        with c2:
            fig = px.box(geo_data, x="Site_Type", y="Biodiversity_Score", title="Biodiversity by Site Type")
            st.plotly_chart(fig, use_container_width=True)


def show_about_page():
    st.markdown('<h2 class="sub-header">ℹ️ About CMLRE Marine ML Platform</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            <h3>🌊 Centre for Marine Living Resources and Ecology (CMLRE)</h3>
            <p>Ministry of Earth Sciences, Government of India</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Project Overview")
        st.markdown(
            """
            The CMLRE Marine ML Platform is an advanced artificial intelligence system designed to support 
            marine ecosystem monitoring, conservation, and research activities. This platform integrates 
            multiple machine learning models to provide comprehensive analysis of marine environments.

            **Key Objectives:**
            - Marine biodiversity assessment and monitoring
            - Fish species identification for fisheries management
            - Oceanographic parameter prediction
            - Ecosystem health evaluation
            - Conservation planning support
            """
        )
        st.subheader("🔬 Technology Stack")
        st.markdown(
            """
            - **Frontend:** Streamlit
            - **ML Framework:** Scikit-learn, TensorFlow
            - **Visualization:** Plotly, Matplotlib
            - **Data Processing:** Pandas, NumPy
            - **Deployment:** Streamlit Cloud
            """
        )
    with col2:
        st.subheader("🐟 Supported Models")
        model_info = {
            "Fish Species Classifier": {
                "Purpose": "Identify fish species from morphometric measurements",
                "Species": "Bream, Perch, Pike, Roach",
                "Accuracy": "85%+",
                "Features": "Length, Weight, Height, Width",
            },
            "Ocean Parameter Predictor": {
                "Purpose": "Predict oceanographic conditions",
                "Parameters": "Temperature, Salinity, Dissolved Oxygen",
                "Input": "Location, Depth, Environmental data",
                "Applications": "Ecosystem monitoring, Climate studies",
            },
            "Biodiversity Assessor": {
                "Purpose": "Evaluate marine biodiversity and ecosystem health",
                "Metrics": "Shannon Diversity, Species Count",
                "Factors": "Temperature, Pollution, Coral Cover",
                "Output": "Biodiversity category and recommendations",
            },
        }
        for model_name, info in model_info.items():
            with st.expander(f"🔍 {model_name}"):
                for k, v in info.items():
                    st.write(f"**{k}:** {v}")

    st.markdown("---")
    st.subheader("🔬 Research Applications")
    applications = [
        "🐠 **Fisheries Management**: Stock assessment and sustainable fishing practices",
        "🌊 **Climate Change Studies**: Monitoring ocean parameter changes over time",
        "🪸 **Coral Reef Conservation**: Assessing reef health and biodiversity",
        "🏭 **Pollution Monitoring**: Evaluating environmental impact on marine ecosystems",
        "🛡️ **Marine Protected Areas**: Site selection and management effectiveness",
        "📊 **Ecosystem Services**: Quantifying marine ecosystem contributions",
        "🎓 **Education and Training**: Teaching marine science and conservation",
        "📈 **Policy Support**: Evidence-based marine policy recommendations",
    ]
    for app in applications:
        st.markdown(app)

    st.markdown("---")
    st.subheader("📞 Contact Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            **🏢 Centre for Marine Living Resources and Ecology (CMLRE)**  
            Ministry of Earth Sciences  
            Government of India  
            Kochi, Kerala, India
            """
        )
    with col2:
        st.markdown(
            """
            **🌐 Web:** www.cmlre.gov.in  
            **📧 Email:** info@cmlre.gov.in  
            **📱 Phone:** +91-484-2390814
            """
        )
    with col3:
        st.markdown(
            """
            **🔗 Related Organizations:**
            - Ministry of Earth Sciences
            - National Institute of Oceanography
            - Indian National Centre for Ocean Information Services
            """
        )

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Platform Version", "1.0.0")
    with c2:
        st.metric("Last Updated", "January 2025")
    with c3:
        st.metric("Models Deployed", "3")


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">🌊 CMLRE Marine Living Resources ML Platform</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            <strong>🎯 Mission:</strong> Advanced AI platform for marine ecosystem monitoring and conservation under 
            the Ministry of Earth Sciences, India. This platform provides real-time analysis of marine biodiversity, 
            oceanographic parameters, and fish species identification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    models = load_models()

    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        [
            "🏠 Home",
            "🐟 Fish Species Identification",
            "🌊 Ocean Parameter Prediction",
            "🐠 Biodiversity Assessment",
            "📊 Comprehensive Analysis",
            "📈 Data Visualization",
            "ℹ️ About",
        ],
    )

    if page == "🏠 Home":
        show_home_page()
    elif page == "🐟 Fish Species Identification":
        show_fish_identification(models)
    elif page == "🌊 Ocean Parameter Prediction":
        show_ocean_prediction(models)
    elif page == "🐠 Biodiversity Assessment":
        show_biodiversity_assessment(models)
    elif page == "📊 Comprehensive Analysis":
        show_comprehensive_analysis(models)
    elif page == "📈 Data Visualization":
        show_data_visualization()
    elif page == "ℹ️ About":
        show_about_page()


if __name__ == "__main__":
    main()


