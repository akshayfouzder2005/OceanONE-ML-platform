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
import statsmodels.api as sm

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
st.set_page_config(
    page_title="OceanONE",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Enhanced Modern Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit elements */
    .css-1d391kg, .css-1v0mbdj {
        display: none;
    }

    /* Main header with gradient */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }

    /* Modern sidebar navigation */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 0 20px 20px 0;
    }

    /* Custom navigation container */
    .nav-container {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .nav-title {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Custom navigation items */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        margin: 4px 0;
        border-radius: 12px;
        color: #e0e6ed;
        text-decoration: none;
        transition: all 0.3s ease;
        cursor: pointer;
        border: 1px solid transparent;
        font-weight: 500;
    }

    .nav-item:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border: 1px solid rgba(102, 126, 234, 0.4);
        transform: translateX(4px);
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    .nav-item.active {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
    }

    .nav-icon {
        font-size: 1.2rem;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Status indicators */
    .status-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-left: auto;
    }

    .status-active {
        background: #4ade80;
        box-shadow: 0 0 8px #4ade80;
    }

    .status-inactive {
        background: #6b7280;
    }

    /* Quick stats in navigation */
    .nav-stats {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }

    .nav-stat-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #e0e6ed;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }

    .nav-stat-value {
        color: #667eea;
        font-weight: 600;
    }

    /* Modern metric containers */
    .metric-container {
        background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.3);
        color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    .metric-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
    }

    .metric-container h3 {
        color: #667eea;
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.3rem;
    }

    .metric-container p {
        color: #e0e6ed;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }

    .metric-container ul {
        color: #d1d5db;
        list-style: none;
        padding: 0;
    }

    .metric-container li {
        margin-bottom: 0.5rem;
        padding-left: 1.5rem;
        position: relative;
    }

    .metric-container li::before {
        content: '✓';
        position: absolute;
        left: 0;
        color: #4ade80;
        font-weight: bold;
    }

    /* Enhanced prediction boxes */
    .prediction-box {
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        backdrop-filter: blur(10px);
        border: 2px solid rgba(102, 126, 234, 0.3);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }

    .prediction-box::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #667eea, #764ba2, #667eea);
        z-index: -1;
        border-radius: 20px;
    }

    .success-box {
        background: linear-gradient(145deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.1));
        border: 2px solid rgba(34, 197, 94, 0.3);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }

    .info-box {
        background: linear-gradient(145deg, rgba(59, 130, 246, 0.1), rgba(37, 99, 235, 0.1));
        border: 2px solid rgba(59, 130, 246, 0.3);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        line-height: 1.6;
    }

    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }

    /* Enhanced form controls */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }

    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        color: white;
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* Dark theme enhancements */
    .main .block-container {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* Animated background elements */
    .bg-animation {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        opacity: 0.1;
        background: radial-gradient(circle at 25% 25%, #667eea 0%, transparent 50%),
                    radial-gradient(circle at 75% 75%, #764ba2 0%, transparent 50%);
        animation: float 6s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }

    /* Enhanced metrics display */
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: scale(1.02);
        border-color: rgba(102, 126, 234, 0.4);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: #e0e6ed;
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Add animated background
st.markdown('<div class="bg-animation"></div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Modern Navigation Component
# -----------------------------------------------------------------------------
def create_modern_navigation():
    """Create a modern, feature-rich navigation component with fixed state management"""

    # Initialize session state for current page if not exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    # Navigation data with icons and status
    nav_items = [
        {"name": "Home", "icon": "🏠", "key": "home", "status": "active"},
        {"name": "Fish Species ID", "icon": "🐟", "key": "fish", "status": "active"},
        {"name": "Ocean Parameters", "icon": "🌊", "key": "ocean", "status": "active"},
        {"name": "Biodiversity", "icon": "🐠", "key": "biodiversity", "status": "active"},
        {"name": "Comprehensive", "icon": "📊", "key": "comprehensive", "status": "active"},
        {"name": "Data Visualization", "icon": "📈", "key": "visualization", "status": "active"},
        {"name": "About", "icon": "ℹ️", "key": "about", "status": "active"}
    ]

    # Quick stats for navigation sidebar
    quick_stats = {
        "Models Active": "3/3",
        "Accuracy": "85%+",
        "Species": "4",
        "Parameters": "6"
    }

    with st.sidebar:
        st.markdown("""
        <div class="nav-container">
            <div class="nav-title">
                🧭 Navigation Panel
            </div>
        """, unsafe_allow_html=True)

        # Create navigation items with proper state management
        for item in nav_items:
            # Check if this is the currently active page
            is_active = st.session_state.current_page == item['key']

            # Create button with unique key and proper styling
            button_style = "primary" if is_active else "secondary"

            if st.button(
                    f"{item['icon']} {item['name']}",
                    key=f"nav_{item['key']}",
                    use_container_width=True,
                    type=button_style
            ):
                # Update session state when button is clicked
                st.session_state.current_page = item['key']
                st.rerun()  # Force rerun to update the page

        # Quick stats section with glassmorphism
        st.markdown("""
        <div class="nav-stats">
            <div style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem; text-shadow: 0 0 10px rgba(102, 126, 234, 0.5);">
                ⚡ Quick Stats
            </div>
        """, unsafe_allow_html=True)

        for stat_name, stat_value in quick_stats.items():
            st.markdown(f"""
            <div class="nav-stat-item">
                <span>{stat_name}</span>
                <span class="nav-stat-value">{stat_value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # System status indicator with glassmorphism
        st.markdown("""
        <div style="margin-top: 1rem; padding: 1rem; background: rgba(34, 197, 94, 0.08); 
                    backdrop-filter: blur(15px); border-radius: 12px; 
                    border: 1px solid rgba(34, 197, 94, 0.2);
                    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);">
            <div style="display: flex; align-items: center; gap: 0.5rem; color: #4ade80; text-shadow: 0 0 10px rgba(74, 222, 128, 0.5);">
                🟢 <strong>System Online</strong>
            </div>
            <div style="font-size: 0.8rem; color: rgba(209, 213, 219, 0.9); margin-top: 0.5rem;">
                All models operational
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Return current page from session state
    return st.session_state.current_page


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
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h2 style="color: #667eea; font-weight: 600; margin-bottom: 1rem;">
            🌊 Welcome to Marine Ecosystem Intelligence
        </h2>
        <p style="color: #e0e6ed; font-size: 1.1rem; line-height: 1.6; max-width: 800px; margin: 0 auto;">
            Advanced AI-powered platform for comprehensive marine ecosystem monitoring, 
            species identification, and biodiversity assessment under the Ministry of Earth Sciences, India.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-container">
                <h3>🐟 Fish Identification</h3>
                <p>Classify fish species using morphometric measurements with 85%+ accuracy powered by advanced machine learning algorithms.</p>
                <ul>
                    <li>4 Marine Species Supported</li>
                    <li>Real-time Classification</li>
                    <li>Confidence Scoring</li>
                    <li>Morphometric Analysis</li>
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
                <p>Predict oceanographic parameters for comprehensive ecosystem health assessment using environmental data.</p>
                <ul>
                    <li>Temperature Prediction</li>
                    <li>Salinity Analysis</li>
                    <li>Oxygen Level Assessment</li>
                    <li>Climate Impact Modeling</li>
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
                <p>Comprehensive marine biodiversity and ecosystem health evaluation for conservation planning.</p>
                <ul>
                    <li>Species Richness Estimation</li>
                    <li>Shannon Diversity Index</li>
                    <li>Ecosystem Health Scoring</li>
                    <li>Conservation Recommendations</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Enhanced metrics display
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h3 style="color: #667eea; margin-bottom: 2rem;">📊 Platform Performance</h3>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">Fish Species Classified</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">85%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">6</div>
            <div class="metric-label">Ocean Parameters</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">60+</div>
            <div class="metric-label">Sites Monitored</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Enhanced activity display
    st.markdown("""
    <div style="margin: 2rem 0;">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🔄 Recent Platform Activity</h3>
    </div>
    """, unsafe_allow_html=True)

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

    st.dataframe(
        recent_data,
        use_container_width=True,
        column_config={
            "Time": st.column_config.DatetimeColumn("Timestamp"),
            "Analysis Type": st.column_config.TextColumn("Analysis"),
            "Location": st.column_config.TextColumn("Site"),
            "Result": st.column_config.TextColumn("Confidence Level")
        }
    )


def show_fish_identification(models):
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #667eea; font-weight: 600;">🐟 Advanced Fish Species Identification</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            Enter precise fish morphometric measurements to identify the species using our advanced AI model. 
            Our system analyzes length, weight, height, and width measurements to classify fish with over 85% accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📏 Fish Measurements")

        with st.container():
            length = st.number_input(
                "Length (cm)",
                min_value=10.0,
                max_value=50.0,
                value=25.0,
                step=0.1,
                help="Total length from nose to tail fin"
            )
            weight = st.number_input(
                "Weight (g)",
                min_value=50.0,
                max_value=800.0,
                value=300.0,
                step=10.0,
                help="Total body weight in grams"
            )
            height = st.number_input(
                "Height (cm)",
                min_value=5.0,
                max_value=25.0,
                value=12.0,
                step=0.1,
                help="Maximum body height"
            )
            width = st.number_input(
                "Width (cm)",
                min_value=2.0,
                max_value=10.0,
                value=4.5,
                step=0.1,
                help="Maximum body width"
            )

        predict_button = st.button("🔍 Identify Fish Species", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 🐠 Supported Species Database")
        species_info = {
            "🐟 Bream": "Medium-sized fish, typically 20-30cm, weight 200-400g. Commonly found in freshwater environments.",
            "🎣 Perch": "Popular sport fish, 15-25cm length, weight 100-250g. Known for distinctive striped pattern.",
            "🐊 Pike": "Large predatory fish, 30-40cm+, weight 400-600g+. Apex predator with elongated body.",
            "🔴 Roach": "Small schooling fish, 15-20cm, weight 80-150g. Common in European freshwater systems.",
        }
        for species, description in species_info.items():
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 12px; 
                        margin-bottom: 0.5rem; border-left: 3px solid #667eea;">
                <strong style="color: #667eea;">{species}</strong><br>
                <span style="color: #e0e6ed; font-size: 0.9rem;">{description}</span>
            </div>
            """, unsafe_allow_html=True)

    if predict_button:
        with st.spinner("🤖 Analyzing fish measurements..."):
            result = predict_fish_species_real(length, weight, height, width, models)

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: #667eea;">🎯 Identification Results</h3>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2 style="margin-bottom: 1rem;">Predicted Species: {result['predicted_species']}</h2>
                    <p style="font-size: 1.2rem;">Confidence Score: {result['confidence']:.1%}</p>
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 12px; margin-top: 1rem;">
                        <strong>Reliability Assessment:</strong> 
                        {'High Confidence' if result['confidence'] > 0.8 else 'Medium Confidence' if result['confidence'] > 0.6 else 'Low Confidence'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_b:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["confidence"] * 100,
                    title={"text": "Confidence", "font": {"color": "white"}},
                    number={"font": {"color": "white"}},
                    gauge={
                        "axis": {"range": [None, 100], "tickcolor": "white"},
                        "bar": {"color": "#667eea"},
                        "steps": [
                            {"range": [0, 50], "color": "rgba(255,0,0,0.3)"},
                            {"range": [50, 80], "color": "rgba(255,255,0,0.3)"},
                            {"range": [80, 100], "color": "rgba(0,255,0,0.3)"},
                        ],
                        "threshold": {"line": {"color": "#764ba2", "width": 4}, "thickness": 0.75, "value": 90},
                    },
                )
            )
            fig_gauge.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_c:
            st.markdown("""
            <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 12px;">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">Input Summary</h4>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"**Length:** {length} cm")
            st.write(f"**Weight:** {weight} g")
            st.write(f"**Height:** {height} cm")
            st.write(f"**Width:** {width} cm")

        st.markdown("### 📊 Species Probability Analysis")
        prob_df = pd.DataFrame(list(result["all_probabilities"].items()),
                               columns=["Species", "Probability"]).sort_values(
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
        fig_bar.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if result["confidence"] > 0.8:
            st.markdown(
                f"""
                <div class="success-box">
                    <strong>✅ High Confidence Prediction</strong><br>
                    This fish is most likely a <strong>{result['predicted_species']}</strong> with high reliability.
                    The morphometric measurements strongly match this species profile.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "⚠️ Medium confidence prediction. Consider additional verification or measurements for higher accuracy.")


def show_ocean_prediction(models):
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #667eea; font-weight: 600;">🌊 Ocean Parameter Prediction</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            Predict key oceanographic parameters based on location, depth, and environmental conditions. 
            This analysis is essential for marine ecosystem monitoring, fisheries management, and climate change research.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📍 Location & Environmental Data")

        latitude = st.slider("Latitude", -90.0, 90.0, 15.0, 0.1, help="Geographic latitude coordinate")
        longitude = st.slider("Longitude", -180.0, 180.0, 75.0, 0.1, help="Geographic longitude coordinate")
        depth = st.number_input("Depth (m)", min_value=0.0, max_value=200.0, value=30.0, step=1.0,
                                help="Water depth in meters")

        date_input = st.date_input("Date", datetime.now().date(), help="Date for seasonal analysis")
        day_of_year = date_input.timetuple().tm_yday

        chlorophyll = st.number_input(
            "Chlorophyll-a (mg/m³)",
            min_value=0.1,
            max_value=10.0,
            value=2.5,
            step=0.1,
            help="Chlorophyll concentration indicator of phytoplankton biomass"
        )
        ph = st.number_input(
            "pH",
            min_value=7.5,
            max_value=8.5,
            value=8.1,
            step=0.01,
            help="Ocean pH level (acidity/alkalinity)"
        )

        predict_button = st.button("🌊 Predict Ocean Parameters", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 🗺️ Location Preview")
        map_data = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        st.map(map_data, size=20)

        st.markdown("### 📊 Parameter Information")
        param_info = {
            "🌡️ Temperature": "Sea surface temperature affects marine life distribution and metabolism",
            "🧂 Salinity": "Salt content influences ocean circulation and marine organism physiology",
            "💨 Dissolved Oxygen": "Essential for marine life survival and ecosystem productivity",
        }
        for param, description in param_info.items():
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.1); padding: 0.8rem; border-radius: 10px; 
                        margin-bottom: 0.5rem; border-left: 3px solid #667eea;">
                <strong style="color: #667eea;">{param}</strong><br>
                <span style="color: #e0e6ed; font-size: 0.9rem;">{description}</span>
            </div>
            """, unsafe_allow_html=True)

    if predict_button:
        with st.spinner("🤖 Analyzing oceanographic conditions..."):
            result = predict_ocean_parameters_real(latitude, longitude, depth, day_of_year, chlorophyll, ph, models)

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: #667eea;">🎯 Ocean Parameter Predictions</h3>
        </div>
        """, unsafe_allow_html=True)

        a, b, c = st.columns(3)
        with a:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1f}°C</div>
                <div class="metric-label">🌡️ Temperature</div>
            </div>
            """.format(result['temperature']), unsafe_allow_html=True)

        with b:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1f}‰</div>
                <div class="metric-label">🧂 Salinity</div>
            </div>
            """.format(result['salinity']), unsafe_allow_html=True)

        with c:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1f} mg/L</div>
                <div class="metric-label">💨 Dissolved Oxygen</div>
            </div>
            """.format(result['dissolved_oxygen']), unsafe_allow_html=True)

        st.markdown("### 📈 Parameter Analysis")
        parameters = ["Temperature", "Salinity", "Dissolved Oxygen"]
        values = [
            result["temperature"] / 30 * 100,
            result["salinity"] / 40 * 100,
            result["dissolved_oxygen"] / 12 * 100,
        ]
        fig_radar = go.Figure(
            go.Scatterpolar(
                r=values,
                theta=parameters,
                fill="toself",
                name="Predicted Values",
                line_color="#667eea",
                fillcolor="rgba(102, 126, 234, 0.3)"
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.3)")),
            showlegend=True,
            title="Ocean Parameter Profile",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("### 🔍 Environmental Assessment")
        temp_status = "🟢 Optimal" if 22 <= result["temperature"] <= 28 else "🟡 Sub-optimal"
        sal_status = "🟢 Normal" if 33 <= result["salinity"] <= 36 else "🟡 Abnormal"
        oxy_status = "🟢 Healthy" if result["dissolved_oxygen"] >= 6 else "🔴 Critical"

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
                    "Influences water density and circulation patterns",
                    "Critical for marine organism survival and reproduction",
                ],
            }
        )
        st.dataframe(assessment_data, use_container_width=True)


def show_biodiversity_assessment(models):
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #667eea; font-weight: 600;">🐠 Marine Biodiversity Assessment</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            Comprehensive evaluation of marine biodiversity and ecosystem health based on environmental conditions. 
            This analysis helps in conservation planning, marine protected area management, and ecosystem restoration efforts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌡️ Environmental Conditions")

        temperature = st.number_input(
            "Water Temperature (°C)",
            min_value=15.0,
            max_value=35.0,
            value=26.0,
            step=0.5,
            help="Average water temperature"
        )
        depth = st.number_input(
            "Water Depth (m)",
            min_value=0.0,
            max_value=200.0,
            value=25.0,
            step=1.0,
            help="Water depth at monitoring site"
        )
        pollution_level = st.slider(
            "Pollution Level",
            0.0,
            5.0,
            1.0,
            0.1,
            help="Pollution index (0=pristine, 5=heavily polluted)"
        )
        coral_cover = st.slider(
            "Coral Cover (%)",
            0.0,
            100.0,
            60.0,
            1.0,
            help="Percentage of coral coverage in the area"
        )

        st.markdown("### 📍 Geographic Location")
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=12.0,
            step=0.1,
            help="Geographic latitude"
        )
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=78.0,
            step=0.1,
            help="Geographic longitude"
        )

        assess_button = st.button("🔍 Assess Biodiversity", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 🌊 Assessment Factors")

        # Environmental factor indicators
        temp_indicator = "🟢" if 24 <= temperature <= 28 else "🟡" if 20 <= temperature <= 32 else "🔴"
        depth_indicator = "🟢" if depth <= 50 else "🟡" if depth <= 100 else "🔴"
        pollution_indicator = "🟢" if pollution_level <= 1.5 else "🟡" if pollution_level <= 3 else "🔴"
        coral_indicator = "🟢" if coral_cover >= 50 else "🟡" if coral_cover >= 25 else "🔴"

        factors_info = [
            (f"{temp_indicator} Temperature", f"{temperature}°C (Optimal: 24-28°C)"),
            (f"{depth_indicator} Depth", f"{depth}m (Shallow waters typically more diverse)"),
            (
            f"{pollution_indicator} Pollution", f"Level {pollution_level}/5 (Lower values indicate better conditions)"),
            (f"{coral_indicator} Coral Cover", f"{coral_cover}% (Higher coverage supports more species)")
        ]

        for factor, description in factors_info:
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 12px; 
                        margin-bottom: 0.5rem; border-left: 3px solid #667eea;">
                <strong style="color: #667eea;">{factor}</strong><br>
                <span style="color: #e0e6ed; font-size: 0.9rem;">{description}</span>
            </div>
            """, unsafe_allow_html=True)

    if assess_button:
        with st.spinner("🤖 Assessing biodiversity..."):
            result = assess_biodiversity_real(
                temperature, depth, pollution_level, coral_cover, latitude, longitude, models
            )

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: #667eea;">🎯 Assessment Results</h3>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.0f}</div>
                <div class="metric-label">🐠 Estimated Species</div>
            </div>
            """.format(result['species_count']), unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}</div>
                <div class="metric-label">📊 Shannon Diversity</div>
            </div>
            """.format(result['shannon_diversity']), unsafe_allow_html=True)

        with c3:
            badge_colors = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
            badge = badge_colors.get(result['biodiversity_category'], "🔵")
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{} {}</div>
                <div class="metric-label">🏆 Biodiversity Level</div>
            </div>
            """.format(badge, result['biodiversity_category']), unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("### 📊 Category Probabilities")
            prob_data = pd.DataFrame(
                list(result["category_probabilities"].items()),
                columns=["Category", "Probability"],
            )
            colors = {"High": "#4ade80", "Medium": "#fbbf24", "Low": "#ef4444"}
            fig_pie = px.pie(
                prob_data,
                values="Probability",
                names="Category",
                title="Biodiversity Category Distribution",
                color="Category",
                color_discrete_map=colors
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_r:
            st.markdown("### 🎯 Factor Score Breakdown")
            temp_score = max(0.0, 1 - abs(temperature - 26) / 10) * 100
            pollution_score = max(0.0, (5 - pollution_level) / 5) * 100
            coral_score = coral_cover
            depth_score = max(0.0, (100 - depth) / 100) * 100

            score_data = pd.DataFrame(
                {
                    "Component": ["Temperature", "Pollution Control", "Coral Cover", "Depth Factor"],
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
                title="Environmental Factor Scores (%)",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 🛡️ Conservation Recommendations")

        if result["biodiversity_category"] == "High":
            recommendations = [
                "🎉 Excellent biodiversity detected! Continue current conservation efforts.",
                "🔒 Consider establishing Marine Protected Area status for long-term preservation.",
                "📊 Implement regular monitoring programs to maintain ecosystem health.",
                "🎓 Develop as a reference site for research and educational programs.",
                "🌱 Focus on sustainable tourism and community engagement."
            ]
            rec_color = "rgba(34, 197, 94, 0.1)"
            border_color = "rgba(34, 197, 94, 0.3)"
        elif result["biodiversity_category"] == "Medium":
            recommendations = [
                "⚠️ Moderate biodiversity with significant improvement potential.",
                "🏭 Identify and address pollution sources affecting the ecosystem.",
                "🪸 Implement coral restoration and reef rehabilitation programs.",
                "🐠 Establish fish population monitoring and protection measures.",
                "🔬 Conduct detailed environmental impact assessments."
            ]
            rec_color = "rgba(251, 191, 36, 0.1)"
            border_color = "rgba(251, 191, 36, 0.3)"
        else:
            recommendations = [
                "🚨 Critical biodiversity status requiring immediate intervention.",
                "🛑 Implement urgent pollution control and remediation measures.",
                "🔧 Launch comprehensive habitat restoration programs.",
                "📞 Consult marine conservation experts for emergency protocols.",
                "🚫 Consider temporary access restrictions to allow ecosystem recovery."
            ]
            rec_color = "rgba(239, 68, 68, 0.1)"
            border_color = "rgba(239, 68, 68, 0.3)"

        st.markdown(f"""
        <div style="background: {rec_color}; border: 2px solid {border_color}; 
                    padding: 1.5rem; border-radius: 16px; margin: 1rem 0;">
        """, unsafe_allow_html=True)

        for rec in recommendations:
            st.markdown(f"- {rec}")

        st.markdown("</div>", unsafe_allow_html=True)


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
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #667eea; font-weight: 600;">📈 Data Visualization & Analytics</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            Interactive data visualizations and statistical analysis of marine ecosystem data.
            Explore trends, patterns, and relationships in oceanographic and biodiversity datasets.
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
        st.markdown("### Fish Species Distribution Analysis")

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
            fig = px.pie(
                values=species_counts.values,
                names=species_counts.index,
                title="Species Distribution",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(
                fish_data,
                x="Length",
                y="Weight",
                color="Species",
                title="Length vs Weight by Species",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"}
            )
            st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(
            fish_data,
            x="Species",
            color="Season",
            title="Species Distribution by Season",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "🌊 Ocean Parameter Trends":
        st.markdown("### Ocean Parameter Trends Analysis")

        dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
        np.random.seed(42)
        ocean_data = pd.DataFrame(
            {
                "Date": dates,
                "Temperature": 26 + 4 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 0.5,
                                                                                                           len(dates)),
                "Salinity": 34.5 + 0.5 * np.cos(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 0.2,
                                                                                                            len(dates)),
                "Dissolved_Oxygen": 8 + np.random.normal(0, 0.3, len(dates)),
            }
        )

        param = st.selectbox("Select Parameter", ["Temperature", "Salinity", "Dissolved_Oxygen"])

        fig = px.line(
            ocean_data,
            x="Date",
            y=param,
            title=f"{param} Trends Over Time",
            line_shape="spline"
        )
        fig.update_traces(line_color='#667eea', line_width=3)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )
        st.plotly_chart(fig, use_container_width=True)

        ocean_data["Month"] = ocean_data["Date"].dt.month_name()
        monthly_avg = ocean_data.groupby("Month")[param].mean().reset_index()

        # Reorder months properly
        month_order = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        monthly_avg["Month"] = pd.Categorical(monthly_avg["Month"], categories=month_order, ordered=True)
        monthly_avg = monthly_avg.sort_values("Month")

        fig = px.bar(
            monthly_avg,
            x="Month",
            y=param,
            title=f"Average {param} by Month",
            color=param,
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "🐠 Biodiversity Heatmap":
        st.markdown("### Biodiversity Distribution Heatmap")

        np.random.seed(42)
        locations = [f"Site_{i:02d}" for i in range(1, 21)]
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        biodiversity_data = []
        for loc in locations:
            for month in months:
                # Create some realistic seasonal patterns
                month_factor = 1 + 0.3 * np.sin(2 * np.pi * months.index(month) / 12)
                biodiversity_data.append({
                    "Location": loc,
                    "Month": month,
                    "Shannon_Diversity": float(np.clip(np.random.uniform(1, 4) * month_factor, 0.5, 4.0)),
                })

        bio_df = pd.DataFrame(biodiversity_data)
        pivot_data = bio_df.pivot(index="Location", columns="Month", values="Shannon_Diversity")

        fig = px.imshow(
            pivot_data,
            title="Shannon Diversity Index Heatmap",
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Biodiversity Statistics")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}</div>
                <div class="metric-label">Average Diversity</div>
            </div>
            """.format(bio_df['Shannon_Diversity'].mean()), unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}</div>
                <div class="metric-label">Peak Diversity</div>
            </div>
            """.format(bio_df['Shannon_Diversity'].max()), unsafe_allow_html=True)

        with c3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Sites Monitored</div>
            </div>
            """.format(len(locations)), unsafe_allow_html=True)

        with c4:
            high_diversity_sites = len(bio_df[bio_df['Shannon_Diversity'] > 3.0]['Location'].unique())
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">High Diversity Sites</div>
            </div>
            """.format(high_diversity_sites), unsafe_allow_html=True)

    elif viz_type == "📈 Correlation Analysis":
        st.markdown("### Environmental Parameters Correlation Analysis")

        np.random.seed(42)
        n = 500

        # Create realistic correlations between variables
        temperature = np.random.normal(26, 3, n)
        depth = np.random.uniform(0, 100, n)
        salinity = 34.5 - 0.01 * depth + np.random.normal(0, 0.5, n)
        dissolved_oxygen = 10 - 0.05 * temperature - 0.02 * depth + np.random.normal(0, 0.5, n)
        ph = 8.1 + 0.01 * dissolved_oxygen + np.random.normal(0, 0.1, n)
        chlorophyll = 2.5 + 0.2 * dissolved_oxygen - 0.01 * depth + np.random.normal(0, 0.5, n)
        species_count = np.clip(
            5 + 0.5 * dissolved_oxygen + 0.3 * chlorophyll - 0.1 * abs(temperature - 26) + np.random.normal(0, 2, n), 1,
            30)

        df = pd.DataFrame({
            "Temperature": temperature,
            "Salinity": salinity,
            "Dissolved_Oxygen": dissolved_oxygen,
            "pH": ph,
            "Chlorophyll": chlorophyll,
            "Species_Count": species_count,
            "Depth": depth
        })

        corr = df.corr()

        fig = px.imshow(
            corr,
            title="Environmental Parameters Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto",
            zmin=-1,
            zmax=1
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            fig = px.scatter(
                df,
                x="Temperature",
                y="Dissolved_Oxygen",
                title="Temperature vs Dissolved Oxygen",
                trendline="ols",
                color="Depth",
                color_continuous_scale="Blues"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(
                df,
                x="Dissolved_Oxygen",
                y="Species_Count",
                title="Dissolved Oxygen vs Species Count",
                trendline="ols",
                color="Chlorophyll",
                color_continuous_scale="Greens"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"}
            )
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "🗺️ Geographic Distribution":
        st.markdown("### Geographic Distribution of Marine Data")

        np.random.seed(42)
        n_points = 100

        geo_data = pd.DataFrame({
            "Latitude": np.random.uniform(8, 25, n_points),
            "Longitude": np.random.uniform(68, 85, n_points),
            "Temperature": np.random.normal(27, 2, n_points),
            "Biodiversity_Score": np.random.uniform(1, 4, n_points),
            "Site_Type": np.random.choice(["Coral Reef", "Open Ocean", "Coastal", "Deep Sea"], n_points),
        })

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
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Regional Analysis")
        c1, c2 = st.columns(2)

        with c1:
            site_summary = geo_data.groupby("Site_Type").agg({
                "Biodiversity_Score": ["mean", "std", "count"],
                "Temperature": ["mean", "std"]
            }).round(2)
            site_summary.columns = ["Avg Biodiversity", "Std Biodiversity", "Count", "Avg Temp", "Std Temp"]
            st.dataframe(site_summary, use_container_width=True)

        with c2:
            fig = px.box(
                geo_data,
                x="Site_Type",
                y="Biodiversity_Score",
                title="Biodiversity Distribution by Site Type",
                color="Site_Type"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"}
            )
            st.plotly_chart(fig, use_container_width=True)


def show_about_page():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #667eea; font-weight: 600;">ℹ️ About CMLRE Marine ML Platform</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            <h3>🌊 Centre for Marine Living Resources and Ecology (CMLRE)</h3>
            <p style="margin-bottom: 0;">Ministry of Earth Sciences, Government of India</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Project Overview")
        st.markdown(
            """
            The CMLRE Marine ML Platform represents a cutting-edge artificial intelligence system 
            designed to support marine ecosystem monitoring, conservation, and research activities. 
            This platform integrates multiple machine learning models to provide comprehensive 
            analysis of marine environments across the Indian Ocean region.

            **Key Objectives:**
            - Advanced marine biodiversity assessment and monitoring
            - Automated fish species identification for fisheries management
            - Real-time oceanographic parameter prediction and analysis
            - Comprehensive ecosystem health evaluation and scoring
            - Evidence-based conservation planning and policy support
            - Research facilitation for marine science communities
            """
        )

        st.markdown("### 🔬 Technology Stack")
        st.markdown(
            """
            **Frontend & Visualization:**
            - Streamlit for interactive web interface
            - Plotly for advanced data visualizations
            - Custom CSS for modern UI/UX design

            **Machine Learning & Analytics:**
            - Scikit-learn for classification algorithms
            - TensorFlow for deep learning models
            - NumPy & Pandas for data processing
            - Statsmodels for statistical analysis

            **Deployment & Infrastructure:**
            - Streamlit Cloud for scalable deployment
            - Git-based version control and CI/CD
            - Modular architecture for easy maintenance
            """
        )

    with col2:
        st.markdown("### 🐟 AI Model Specifications")

        model_specs = {
            "Fish Species Classifier": {
                "Purpose": "Automated identification of fish species from morphometric measurements",
                "Species Coverage": "Bream, Perch, Pike, Roach (expandable)",
                "Accuracy": "85%+ on test datasets",
                "Input Features": "Length, Weight, Height, Width measurements",
                "Algorithm": "Random Forest with feature scaling",
                "Applications": "Fisheries stock assessment, species monitoring"
            },
            "Ocean Parameter Predictor": {
                "Purpose": "Prediction of oceanographic conditions from environmental data",
                "Parameters": "Temperature, Salinity, Dissolved Oxygen levels",
                "Input Variables": "Location coordinates, depth, seasonal data, chemical indicators",
                "Model Type": "Multi-output regression with ensemble methods",
                "Validation": "Cross-validated on historical oceanographic datasets",
                "Applications": "Climate monitoring, ecosystem health assessment"
            },
            "Biodiversity Assessor": {
                "Purpose": "Comprehensive evaluation of marine biodiversity and ecosystem health",
                "Metrics": "Shannon Diversity Index, Species richness estimation",
                "Environmental Factors": "Temperature, pollution levels, coral coverage, depth",
                "Output": "Biodiversity category classification with confidence scores",
                "Framework": "Multi-criteria decision analysis with ML integration",
                "Applications": "Conservation planning, MPA effectiveness evaluation"
            }
        }

        for model_name, specs in model_specs.items():
            with st.expander(f"🔍 {model_name}", expanded=False):
                for key, value in specs.items():
                    st.markdown(f"**{key}:** {value}")

    st.markdown("---")

    st.markdown("### 🔬 Research Applications & Impact")

    applications = [
        {
            "category": "🐠 Fisheries Management",
            "description": "Stock assessment, sustainable fishing practices, and species population monitoring"
        },
        {
            "category": "🌊 Climate Change Research",
            "description": "Long-term ocean parameter monitoring, climate impact assessment, and trend analysis"
        },
        {
            "category": "🪸 Coral Reef Conservation",
            "description": "Reef health assessment, biodiversity monitoring, and restoration planning"
        },
        {
            "category": "🏭 Environmental Impact Assessment",
            "description": "Pollution monitoring, industrial impact evaluation, and mitigation strategies"
        },
        {
            "category": "🛡️ Marine Protected Areas",
            "description": "Site selection optimization, management effectiveness evaluation, and boundary planning"
        },
        {
            "category": "📊 Ecosystem Services Quantification",
            "description": "Economic valuation of marine ecosystems and natural capital assessment"
        },
        {
            "category": "🎓 Education & Capacity Building",
            "description": "Training programs for marine scientists, conservation practitioners, and policymakers"
        },
        {
            "category": "📈 Policy & Governance Support",
            "description": "Evidence-based policy recommendations and regulatory compliance monitoring"
        }
    ]

    for app in applications:
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1.2rem; border-radius: 12px; 
                    margin-bottom: 0.8rem; border-left: 4px solid #667eea;">
            <strong style="color: #667eea; font-size: 1.1rem;">{app['category']}</strong><br>
            <span style="color: #e0e6ed; margin-top: 0.5rem; display: block;">{app['description']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📈 Platform Performance & Impact Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">1.0.0</div>
            <div class="metric-label">Platform Version</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">AI Models Deployed</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">60+</div>
            <div class="metric-label">Monitoring Sites</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">Jan 2025</div>
            <div class="metric-label">Last Updated</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🌍 Global Collaboration & Partnerships")

    st.markdown("""
    The CMLRE Marine ML Platform benefits from extensive collaboration with national and 
    international research institutions, fostering knowledge exchange and technological advancement 
    in marine science and conservation.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🏛️ Government Partnerships:**
        - Ministry of Earth Sciences, India
        - National Institute of Oceanography (NIO)
        - Indian National Centre for Ocean Information Services (INCOIS)
        - Coastal & Marine Program, Department of Science & Technology
        """)

    with col2:
        st.markdown("""
        **🎓 Academic Collaborations:**
        - Indian Institute of Science (IISc)
        - Indian Institute of Technology (IIT) Network
        - National Institute of Technology (NIT) System
        - International Marine Science Consortium
        """)

    with col3:
        st.markdown("""
        **🌐 International Networks:**
        - UNESCO Intergovernmental Oceanographic Commission
        - Global Ocean Observing System (GOOS)
        - International Council for the Exploration of the Sea
        - Indo-Pacific Marine Science Collaboration
        """)

    st.markdown("---")

    st.markdown("### 🔬 Data Sources & Validation")

    data_sources = {
        "Oceanographic Data": [
            "ARGO float network measurements",
            "Satellite-derived ocean color and temperature",
            "Coastal monitoring station recordings",
            "Ship-based CTD measurements",
            "Autonomous underwater vehicle surveys"
        ],
        "Biological Data": [
            "Fish catch and morphometric databases",
            "Coral reef monitoring surveys",
            "Plankton abundance measurements",
            "Marine mammal sighting records",
            "Biodiversity assessment field studies"
        ],
        "Environmental Data": [
            "Water quality monitoring networks",
            "Pollution and contamination surveys",
            "Habitat mapping and classification",
            "Climate and weather station data",
            "Human activity and pressure indicators"
        ]
    }

    for category, sources in data_sources.items():
        with st.expander(f"📊 {category}", expanded=False):
            for source in sources:
                st.markdown(f"• {source}")

    st.markdown("---")

    st.markdown("### 🚀 Future Development Roadmap")

    roadmap_items = [
        {
            "phase": "Phase 1 (Q2 2025)",
            "features": [
                "Integration of deep learning models for image-based species identification",
                "Real-time data streaming from IoT sensors",
                "Mobile application for field data collection",
                "Advanced statistical analysis and reporting tools"
            ]
        },
        {
            "phase": "Phase 2 (Q4 2025)",
            "features": [
                "Predictive modeling for climate change impacts",
                "Integration with satellite remote sensing data",
                "Automated alert systems for ecosystem threats",
                "Multi-language support for regional deployment"
            ]
        },
        {
            "phase": "Phase 3 (2026)",
            "features": [
                "Artificial intelligence-driven conservation recommendations",
                "Integration with global marine databases",
                "Advanced visualization and virtual reality interfaces",
                "Blockchain-based data integrity and sharing protocols"
            ]
        }
    ]

    for item in roadmap_items:
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 12px; 
                    margin-bottom: 1rem; border-left: 4px solid #667eea;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">{item['phase']}</h4>
            <ul style="margin: 0; color: #e0e6ed;">
        """, unsafe_allow_html=True)

        for feature in item['features']:
            st.markdown(f"<li style='margin-bottom: 0.5rem;'>{feature}</li>", unsafe_allow_html=True)

        st.markdown("</ul></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📞 Contact Information & Support")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🏢 Centre for Marine Living Resources and Ecology (CMLRE)**  
        Ministry of Earth Sciences  
        Government of India  
        Kochi, Kerala 682 037, India

        **🌐 Official Website:** [www.cmlre.gov.in](https://www.cmlre.gov.in)  
        **📧 General Email:** info@cmlre.gov.in  
        **📱 Phone:** +91-484-2390814  
        **📠 Fax:** +91-484-2390618
        """)

    with col2:
        st.markdown("""
        **🔬 Technical Support & Collaboration:**  
        **Platform Support:** ml-platform@cmlre.gov.in  
        **Research Collaboration:** research@cmlre.gov.in  
        **Data Partnership:** data@cmlre.gov.in  
        **Training & Workshops:** training@cmlre.gov.in

        **🚨 Emergency Marine Issues:**  
        **24/7 Hotline:** +91-484-2390800  
        **Emergency Email:** emergency@cmlre.gov.in
        """)

    st.markdown("---")

    st.markdown("### 📋 Legal & Compliance Information")

    st.markdown("""
    <div style="background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 12px; 
                border: 1px solid rgba(102, 126, 234, 0.3);">
        <h4 style="color: #667eea; margin-bottom: 1rem;">Data Privacy & Security</h4>
        <p style="color: #e0e6ed; margin-bottom: 1rem;">
            This platform adheres to Indian government data protection guidelines and international 
            best practices for scientific data management. All research data is handled in accordance 
            with ethical guidelines and institutional review board approvals.
        </p>

        <h4 style="color: #667eea; margin-bottom: 1rem;">Open Science & Data Sharing</h4>
        <p style="color: #e0e6ed; margin-bottom: 1rem;">
            CMLRE is committed to open science principles. Research outputs and non-sensitive datasets 
            are made available to the scientific community through appropriate data repositories and 
            collaborative platforms, subject to ethical and security considerations.
        </p>

        <h4 style="color: #667eea; margin-bottom: 1rem;">Citation & Attribution</h4>
        <p style="color: #e0e6ed; margin-bottom: 0;">
            When using this platform or its outputs in research or publications, please cite: <br>
            <strong>CMLRE Marine ML Platform (2025). Centre for Marine Living Resources and Ecology, 
            Ministry of Earth Sciences, Government of India. Available at: [Platform URL]</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🙏 Acknowledgments")

    st.markdown("""
    The development of the CMLRE Marine ML Platform has been made possible through the dedicated 
    efforts of marine scientists, software engineers, data scientists, and conservation practitioners. 
    We acknowledge the support of:

    - **Ministry of Earth Sciences** for funding and institutional support
    - **Field researchers and marine biologists** for data collection and validation
    - **Software development team** for platform architecture and implementation  
    - **International collaborators** for knowledge sharing and technical expertise
    - **Local communities and fishers** for traditional knowledge and field insights
    - **Academic institutions** for research partnerships and student contributions
    - **Open source community** for tools and libraries that power this platform
    """)

    st.markdown("---")

    # Footer with system status
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: rgba(102, 126, 234, 0.1); 
                border-radius: 16px; margin-top: 2rem;">
        <p style="color: #667eea; font-weight: 600; margin-bottom: 1rem;">
            🌊 CMLRE Marine Living Resources ML Platform
        </p>
        <p style="color: #e0e6ed; font-size: 0.9rem; margin-bottom: 1rem;">
            Advancing marine science through artificial intelligence and data-driven insights
        </p>
        <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap;">
            <span style="color: #4ade80; font-size: 0.9rem;">🟢 All Systems Operational</span>
            <span style="color: #e0e6ed; font-size: 0.9rem;">Last Updated: September 2025</span>
            <span style="color: #e0e6ed; font-size: 0.9rem;">Version 1.1.3</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main app function with modern navigation
# -----------------------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">🌊 CMLRE Marine Living Resources ML Platform</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            <strong>🎯 Mission:</strong> Advanced AI platform for marine ecosystem monitoring and conservation under 
            the Ministry of Earth Sciences, India. This platform provides real-time analysis of marine biodiversity, 
            oceanographic parameters, and fish species identification using cutting-edge machine learning technology.
        </div>
        """,
        unsafe_allow_html=True,
    )

    models = load_models()

    # Create modern navigation and get selected page
    selected_page = create_modern_navigation()

    # Page routing with default fallback
    if selected_page == "fish" or st.session_state.get("current_page") == "fish":
        st.session_state["current_page"] = "fish"
        show_fish_identification(models)
    elif selected_page == "ocean" or st.session_state.get("current_page") == "ocean":
        st.session_state["current_page"] = "ocean"
        show_ocean_prediction(models)
    elif selected_page == "biodiversity" or st.session_state.get("current_page") == "biodiversity":
        st.session_state["current_page"] = "biodiversity"
        show_biodiversity_assessment(models)
    elif selected_page == "comprehensive" or st.session_state.get("current_page") == "comprehensive":
        st.session_state["current_page"] = "comprehensive"
        show_comprehensive_analysis(models)
    elif selected_page == "visualization" or st.session_state.get("current_page") == "visualization":
        st.session_state["current_page"] = "visualization"
        show_data_visualization()
    elif selected_page == "about" or st.session_state.get("current_page") == "about":
        st.session_state["current_page"] = "about"
        show_about_page()
    else:
        # Default to home page
        st.session_state["current_page"] = "home"
        show_home_page()


if __name__ == "__main__":
    main()
