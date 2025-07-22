import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Configuration initiale
load_dotenv()
st.set_page_config(
    page_title="Air Quality Analytics Dashboard",
    layout="wide",
    page_icon="🌫️",
    initial_sidebar_state="expanded"
)

# Style personnalisé en dark mode
st.markdown("""
    <style>
        /* Couleurs de base */
        :root {
            --primary: #1E88E5;
            --background: #121212;
            --surface: #1E1E1E;
            --on-background: #E1E1E1;
            --on-surface: #FFFFFF;
            --error: #CF6679;
        }
        
        /* Application du dark mode */
        .main {background-color: var(--background); color: var(--on-background);}
        .stMetric {
            border-radius: 10px; 
            padding: 15px; 
            background-color: var(--surface); 
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            border-left: 4px solid var(--primary);
            color: var(--on-surface);
        }
        .stPlotlyChart {
            border-radius: 10px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            background-color: var(--surface);
        }
        .css-1aumxhk {
            background-color: var(--surface); 
            border-radius: 10px; 
            padding: 20px;
            border: 1px solid #333;
        }
        .stAlert {
            border-radius: 10px;
            border-left: 4px solid;
        }
        .st-bb {background-color: var(--surface);}
        .st-at {background-color: var(--primary);}
        .st-ae {background-color: #2E2E2E;}
        
        /* Textes */
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stMarkdown p {
            color: var(--on-surface) !important;
        }
        
        /* Onglets */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: var(--surface);
            border-radius: 8px 8px 0 0 !important;
            padding: 10px 20px;
            transition: all 0.3s;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--primary) !important;
            color: white !important;
        }
        
        /* Couleurs pour les catégories AQI */
        .good {background-color: #1B5E20; color: #A5D6A7; border-left: 4px solid #4CAF50;}
        .moderate {background-color: #F57F17; color: #FFF9C4; border-left: 4px solid #FFC107;}
        .unhealthy-sensitive {background-color: #E65100; color: #FFCC80; border-left: 4px solid #FF9800;}
        .unhealthy {background-color: #B71C1C; color: #FFCDD2; border-left: 4px solid #F44336;}
        .very-unhealthy {background-color: #4A148C; color: #D1C4E9; border-left: 4px solid #7C4DFF;}
        .hazardous {background-color: #263238; color: #ECEFF1; border-left: 4px solid #78909C;}
        
        /* Tableaux */
        .dataframe {
            background-color: var(--surface) !important;
            color: var(--on-surface) !important;
        }
        .dataframe thead th {
            background-color: #333 !important;
            color: white !important;
        }
        .dataframe tbody tr:nth-child(even) {
            background-color: #2A2A2A !important;
        }
        
        /* Slider */
        .stSlider .st-cc {
            color: var(--on-surface);
        }
        .stSlider .st-cd {
            background-color: var(--primary);
        }
        
        /* Sélecteurs */
        .stSelectbox, .stRadio, .stButton>button {
            background-color: var(--surface);
            color: var(--on-surface);
            border: 1px solid #444;
        }
    </style>
""", unsafe_allow_html=True)

# Dictionnaire de villes
CITY_DATA = {
    "Paris": {"coordinates": (48.8566, 2.3522), "population": "2.1M", "country": "France"},
    "London": {"coordinates": (51.5074, -0.1278), "population": "8.9M", "country": "UK"},
    "New York": {"coordinates": (40.7128, -74.0060), "population": "8.4M", "country": "USA"},
    "Tokyo": {"coordinates": (35.6762, 139.6503), "population": "13.9M", "country": "Japan"},
    "Beijing": {"coordinates": (39.9042, 116.4074), "population": "21.5M", "country": "China"},
    "Delhi": {"coordinates": (28.7041, 77.1025), "population": "31.4M", "country": "India"}
}

# Échelle de couleurs adaptée au dark mode
COLOR_SCALE = [
    "#4CAF50",  # Good
    "#FFC107",  # Moderate
    "#FF9800",  # Unhealthy for Sensitive Groups
    "#F44336",  # Unhealthy
    "#7C4DFF",  # Very Unhealthy
    "#78909C"   # Hazardous
]

AQI_CATEGORIES = {
    (0, 50): {"label": "Good", "color": COLOR_SCALE[0], "health_implications": "Air quality is satisfactory."},
    (51, 100): {"label": "Moderate", "color": COLOR_SCALE[1], "health_implications": "Acceptable quality."},
    (101, 150): {"label": "Unhealthy for Sensitive Groups", "color": COLOR_SCALE[2], "health_implications": "General public not likely affected."},
    (151, 200): {"label": "Unhealthy", "color": COLOR_SCALE[3], "health_implications": "Everyone may begin to experience health effects."},
    (201, 300): {"label": "Very Unhealthy", "color": COLOR_SCALE[4], "health_implications": "Health warnings of emergency conditions."},
    (301, float('inf')): {"label": "Hazardous", "color": COLOR_SCALE[5], "health_implications": "Health alert: everyone may experience serious health effects."}
}

def get_aqi_category(pm25_value):
    for (low, high), category in AQI_CATEGORIES.items():
        if low <= pm25_value <= high:
            return category
    return {"label": "Unknown", "color": "#999999"}

def get_coordinates_fallback(city_name):
    if city_name in CITY_DATA:
        return CITY_DATA[city_name]["coordinates"]
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if api_key:
        try:
            response = requests.get(
                "http://api.openweathermap.org/geo/1.0/direct",
                params={'q': city_name, 'limit': 1, 'appid': api_key},
                timeout=5
            )
            if response.status_code == 200 and response.json():
                data = response.json()[0]
                return data['lat'], data['lon']
        except Exception as e:
            st.error(f"Geocoding error: {str(e)}")
    
    st.warning(f"City '{city_name}' not found. Using Paris as default.")
    return CITY_DATA["Paris"]["coordinates"]

def get_air_quality_data(lat, lon):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        st.warning("Demo mode: using simulated data")
        return generate_demo_data(lat, lon), generate_historical_data(lat, lon)
    
    try:
        current_response = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution",
            params={'lat': lat, 'lon': lon, 'appid': api_key},
            timeout=10
        )
        
        historical_response = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution/history",
            params={
                'lat': lat,
                'lon': lon,
                'start': int((datetime.now() - timedelta(days=30)).timestamp()),
                'end': int(datetime.now().timestamp()),
                'appid': api_key
            },
            timeout=15
        )
        
        current_data = None
        historical_df = None
        
        if current_response.status_code == 200:
            current_data = current_response.json()['list'][0]['components']
            current_data['aqi'] = current_response.json()['list'][0]['main']['aqi']
        
        if historical_response.status_code == 200:
            historical_data = historical_response.json()['list']
            historical_df = pd.DataFrame([
                {
                    'ds': datetime.fromtimestamp(item['dt']),
                    'y': item['components']['pm2_5'],
                    'pm10': item['components']['pm10'],
                    'no2': item['components']['no2'],
                    'o3': item['components']['o3'],
                    'so2': item['components']['so2'],
                    'co': item['components']['co']
                }
                for item in historical_data
            ])
        
        if not current_data:
            st.warning("Current data unavailable. Demo mode activated.")
            current_data = generate_demo_data(lat, lon)
        
        if historical_df is None:
            st.warning("Historical data unavailable. Generating simulated data.")
            historical_df = generate_historical_data(lat, lon)
        
        return current_data, historical_df
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        st.warning("Using simulated data")
        current_data = generate_demo_data(lat, lon)
        historical_df = generate_historical_data(lat, lon)
        return current_data, historical_df

def generate_demo_data(lat, lon):
    base_pm25 = 30 if lat >= 0 else 20
    return {
        'pm2_5': round(base_pm25 * (1 + np.random.uniform(-0.3, 0.5))),
        'pm10': round(base_pm25 * 1.5 * (1 + np.random.uniform(-0.2, 0.4))),
        'no2': round(20 * (1 + np.random.uniform(-0.4, 0.6))),
        'o3': round(50 * (1 + np.random.uniform(-0.3, 0.5))),
        'so2': round(5 * (1 + np.random.uniform(-0.5, 0.7))),
        'co': round(0.5 * (1 + np.random.uniform(-0.4, 0.8)), 1),
        'aqi': round((base_pm25 / 25) * 100 * (1 + np.random.uniform(-0.2, 0.3)))
    }

def generate_historical_data(lat, lon, days=30):
    dates = pd.date_range(end=datetime.now(), periods=days).tz_localize(None)
    base = np.linspace(20, 40, days) if lat >= 0 else np.linspace(10, 30, days)
    seasonal = 15 * np.sin(np.linspace(0, 3*np.pi, days))
    noise = np.random.normal(0, 5, days)
    pm25 = np.clip(base + seasonal + noise, 5, 150)
    
    return pd.DataFrame({
        'ds': dates,
        'y': pm25,
        'pm10': np.clip(pm25 * 1.5 + np.random.normal(0, 3, days), 5, 200),
        'no2': np.clip(np.linspace(15, 25, days) + np.random.normal(0, 4, days), 5, 50),
        'o3': np.clip(np.linspace(40, 60, days) + np.random.normal(0, 5, days), 20, 100),
        'so2': np.clip(np.linspace(3, 8, days) + np.random.normal(0, 1, days), 1, 15),
        'co': np.clip(np.linspace(0.3, 0.7, days) + np.random.normal(0, 0.1, days), 0.1, 1.0)
    })

def create_gauge_chart(value, min_val, max_val, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={'suffix': "µg/m³", 'font': {'size': 20, 'color': 'white'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': title,
            'font': {'size': 18, 'color': 'white'}
        },
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': 'white'},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': '#1E1E1E',
            'borderwidth': 2,
            'bordercolor': '#444',
            'steps': [
                {'range': [min_val, min_val + (max_val-min_val)*0.3], 'color': '#1B5E20'},
                {'range': [min_val + (max_val-min_val)*0.3, min_val + (max_val-min_val)*0.6], 'color': '#F57F17'},
                {'range': [min_val + (max_val-min_val)*0.6, max_val], 'color': '#B71C1C'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        margin=dict(t=50, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"}
    )
    return fig

# Interface utilisateur
st.title("🌍 Air Quality Analytics Dashboard")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=100)
    st.header("Configuration")
    city = st.selectbox(
        "Select City",
        list(CITY_DATA.keys()),
        index=0,
        help="Choose a city to analyze air quality"
    )
    
    days_to_predict = st.slider(
        "Forecast Period (days)",
        1, 14, 7,
        help="Number of days to predict air quality"
    )
    
    analysis_type = st.radio(
        "Analysis Focus",
        ["PM2.5", "Comprehensive"],
        index=0,
        help="Select primary analysis focus"
    )
    
    if st.button("Refresh Data", help="Fetch latest air quality data"):
        st.rerun()
    
    st.markdown("---")
    st.subheader("City Information")
    if city in CITY_DATA:
        city_info = CITY_DATA[city]
        st.markdown(f"**Country:** {city_info['country']}")
        st.markdown(f"**Population:** {city_info['population']}")
        st.markdown(f"**Coordinates:** {city_info['coordinates'][0]:.4f}, {city_info['coordinates'][1]:.4f}")

# Données principales
lat, lon = get_coordinates_fallback(city)
current_data, historical_data = get_air_quality_data(lat, lon)

# Section d'en-tête avec métriques
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    aqi_category = get_aqi_category(current_data['pm2_5'])
    st.metric(
        "AQI Index",
        f"{current_data.get('aqi', (current_data['pm2_5'] / 25) * 100):.0f}",
        aqi_category['label']
    )

with col2:
    st.metric(
        "PM2.5 Concentration",
        f"{current_data['pm2_5']} µg/m³",
        "WHO Guideline: 5 µg/m³"
    )

with col3:
    st.metric(
        "PM10 Concentration",
        f"{current_data['pm10']} µg/m³",
        "WHO Guideline: 15 µg/m³"
    )

with col4:
    st.metric(
        "Primary Pollutant",
        "PM2.5" if current_data['pm2_5'] > current_data['pm10'] else "PM10",
        "Most significant pollutant"
    )

# Onglets
tab1, tab2, tab3, tab4 = st.tabs(["Current Status", "Historical Trends", "Forecast", "Health Recommendations"])

with tab1:
    st.subheader(f"Current Air Quality in {city}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(create_gauge_chart(
            current_data['pm2_5'], 0, 100,
            "PM2.5 Concentration", aqi_category['color']
        ), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_gauge_chart(
            current_data['pm10'], 0, 150,
            "PM10 Concentration", aqi_category['color']
        ), use_container_width=True)
    
    with col3:
        st.plotly_chart(create_gauge_chart(
            current_data['no2'], 0, 50,
            "NO₂ Concentration", aqi_category['color']
        ), use_container_width=True)
    
    pollutants = ['pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co']
    pollutant_names = ['PM2.5', 'PM10', 'NO₂', 'O₃', 'SO₂', 'CO']
    values = [current_data[p] for p in pollutants]
    
    fig = px.bar(
        x=pollutant_names,
        y=values,
        labels={'x': 'Pollutant', 'y': 'Concentration (µg/m³)'},
        title="Current Pollutant Concentrations",
        color=pollutant_names,
        color_discrete_sequence=COLOR_SCALE
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        xaxis={'tickfont': {'color': 'white'}},
        yaxis={'tickfont': {'color': 'white'}}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"Historical Air Quality Trends in {city}")
    
    period = st.radio(
        "Time Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days"],
        horizontal=True
    )
    
    days = 7 if period == "Last 7 Days" else 30 if period == "Last 30 Days" else 90
    filtered_data = historical_data.tail(days)
    
    fig = px.line(
        filtered_data,
        x='ds',
        y=['y', 'pm10', 'no2', 'o3'],
        labels={'value': 'Concentration (µg/m³)', 'ds': 'Date'},
        title=f"Pollutant Trends Over Time",
        color_discrete_sequence=COLOR_SCALE[:4]
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        legend_title_text='Pollutants',
        legend={'font': {'color': 'white'}},
        xaxis={'tickfont': {'color': 'white'}},
        yaxis={'tickfont': {'color': 'white'}}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.dataframe(filtered_data.describe().style.background_gradient(cmap='Blues'))

with tab3:
    st.subheader(f"Air Quality Forecast for {city}")
    
    with st.spinner("Training prediction model..."):
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        model.fit(historical_data[['ds', 'y']])
        future = model.make_future_dataframe(periods=days_to_predict)
        forecast = model.predict(future)
    
    fig = px.line(
        forecast,
        x='ds',
        y=['yhat', 'yhat_lower', 'yhat_upper'],
        labels={'value': 'PM2.5 (µg/m³)', 'ds': 'Date'},
        title=f"PM2.5 Forecast for Next {days_to_predict} Days",
        color_discrete_sequence=[COLOR_SCALE[0], 'rgba(76, 175, 80, 0.2)', 'rgba(76, 175, 80, 0.2)']
    )
    fig.update_traces(fill='tonexty', line=dict(width=0))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        xaxis={'tickfont': {'color': 'white'}},
        yaxis={'tickfont': {'color': 'white'}}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Forecast Details")
    cols = st.columns(2)
    with cols[0]:
        st.metric(
            "Peak PM2.5",
            f"{forecast['yhat'].max():.1f} µg/m³",
            f"on {forecast.loc[forecast['yhat'].idxmax(), 'ds'].strftime('%b %d')}"
        )
    
    with cols[1]:
        st.metric(
            "Average PM2.5",
            f"{forecast['yhat'].mean():.1f} µg/m³",
            f"±{forecast['yhat_upper'].mean() - forecast['yhat'].mean():.1f} µg/m³"
        )
    
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_to_predict).style.background_gradient(
        subset=['yhat'], cmap='YlOrRd'
    ))

with tab4:
    st.subheader("Health Recommendations")
    aqi_value = current_data.get('aqi', (current_data['pm2_5'] / 25) * 100)
    category = get_aqi_category(current_data['pm2_5'])
    
    # Déterminer la classe CSS en fonction de la catégorie AQI
    aqi_class = ""
    if aqi_value <= 50:
        aqi_class = "good"
    elif aqi_value <= 100:
        aqi_class = "moderate"
    elif aqi_value <= 150:
        aqi_class = "unhealthy-sensitive"
    elif aqi_value <= 200:
        aqi_class = "unhealthy"
    elif aqi_value <= 300:
        aqi_class = "very-unhealthy"
    else:
        aqi_class = "hazardous"
    
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"""
        <div class="{aqi_class}" style="padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3>Current Air Quality Status</h3>
            <p><strong>Category:</strong> {category['label']}</p>
            <p><strong>PM2.5 Level:</strong> {current_data['pm2_5']} µg/m³</p>
            <p><strong>Health Implications:</strong> {category['health_implications']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="{aqi_class}" style="padding: 15px; border-radius: 10px;">
            <h3>Recommended Actions</h3>
            {f"<p style='color:white; font-weight:bold;'>✅ Normal outdoor activities are safe for everyone.</p>" if aqi_value <= 50 else ""}
            {f"<p style='color:white; font-weight:bold;'>⚠️ Sensitive individuals should consider reducing prolonged outdoor exertion.</p>" if 50 < aqi_value <= 100 else ""}
            {f"<p style='color:white; font-weight:bold;'>⚠️ Sensitive groups should avoid prolonged outdoor exertion.</p>" if 100 < aqi_value <= 150 else ""}
            {f"<p style='color:white; font-weight:bold;'>❌ Everyone should avoid prolonged outdoor exertion.</p>" if aqi_value > 150 else ""}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### Additional Recommendations")
    st.markdown("""
    - Check real-time air quality before planning outdoor activities
    - Use air purifiers indoors when AQI is high
    - Keep windows closed during high pollution periods
    - Consider wearing N95 masks in very unhealthy conditions
    - Stay hydrated to help your body cope with pollution
    """)

# Pied de page
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #9E9E9E; font-size: 0.9em;">
    <p>Data sources: OpenWeatherMap API | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p>For reference: WHO annual air quality guideline values - PM2.5: 5 µg/m³, PM10: 15 µg/m³</p>
</div>
""", unsafe_allow_html=True)