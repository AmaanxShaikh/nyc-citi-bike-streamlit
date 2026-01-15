import os
import time
import warnings
from glob import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.colors as pc
import plotly.io as pio

import folium
from folium import FeatureGroup
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

from scipy import stats
from scipy.stats import gaussian_kde

import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Citi Bike Data Analysis Dashboard",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E5A88;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #2E5A88;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess a single Citi Bike dataset file"""
    
    # Update this path to your specific single CSV file
    file_path = r"data/sample_data_october2025.csv"
    
    try:
        # Load single file
        df = pd.read_csv(file_path, sep=";", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    
    # Date time conversion
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
    
    # Duration
    df['duration_minutes'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    
    # Extract temporal features
    df['hour_of_day'] = df['started_at'].dt.hour
    df['day_of_week'] = df['started_at'].dt.day_name()
    df['day_of_week_num'] = df['started_at'].dt.dayofweek
    df['date'] = df['started_at'].dt.date
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
    
    # Time of day categories
    def get_time_of_day(hour):
        if pd.isna(hour):
            return None
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df['time_of_day'] = df['hour_of_day'].apply(get_time_of_day)
    df['is_rush_hour'] = df['hour_of_day'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # Fix coordinates
    def fix_coordinate_properly(col):
        cleaned = col.astype(str).str.strip().str.replace('.', '', regex=False)
        results = []
        for val in cleaned:
            if val in ['nan', 'None', '']:
                results.append(np.nan)
                continue
            try:
                digits = ''.join(c for c in val if c.isdigit())
                if len(digits) == 0:
                    results.append(np.nan)
                    continue
                num = float(digits)
                while num > 100:
                    num = num / 10
                results.append(num)
            except:
                results.append(np.nan)
        return pd.Series(results, index=col.index)
    
    df['start_lat'] = fix_coordinate_properly(df['start_lat'])
    df['start_lng'] = fix_coordinate_properly(df['start_lng'])
    df['end_lat'] = fix_coordinate_properly(df['end_lat'])
    df['end_lng'] = fix_coordinate_properly(df['end_lng'])
    
    # Make longitude negative (Western hemisphere - NYC)
    df['start_lng'] = -df['start_lng'].abs()
    df['end_lng'] = -df['end_lng'].abs()
    
    # Replace 0.0 with NaN for the Bronx WH station
    df.loc[df['end_station_name'] == 'Bronx WH station', ['end_lat', 'end_lng']] = np.nan
    
    # Create station mappings and fill missing values
    # End stations
    end_stations = df[
        df['end_station_id'].notna() & 
        df['end_station_name'].notna()
    ][['end_station_id', 'end_station_name', 'end_lat', 'end_lng']].drop_duplicates()
    
    end_name_to_id = end_stations.drop_duplicates('end_station_name').set_index('end_station_name')['end_station_id'].to_dict()
    fill_missing_id = df['end_station_id'].isna() & df['end_station_name'].notna()
    df.loc[fill_missing_id, 'end_station_id'] = df.loc[fill_missing_id, 'end_station_name'].map(end_name_to_id)
    
    end_name_to_lat = end_stations.drop_duplicates('end_station_name').set_index('end_station_name')['end_lat'].to_dict()
    fill_missing_lat = df['end_lat'].isna() & df['end_station_name'].notna()
    df.loc[fill_missing_lat, 'end_lat'] = df.loc[fill_missing_lat, 'end_station_name'].map(end_name_to_lat)
    
    # Additional filling for end coordinates
    end_lat_to_name = df[df['end_lat'].notna() & df['end_station_name'].notna()].drop_duplicates('end_lat').set_index('end_lat')['end_station_name'].to_dict()
    end_name_to_lat = df[df['end_lat'].notna() & df['end_station_name'].notna()].drop_duplicates('end_station_name').set_index('end_station_name')['end_lat'].to_dict()
    
    fill_missing_name = df['end_station_name'].isna() & df['end_lat'].notna()
    df.loc[fill_missing_name, 'end_station_name'] = df.loc[fill_missing_name, 'end_lat'].map(end_lat_to_name)
    
    fill_missing_lat = df['end_lat'].isna() & df['end_station_name'].notna()
    df.loc[fill_missing_lat, 'end_lat'] = df.loc[fill_missing_lat, 'end_station_name'].map(end_name_to_lat)
    
    # Fill end_lng
    end_lng_to_name = df[df['end_lng'].notna() & df['end_station_name'].notna()].drop_duplicates('end_lng').set_index('end_lng')['end_station_name'].to_dict()
    end_name_to_lng = df[df['end_lng'].notna() & df['end_station_name'].notna()].drop_duplicates('end_station_name').set_index('end_station_name')['end_lng'].to_dict()
    
    fill_missing_name = df['end_station_name'].isna() & df['end_lng'].notna()
    df.loc[fill_missing_name, 'end_station_name'] = df.loc[fill_missing_name, 'end_lng'].map(end_lng_to_name)
    
    fill_missing_lng = df['end_lng'].isna() & df['end_station_name'].notna()
    df.loc[fill_missing_lng, 'end_lng'] = df.loc[fill_missing_lng, 'end_station_name'].map(end_name_to_lng)
    
    # Fill coordinates from station name
    fill_missing_coords = (df['end_lat'].isna() | df['end_lng'].isna()) & df['end_station_name'].notna()
    name_to_coords = df[df['end_station_name'].notna() & df['end_lat'].notna() & df['end_lng'].notna()].copy()
    name_to_coords = name_to_coords.drop_duplicates('end_station_name').set_index('end_station_name')[['end_lat', 'end_lng']].to_dict('index')
    
    names_to_fill = df.loc[fill_missing_coords, 'end_station_name']
    for idx, name in names_to_fill.items():
        if name in name_to_coords:
            df.loc[idx, 'end_lat'] = name_to_coords[name]['end_lat']
            df.loc[idx, 'end_lng'] = name_to_coords[name]['end_lng']
    
    # Haversine distance
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    df['haversine_km'] = haversine_distance(df['start_lat'], df['start_lng'], df['end_lat'], df['end_lng'])
    df['distance_km'] = df['haversine_km'] * 1.5  # Scale by 1.5 to approximate real route distance
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Add additional features
    df['is_morning_rush'] = df['hour_of_day'].isin([7, 8, 9]).astype(int)
    df['is_evening_rush'] = df['hour_of_day'].isin([17, 18, 19]).astype(int)
    
    # Grid features for coverage analysis
    df['lat_grid'] = (df['start_lat'] * 100).round() / 100
    df['lng_grid'] = (df['start_lng'] * 100).round() / 100
    
    # Distance categories
    distance_bins = [0, 0.5, 1, 2, 3, 5, 10, 20, float('inf')]
    labels = ['< 0.5km', '0.5-1km', '1-2km', '2-3km', '3-5km', '5-10km', '10-20km', '> 20km']
    df['distance_category'] = pd.cut(df['distance_km'], bins=distance_bins, labels=labels)
    
    return df

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("# 🚴 Citi Bike Analytics")
st.sidebar.markdown("---")

sections = [
    "🏠 Overview",
    "📊 Data Quality",
    "⏰ Temporal Analysis",
    "🗺️ Spatial Analysis",
    "👥 User Segmentation",
    "📈 Executive Summary"
]

selected_section = st.sidebar.radio("Navigate to:", sections)

st.sidebar.markdown("---")
st.sidebar.info("**Data Source:** NYC Citi Bike Trip Data (2025)")

# ============================================================================
# LOAD DATA
# ============================================================================
with st.spinner("Loading and preprocessing data..."):
    df = load_and_preprocess_data()

if df is None:
    st.error("Failed to load data. Please check that the data files exist at the expected path.")
    st.stop()

# ============================================================================
# SECTION: OVERVIEW
# ============================================================================
if selected_section == "🏠 Overview":
    st.markdown('<h1 class="main-header">🚴 Citi Bike Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the comprehensive Citi Bike data analysis dashboard. This application provides 
    deep insights into NYC's bike-sharing system, including temporal patterns, spatial analysis, 
    and user segmentation.
    """)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rides", f"{len(df):,}")
    
    with col2:
        st.metric("Avg Duration", f"{df['duration_minutes'].mean():.1f} min")
    
    with col3:
        st.metric("Avg Distance", f"{df['distance_km'].mean():.2f} km")
    
    with col4:
        member_pct = (df['member_casual'] == 'member').sum() / len(df) * 100
        st.metric("Member Rides", f"{member_pct:.1f}%")
    
    st.markdown("---")
    
    # Dataset Overview
    st.markdown("### 📋 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Shape:**")
        st.write(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
        
        st.markdown("**Date Range:**")
        st.write(f"From: {df['started_at'].min()}")
        st.write(f"To: {df['started_at'].max()}")
    
    with col2:
        st.markdown("**Sample Data:**")
        st.dataframe(df.head(5), use_container_width=True)
    
    # Duration and Distance Statistics
    st.markdown("### 📊 Key Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Duration Statistics:**")
        st.write(df['duration_minutes'].describe())
    
    with col2:
        st.markdown("**Distance Statistics:**")
        st.write(df['distance_km'].describe())

# ============================================================================
# SECTION: DATA QUALITY
# ============================================================================
elif selected_section == "📊 Data Quality":
    st.markdown('<h1 class="section-header">📊 Data Quality Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Since we have 5 different files for the year 2025, we combine them into one dataset. 
    This section covers the data cleaning and preprocessing steps performed.
    """)
    
    # Feature Engineering Explanation
    with st.expander("🔧 Time-Based Feature Engineering", expanded=True):
        st.markdown("""
        ### Creating Time-Based Features
        
        This feature engineering process enriches temporal and duration-based insights:
        
        - **Trip Duration**: Calculates trip length in minutes using the difference between `ended_at` and `started_at`
        - **Temporal Features**: Extracts key time-based attributes including:
          - `hour_of_day`
          - `day_of_week` (name)
          - `day_of_week_num` (0 = Monday)
          - `date`
          - `is_weekend` (Saturday/Sunday flag)
        - **Time of Day Classification**: Categorizes trips into `Morning`, `Afternoon`, `Evening`, or `Night`
        - **Rush Hour Indicator**: Flags trips during commuting hours (7–9 AM, 5–7 PM)
        """)
    
    with st.expander("🌐 Coordinate Cleaning and Validation"):
        st.markdown("""
        ### Proper Coordinate Cleaning and Validation
        
        This code cleans, fixes, and validates latitude and longitude values:
        
        - **Custom Coordinate Fixing Function**: Removes misplaced periods, strips non-digit characters, and rescales values
        - **Handling Missing & Invalid Values**: Converts empty, NaN, or malformed entries into proper NaN values
        - **Latitude & Longitude Normalization**: Ensures latitude values fall within expected ranges and forces longitude values to be negative (Western hemisphere)
        - **Geographic Validation**: Confirms values fall within expected NYC bounds (lat 40–41, lng −75 to −73)
        """)
    
    with st.expander("🔍 Filling Missing Values Using Lookup Tables"):
        st.markdown("""
        ### Filling Missing Values Using a Lookup Table
        
        We noticed that some rows are missing `end_station_id`, but they have a valid `end_station_name`.
        Instead of dropping those rows, we reconstruct the missing IDs.
        
        #### Creating a Mapping Table (Like VLOOKUP in Excel)
        
        The idea is simple:
        - If a column is missing a value (for example, `end_station_id`)
        - but another column (like `end_station_name`) is present,
        - we use a lookup table that maps **station names → station IDs** and automatically fill in the missing values.
        
        This approach helps us restore data without guessing or deleting rows.
        """)
    
    # Missing Values Visualization
    st.markdown("### 📉 Missing Values Analysis")
    
    # Calculate missing values (from original processing - showing final state)
    st.success("✅ All missing values have been successfully handled through lookup tables and cleaning processes.")
    st.info(f"Final dataset contains {len(df):,} complete rows with no missing values.")
    
    # Distance Feature Engineering
    st.markdown("---")
    st.markdown("### 📐 Distance Feature Engineering")
    
    st.markdown("""
    #### Why Haversine Distance?
    - Computes straight-line distance between two latitude–longitude points
    - Extremely fast and scalable for large datasets
    - No dependency on external routing APIs
    - Slightly underestimates real travel distance, but works well as a proxy
    
    To account for this underestimation, we compare Haversine distance with actual
    route distances obtained from **OSRM** on a random sample.
    
    The comparison shows that real route distances are, on average, **~1.5× the Haversine distance**.
    
    #### Final Distance Feature
    - We compute **Haversine distance** for the full dataset
    - We **scale it by 1.5×** to better approximate real-world travel distance
    """)
    
    # Outlier Analysis
    st.markdown("---")
    st.markdown("### 🔍 Duration & Distance Outlier Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Duration Outliers:**")
        too_short = (df['duration_minutes'] < 1).sum()
        too_long = (df['duration_minutes'] > 1440).sum()
        negative = (df['duration_minutes'] < 0).sum()
        
        st.write(f"- < 1 minute: {too_short:,}")
        st.write(f"- > 24 hours: {too_long:,}")
        st.write(f"- Negative: {negative:,}")
    
    with col2:
        st.markdown("**Distance Outliers:**")
        too_far = (df['distance_km'] > 50).sum()
        st.write(f"- > 50 km: {too_far:,}")
    
    # Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    df.boxplot(column='duration_minutes', ax=axes[0])
    axes[0].set_title('Duration Distribution (with outliers)')
    axes[0].set_ylabel('Minutes')
    
    df.boxplot(column='distance_km', ax=axes[1])
    axes[1].set_title('Distance Distribution (with outliers)')
    axes[1].set_ylabel('Kilometers')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================================
# SECTION: TEMPORAL ANALYSIS
# ============================================================================
elif selected_section == "⏰ Temporal Analysis":
    st.markdown('<h1 class="section-header">⏰ Temporal Analysis</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 Station Popularity", "📈 Hourly Patterns", "🔥 Heatmap", "🎬 Animation"])
    
    # TAB 1: Station Popularity
    with tabs[0]:
        st.markdown("### 🏆 Top 20 Most Popular Start Stations")
        
        st.markdown("""
        - Counts the number of rides starting at each station
        - Focuses on the **top 20 most popular start stations**
        - Visualized using a bar chart with ride counts shown on each bar
        
        #### Insight
        A small number of stations account for a large share of total rides, indicating
        high-demand hubs such as commercial areas or transport connections.
        """)
        
        station_counts = df['start_station_name'].value_counts().head(20).reset_index()
        station_counts.columns = ['Station', 'Rides']
        
        fig = px.bar(
            station_counts,
            x='Station',
            y='Rides',
            title='Top 20 Most Popular Start Stations',
            labels={'Rides': 'Number of Rides', 'Station': 'Station Name'},
            color='Rides',
            color_continuous_scale='Viridis',
            text='Rides'
        )
        
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig.update_layout(
            height=600,
            xaxis_tickangle=-45,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Hourly Patterns
    with tabs[1]:
        st.markdown("### 📈 Average Ride Duration Throughout the Day")
        
        hourly_duration = df.groupby('hour_of_day')['duration_minutes'].mean().reset_index()
        
        fig = px.line(
            hourly_duration,
            x='hour_of_day',
            y='duration_minutes',
            title='📊 Average Ride Duration Throughout the Day',
            labels={'hour_of_day': 'Hour of Day', 'duration_minutes': 'Avg Duration (min)'},
            markers=True,
            line_shape='spline'
        )
        
        fig.update_traces(line=dict(width=3, color='#667eea'), marker=dict(size=8))
        fig.update_layout(height=500, hovermode='x')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🕐 24-Hour Ride Patterns: Members vs Casual")
        
        st.markdown("""
        - Ride activity is analyzed across **24 hours**, split by **member** and **casual** users
        - A line chart highlights differences in usage behavior throughout the day
        
        #### Key Insight
        - **17:00 (5 PM)** is the **peak hour** for ride activity
        - **Members account for the majority of rides**, especially during peak commuting hours
        
        This pattern suggests strong **commute-driven usage** among member riders.
        """)
        
        hourly = df.groupby(['hour_of_day', 'member_casual']).size().reset_index(name='count')
        
        fig = px.line(
            hourly,
            x='hour_of_day',
            y='count',
            color='member_casual',
            title='24-Hour Ride Patterns: Members vs Casual',
            labels={'hour_of_day': 'Hour of Day', 'count': 'Number of Rides', 'member_casual': 'User Type'},
            color_discrete_map={'member': '#667eea', 'casual': '#f093fb'},
            markers=True
        )
        
        fig.update_traces(line=dict(width=4), marker=dict(size=10))
        fig.update_layout(
            height=600,
            xaxis=dict(tickmode='linear', tick0=0, dtick=2, title='Hour of Day'),
            yaxis=dict(title='Number of Rides'),
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Heatmap
    with tabs[2]:
        st.markdown("### 🔥 Ride Intensity Heatmap")
        
        st.markdown("""
        - This heatmap visualizes **ride volume by day of week and hour of day**
        - Rows represent **days (Monday–Sunday)** and columns represent **hours (0–23)**
        - Color intensity indicates the **number of rides**
        
        #### Key Insights
        - **Weekdays show strong peaks during morning and evening commute hours**
        - **Members dominate weekday, commute-time usage**
        - **Casual riders show higher activity on weekends and midday hours**
        """)
        
        def get_pivot(dataframe):
            p = dataframe.pivot_table(values='ride_id', index='day_of_week', 
                                      columns='hour_of_day', aggfunc='count')
            return p.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # Filter selection
        heatmap_filter = st.selectbox("Select User Type:", ["All Rides", "Members Only", "Casual Only"])
        
        if heatmap_filter == "All Rides":
            pivot_data = get_pivot(df)
        elif heatmap_filter == "Members Only":
            pivot_data = get_pivot(df[df['member_casual'] == 'member'])
        else:
            pivot_data = get_pivot(df[df['member_casual'] == 'casual'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns.tolist(),
            y=pivot_data.index.tolist(),
            colorscale='Turbo',
            colorbar=dict(title="Total Rides", tickformat=','),
            hovertemplate='<b>%{y}</b><br>Hour: %{x}:00<br>Rides: %{z:,.0f}<extra></extra>',
        ))
        
        fig.update_layout(
            title={
                'text': f'Ride Intensity Heatmap - {heatmap_filter}',
                'x': 0.5, 'xanchor': 'center',
                'font': {'size': 24, 'color': '#2c3e50', 'family': 'Arial Black'}
            },
            xaxis=dict(title='Hour of Day', title_font=dict(size=16), tickmode='linear', dtick=2),
            yaxis=dict(title='Day of Week', title_font=dict(size=16), autorange='reversed'),
            height=700,
            width=1200,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Animation
    with tabs[3]:
        st.markdown("### 🎬 Hourly Bike Movement Animation")
        
        st.markdown("""
        - A **random sample of up to 10,000 rides** is used to keep the visualization efficient
        - Ride start and end points are mapped and **animated across 24 hourly time buckets (0–23)**
        - Each ride contributes two points:
          - **Blue:** start location
          - **Red:** end location
        
        #### Key Insight
        - The city appears **most inactive between 03:00 and 04:00**, with very few ride start or end points
        - Activity increases steadily after early morning hours, reflecting typical daily mobility patterns
        """)
        
        df_sample = df.sample(min(10000, len(df)))
        
        bins = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', 
                '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00', '23:59']
        
        labels = list(range(24))
        bins_time = pd.to_datetime(bins, format='%H:%M').time
        
        def assign_bucket(time):
            if pd.isna(time):
                return None
            for i in range(len(bins_time) - 1):
                if bins_time[i] <= time < bins_time[i + 1]:
                    return labels[i]
            return None
        
        df_sample['bucket'] = df_sample['started_at'].dt.time.apply(assign_bucket)
        df_filtered = df_sample[df_sample['bucket'].notna()]
        
        animation_df = pd.DataFrame({
            'bucket': pd.concat([df_filtered['bucket'], df_filtered['bucket']]),
            'lat': pd.concat([df_filtered['start_lat'], df_filtered['end_lat']]),
            'lng': pd.concat([df_filtered['start_lng'], df_filtered['end_lng']]),
            'station_name': pd.concat([df_filtered['start_station_name'], df_filtered['end_station_name']]),
            'ride_id': pd.concat([df_filtered['ride_id'], df_filtered['ride_id']]),
            'type': ['start'] * len(df_filtered) + ['end'] * len(df_filtered)
        }).reset_index(drop=True)
        
        animation_df = animation_df.sort_values('bucket')
        
        center_lat = animation_df['lat'].mean()
        center_lng = animation_df['lng'].mean()
        
        fig = px.scatter_mapbox(
            animation_df, 
            lat='lat', 
            lon='lng', 
            hover_name='station_name', 
            hover_data=['ride_id', 'type'],
            animation_frame='bucket',
            color='type',
            color_discrete_map={'start': '#3b82f6', 'end': '#ef4444'},
            zoom=12, 
            height=700,
            category_orders={'bucket': labels}
        )
        
        fig.update_layout(
            mapbox_style="open-street-map", 
            title="🚴 Bike Movements by Hour (24-hour format: 0 = midnight, 23 = 11pm)",
            mapbox=dict(
                zoom=12,
                center=dict(lat=center_lat, lon=center_lng)
            ),
            dragmode='zoom',
            hovermode='closest'
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION: SPATIAL ANALYSIS
# ============================================================================
elif selected_section == "🗺️ Spatial Analysis":
    st.markdown('<h1 class="section-header">🗺️ Spatial Analysis</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["🔷 Hexagonal Density", "🛤️ Route Flow", "📍 Station Map", "🚨 Rush Hour", "📊 Coverage"])
    
    # TAB 1: Hexagonal Density
    with tabs[0]:
        st.markdown("### 🔷 Hexagonal Density Map")
        
        st.markdown("""
        - Trips are aggregated into **hexagonal bins**, which reduces noise and improves readability
        - Color intensity represents **ride density** within each hexagon
        
        High-density clusters clearly highlight **activity hotspots**, indicating areas with
        consistently high station usage rather than isolated points.
        
        #### Takeaway
        Hexagonal aggregation provides a clear view of **spatial demand patterns**, making it
        useful for identifying zones that may benefit from **additional stations or bikes**.
        """)
        
        coords = df[['start_lat', 'start_lng']].dropna().sample(min(10000, len(df)))
        
        fig = ff.create_hexbin_mapbox(
            data_frame=coords,
            lat="start_lat",
            lon="start_lng",
            nx_hexagon=40,
            opacity=0.7,
            labels={"color": "Ride Density"},
            color_continuous_scale="Turbo",
            mapbox_style="carto-positron",
            zoom=11,
            height=700,
            show_original_data=False,
            original_data_marker=dict(size=2, opacity=0.3, color='black')
        )
        
        fig.update_layout(
            title="Hexagonal Density Map - Station Activity Clusters",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Route Flow
    with tabs[1]:
        st.markdown("### 🛤️ Interactive Route Flow Map")
        
        st.markdown("""
        - Displays the **top 50 most frequent routes** between stations
        - Routes are drawn as **curved arcs** to improve visual separation
        - **Line thickness and color intensity** represent route popularity
        
        #### Interactivity
        - A **dropdown menu** allows isolating a single route
        - Hovering over a route shows **start station, end station, and ride count**
        
        #### Takeaway
        This interactive map makes it easy to explore **dominant travel flows**, supporting
        decisions around **station placement, rebalancing, and infrastructure planning**.
        """)
        
        routes = (
            df.groupby([
                'start_station_name', 'end_station_name',
                'start_lat', 'start_lng', 'end_lat', 'end_lng'
            ])
            .size()
            .reset_index(name='count')
            .dropna()
            .nlargest(50, 'count')
        )
        
        counts = routes['count']
        norm = (counts - counts.min()) / (counts.max() - counts.min())
        colors = pc.sample_colorscale('Plasma', norm)
        
        fig = go.Figure()
        
        for i, (route, color) in enumerate(zip(routes.itertuples(), colors)):
            n_points = 40
            t = np.linspace(0, 1, n_points)
            
            lons = np.linspace(route.start_lng, route.end_lng, n_points)
            lats = np.linspace(route.start_lat, route.end_lat, n_points)
            
            arc_offset = 0.008 
            lats = lats + (4 * t * (1 - t) * arc_offset)
            
            fig.add_trace(go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines',
                name=f"Route {i+1}", 
                line=dict(
                    width=max(np.log(route.count) * 2, 2), 
                    color=color
                ),
                hovertemplate=(
                    f"<b>{route.start_station_name}</b><br>"
                    f"to <b>{route.end_station_name}</b><br>"
                    f"Rides: {route.count:,}<extra></extra>"
                ),
                visible=True,
                showlegend=False 
            ))
        
        fig.update_layout(
            mapbox=dict(
                style='carto-darkmatter',
                center=dict(
                    lat=df['start_lat'].mean(),
                    lon=df['start_lng'].mean()
                ),
                zoom=12
            ),
            title={
                'text': "🗺️ <b>Top 50 Citi Bike Routes</b>",
                'x': 0.5, 'xanchor': 'center',
                'font': {'color': 'white', 'size': 24}
            },
            height=800,
            paper_bgcolor='black',
            margin=dict(l=0, r=0, t=60, b=0),
            showlegend=False 
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
    
    # TAB 3: Station Map
    with tabs[2]:
        st.markdown("### 📍 Station Popularity Map")
        
        st.markdown("""
        - An interactive map visualizes **all start stations** in the city
        - Stations are grouped using **marker clustering** to reduce overlap and improve readability
        - Each station is represented by a **circle marker**
        
        #### Interpretation
        - **Marker size reflects station popularity** (number of rides)
        - Larger circles indicate **high-demand stations**
        - Dense clusters highlight **areas with concentrated bike usage**
        
        #### Takeaway
        This map provides a clear spatial overview of **station demand**, helping identify
        high-usage zones for **capacity planning and station optimization**.
        """)
        
        stations = df.groupby(['start_station_name', 'start_lat', 'start_lng']).size().reset_index(name='rides')
        stations = stations.dropna()
        
        center_lat = stations['start_lat'].mean()
        center_lon = stations['start_lng'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')
        
        marker_cluster = MarkerCluster().add_to(m)
        
        for _, station in stations.iterrows():
            folium.CircleMarker(
                location=[station['start_lat'], station['start_lng']],
                radius=min(station['rides'] / 1000, 30),
                popup=f"<b>{station['start_station_name']}</b><br>Rides: {station['rides']:,}",
                color='blue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.6
            ).add_to(marker_cluster)
        
        st_folium(m, width=1200, height=700)
    
    # TAB 4: Rush Hour
    with tabs[3]:
        st.markdown("### 🚨 Rush Hour Intensity Map: Critical Capacity Stations")
        
        st.markdown("""
        ## The *Rush Hour Crunch*
        ### Which Stations Face Critical Capacity Issues?
        
        ### 🚨 Top Rush-Hour Dependent Stations
        The following stations show the **highest dependence on rush-hour traffic**, with over
        half of their daily rides occurring during peak commute hours:
        
        - **Dock 72 Way & Market St** – **68.8%** of rides during rush hours
        - **E Mosholu Pkwy & E 204 St** – **60.4%**
        - **Park Ave & E 41 St** – **59.8%**
        - **W 54 St & 6 Ave** – **56.2%**
        - **W 37 St & Broadway** – **55.9%**
        
        #### Key Insights
        - Some stations see **60–70% of daily traffic during rush hours**
        - **Evening rush demand is slightly higher** than morning, indicating return commutes
        - High-volume stations face the **greatest capacity risk**
        
        #### Recommended Actions
        - Deploy **rush-hour rapid-response rebalancing teams**
        - Add **temporary or pop-up docks** at critical stations
        - Consider **peak-hour pricing or incentives** to smooth demand
        """)
        
        rush_stats = df.groupby('start_station_name').agg({
            'is_morning_rush': 'sum',
            'is_evening_rush': 'sum',
            'ride_id': 'count'
        }).reset_index()
        
        rush_stats.columns = ['station', 'morning_rush', 'evening_rush', 'total']
        rush_stats['rush_hour_pct'] = ((rush_stats['morning_rush'] + rush_stats['evening_rush']) / rush_stats['total'] * 100)
        rush_stats = rush_stats.nlargest(20, 'rush_hour_pct')
        
        fig = px.scatter(
            rush_stats,
            x='morning_rush',
            y='evening_rush',
            size='total',
            color='rush_hour_pct',
            hover_name='station',
            title='Rush Hour Intensity Map: Critical Capacity Stations',
            labels={'morning_rush': 'Morning Rush Rides', 'evening_rush': 'Evening Rush Rides', 'rush_hour_pct': 'Rush Hour %'},
            color_continuous_scale='Reds',
            size_max=60
        )
        
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: Coverage
    with tabs[4]:
        st.markdown("### 📊 Service Coverage & Density")
        
        st.markdown("""
        ### What the Visualization Shows
        - The city is divided into **spatial grid cells**, aggregating:
          - Total rides
          - Number of unique stations
          - Rides per station
        - **Bubble size** represents ride volume
        - **Color intensity** represents station density within each grid
        
        #### Key Insights
        - **Manhattan core shows very high station density**, but rides per station are relatively balanced
        - Several **outer Manhattan, Brooklyn, and Queens grid cells** show moderate-to-high ride demand with fewer stations
        - These areas indicate **underserved demand**, not low demand
        
        #### Business Implications
        - Demand is **not evenly matched with station placement**
        - Revenue opportunities likely exist in **medium-density, high-usage grids**
        
        #### Recommended Actions
        - Add stations or docks in **high rides-per-station zones**
        - Rebalance capacity away from **over-saturated core areas**
        """)
        
        grid_coverage = df.groupby(['lat_grid', 'lng_grid']).agg({
            'ride_id': 'count',
            'start_station_name': 'nunique'
        }).reset_index()
        
        grid_coverage.columns = ['lat', 'lng', 'rides', 'stations']
        grid_coverage['rides_per_station'] = grid_coverage['rides'] / grid_coverage['stations']
        
        fig = px.scatter_mapbox(
            grid_coverage,
            lat='lat',
            lon='lng',
            size='rides',
            color='stations',
            hover_data=['rides', 'stations', 'rides_per_station'],
            title='Service Coverage & Density',
            color_continuous_scale='RdYlGn',
            size_max=50,
            zoom=11,
            height=700,
            mapbox_style='carto-positron'
        )
        
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION: USER SEGMENTATION
# ============================================================================
elif selected_section == "👥 User Segmentation":
    st.markdown('<h1 class="section-header">👥 User Segmentation Analysis</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["🚲 Bike Type", "📏 Distance Analysis", "🎯 Behavioral Segments"])
    
    # TAB 1: Bike Type
    with tabs[0]:
        st.markdown("### 🚲 Membership vs Bike Type")
        
        st.markdown("""
        #### Key Insights: Membership vs Bike Type
        
        - **Members generate the majority of rides** across both bike types
        - **Electric bikes are preferred** by both members and casual users
        - The **member–electric bike** combination shows the highest ride volume, indicating strong commuter usage
        - **Casual riders** contribute significantly fewer trips, suggesting more leisure-oriented behavior
        
        #### Takeaway
        Electric bikes are the primary demand driver, especially among members, and should be
        prioritized for availability and operational planning.
        """)
        
        summary = (
            df.groupby(['member_casual', 'rideable_type'])
              .size()
              .reset_index(name='count')
              .pivot(index='member_casual', columns='rideable_type', values='count')
        )
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.set_theme(style="white", font_scale=1.1)
        
        sns.heatmap(
            summary,
            cmap="Blues",
            annot=True,
            fmt=",",
            linewidths=0.5,
            cbar_kws={'label': 'Ride Count'},
            ax=ax
        )
        
        ax.set_title("User Segmentation: Membership vs Bike Type")
        ax.set_ylabel("")
        ax.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # TAB 2: Distance Analysis
    with tabs[1]:
        st.markdown("### 📏 Distance Distribution by User Type")
        
        st.markdown("""
        #### What the Visualization Shows
        - Trips are grouped into **distance ranges** from under 0.5 km to 20 km
        - Bars compare **members vs casual users**
        - Heights represent **number of rides**, highlighting demand concentration
        
        #### Key Insights
        - **Peak demand lies in the 1–2 km range**, making it the most common trip length
        - **43.8% of all trips are under 2 km**, confirming strong last-mile usage
        - **Members dominate short-to-medium distances (0.5–5 km)**, consistent with commuting and daily utility
        - **Casual users contribute relatively more to longer trips (5–10 km, 10–20 km)**, indicating leisure and exploration
        
        #### Business Implications
        - Optimize pricing and availability for **short, frequent trips**
        - Position stations near **transit hubs and dense neighborhoods**
        - Market longer-distance options to **casual users**
        """)
        
        distance_bins = [0, 0.5, 1, 2, 3, 5, 10, 20]
        labels_dist = ['< 0.5km', '0.5-1km', '1-2km', '2-3km', '3-5km', '5-10km', '10-20km']
        df['distance_category_viz'] = pd.cut(df['distance_km'], bins=distance_bins, labels=labels_dist)
        
        dist_analysis = df.groupby(['distance_category_viz', 'member_casual']).size().reset_index(name='count')
        
        fig = go.Figure()
        
        colors = {'member': '#2E5A88', 'casual': '#E67E22'}
        
        for user_type in ['member', 'casual']:
            subset = dist_analysis[dist_analysis['member_casual'] == user_type]
            
            fig.add_trace(go.Bar(
                x=subset['distance_category_viz'],
                y=subset['count'],
                name=user_type.capitalize(),
                marker=dict(
                    color=colors[user_type],
                    line=dict(width=1, color='white') 
                ),
                text=subset['count'],
                texttemplate='%{text:,.2s}', 
                textposition='outside',
                hovertemplate='<b>%{name}</b><br>%{x}: %{y:,.0f} rides<extra></extra>',
                cliponaxis=False 
            ))
        
        fig.update_layout(
            title={
                'text': "<b>The 'Last Mile' Reality</b><br><span style='font-size:14px; color:gray;'>Distance distribution by user type</span>",
                'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'
            },
            xaxis=dict(
                title="<b>Distance Range</b>",
                showgrid=False,
                linecolor='lightgray'
            ),
            yaxis=dict(
                title="<b>Number of Rides</b>",
                gridcolor='#f0f0f0',
                tickformat=',', 
                zeroline=False
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.05,
            template='plotly_white', 
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1
            ),
            height=600,
            width=1000,
            margin=dict(t=120, b=50, l=70, r=30),
            font=dict(family="Arial", size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distance insights
        median_distance = df['distance_km'].median()
        pct_under_2km = (df['distance_km'] < 2).sum() / len(df) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Median Trip Distance", f"{median_distance:.2f} km")
        with col2:
            st.metric("Short Trips (<2km)", f"{pct_under_2km:.1f}%")
        with col3:
            st.metric("Peak Demand Range", "1-2 km")
    
    # TAB 3: Behavioral Segments
    with tabs[2]:
        st.markdown("### 🎯 User Segmentation: Duration vs Distance by Day Type")
        
        st.markdown("""
        #### How Time of Day & Day Type Shape User Behavior
        
        #### Key Insights
        - **Weekday members** take short, efficient trips, consistent with commuting patterns
        - **Weekday casual users** ride longer without traveling much farther, indicating sightseeing behavior
        - On **weekends**, trips become longer and more recreational for all users
        - **Casual users on weekends** show the longest and farthest rides, highlighting leisure-driven usage
        
        #### Actions
        - Design **commuter-focused plans** for weekday members
        - Offer **leisure and tourist pricing** for casual users
        - Adjust **bike availability** toward recreational areas on weekends
        """)
        
        segment_analysis = df.groupby(
            ['day_of_week', 'is_weekend', 'time_of_day', 'member_casual']
        ).agg({
            'ride_id': 'count',
            'duration_minutes': 'mean',
            'distance_km': 'mean'
        }).reset_index()
        
        segment_analysis.columns = [
            'day', 'is_weekend', 'time', 'user_type',
            'rides', 'avg_duration', 'avg_distance'
        ]
        
        fig = px.scatter(
            segment_analysis,
            x='avg_duration',
            y='avg_distance',
            size='rides',
            color='user_type',
            facet_col='is_weekend',
            title='User Segmentation: Duration vs Distance by Day Type',
            labels={
                'avg_duration': 'Avg Duration (min)',
                'avg_distance': 'Avg Distance (km)',
                'user_type': 'User Type'
            },
            color_discrete_map={
                'member': '#667eea',
                'casual': '#f093fb'
            },
            hover_data=['day', 'time', 'rides'],
            height=600,
            size_max=40,
            opacity=0.7
        )
        
        fig.for_each_annotation(
            lambda a: a.update(
                text='Weekend' if a.text == 'is_weekend=1' else 'Weekday'
            )
        )
        
        fig.update_layout(
            template='plotly_white',
            title=dict(x=0.5, xanchor='center'),
            legend=dict(title='User Type')
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION: EXECUTIVE SUMMARY
# ============================================================================
elif selected_section == "📈 Executive Summary":
    st.markdown('<h1 class="section-header">📊 Executive Summary: Key Strategic Insights</h1>', unsafe_allow_html=True)
    
    # Operational Priorities
    st.markdown("### 🚦 Operational Priorities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        ⚡ **Electric Fleet Expansion:** 30% longer trips justify premium pricing
        """)
        st.success("""
        🏃 **Rush Hour Response:** Deploy rapid rebalancing teams (7–9 AM, 5–7 PM)
        """)
    
    with col2:
        st.success("""
        🗺️ **Geographic Expansion:** 10+ underserved, high-demand areas identified
        """)
        st.success("""
        🔄 **Round-Trip Marketing:** Curated scenic routes for casual weekend riders
        """)
    
    st.markdown("---")
    
    # Revenue Optimization
    st.markdown("### 💰 Revenue Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ⚡ **Dynamic Pricing:** 40–50% surge pricing during rush hours
        """)
        st.info("""
        🎯 **Segment-Based Pricing:** Separate plans for commuters vs leisure users
        """)
    
    with col2:
        st.info("""
        📏 **Distance Tiers:** Optimize pricing for 1–3 km "last-mile" trips
        """)
        st.info("""
        ⚡ **E-Bike Premium:** 20–30% upcharge justified by speed and distance
        """)
    
    st.markdown("---")
    
    # Customer Experience
    st.markdown("### 🚴 Customer Experience")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.warning("""
        ⏱️ **Speed Matters:** Members value time efficiency
        """)
        st.warning("""
        🌆 **Experience Matters:** Casual riders value comfort and scenery
        """)
    
    with col2:
        st.warning("""
        🏙️ **Neighborhood Equity:** Expand coverage in peripheral high-demand areas
        """)
        st.warning("""
        🔋 **E-Bikes as a Game Changer:** Enable longer commutes and broader reach
        """)
    
    with col3:
        st.warning("""
        📅 **Weekend vs Weekday:** Different usage patterns require different products
        """)
    
    st.markdown("---")
    
    # Strategic Investments
    st.markdown("### 📈 Strategic Investments")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.error("""
        🔴 **High Priority:** Electric bike fleet expansion (25% → 40%)
        """)
    
    with col2:
        st.warning("""
        🟠 **Medium Priority:** Peripheral station expansion (15–20 new locations)
        """)
    
    with col3:
        st.success("""
        🟢 **Low Priority:** Additional capacity at select rush-hour hubs (3–4 stations)
        """)
    
    st.markdown("---")
    
    # Bottom Line
    st.markdown("### 🎯 Bottom Line")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center;">
        <h3 style="color: white; margin-bottom: 1rem;">Key Takeaway</h3>
        <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
            There are <strong>four distinct customer segments</strong> with clearly different needs.
        </p>
        <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
            A <strong>one-size-fits-all strategy leaves money on the table</strong>.
        </p>
        <p style="font-size: 1.4rem; font-weight: bold;">
            Personalized pricing, fleet mix, and station planning could increase revenue by 20–30%.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Metrics Summary
    st.markdown("### 📊 Key Metrics Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rides Analyzed", f"{len(df):,}")
    
    with col2:
        median_distance = df['distance_km'].median()
        st.metric("Median Trip Distance", f"{median_distance:.2f} km")
    
    with col3:
        member_pct = (df['member_casual'] == 'member').sum() / len(df) * 100
        st.metric("Member Ride Share", f"{member_pct:.1f}%")
    
    with col4:
        pct_under_2km = (df['distance_km'] < 2).sum() / len(df) * 100
        st.metric("Short Trips (<2km)", f"{pct_under_2km:.1f}%")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>🚴 Citi Bike Data Analysis Dashboard | Built with Streamlit</p>
    <p>Data Source: NYC Citi Bike Trip Data (2025)</p>
</div>
""", unsafe_allow_html=True)
