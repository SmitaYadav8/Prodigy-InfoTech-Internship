"""
Traffic Accident Data Analysis
Analyze patterns related to road conditions, weather, and time of day
Visualize accident hotspots and contributing factors
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Setup directories
os.makedirs('../data', exist_ok=True)
os.makedirs('../output', exist_ok=True)

def load_accident_data():
    """Load and preprocess accident data"""
    # Sample accident data structure (replace with actual data source)
    data = {
        'timestamp': pd.date_range('2023-01-01', '2023-12-31', freq='H')[:1000],
        'latitude': np.random.uniform(28.4, 28.7, 1000),
        'longitude': np.random.uniform(77.0, 77.3, 1000),
        'weather': np.random.choice(['Clear', 'Rain', 'Fog', 'Snow', 'Windy'], 1000, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
        'road_condition': np.random.choice(['Dry', 'Wet', 'Icy', 'Snowy', 'Flooded'], 1000, p=[0.7, 0.2, 0.05, 0.03, 0.02]),
        'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], 1000, p=[0.3, 0.2, 0.3, 0.2]),
        'severity': np.random.choice(['Minor', 'Moderate', 'Severe', 'Fatal'], 1000, p=[0.5, 0.3, 0.15, 0.05]),
        'cause': np.random.choice(['Speeding', 'Distraction', 'Weather', 'Road Condition', 'Vehicle Failure'], 1000)
    }
    
    df = pd.DataFrame(data)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month_name()
    
    return df

def plot_time_analysis(df):
    """Analyze accidents by time patterns"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Hourly distribution
    hourly_counts = df['hour'].value_counts().sort_index()
    ax1.bar(hourly_counts.index, hourly_counts.values, color='#e74c3c')
    ax1.set_title('Accidents by Hour of Day', fontsize=14, pad=15)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Number of Accidents')
    ax1.grid(True, alpha=0.3)
    
    # Day of week distribution
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().reindex(day_order)
    ax2.bar(range(len(day_counts)), day_counts.values, color='#3498db')
    ax2.set_title('Accidents by Day of Week', fontsize=14, pad=15)
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Number of Accidents')
    ax2.set_xticks(range(len(day_counts)))
    ax2.set_xticklabels(day_counts.index, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Time of day distribution
    time_counts = df['time_of_day'].value_counts()
    ax3.pie(time_counts.values, labels=time_counts.index, autopct='%1.1f%%',
            colors=['#f39c12', '#2ecc71', '#9b59b6', '#34495e'])
    ax3.set_title('Accidents by Time of Day', fontsize=14, pad=15)
    
    # Monthly distribution
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    month_counts = df['month'].value_counts().reindex(month_order)
    ax4.plot(month_counts.index, month_counts.values, marker='o', color='#27ae60')
    ax4.set_title('Accidents by Month', fontsize=14, pad=15)
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Number of Accidents')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../output/time_of_day_accidents.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_weather_analysis(df):
    """Analyze accidents by weather conditions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Weather conditions
    weather_counts = df['weather'].value_counts()
    ax1.bar(weather_counts.index, weather_counts.values, color=['#f1c40f', '#3498db', '#95a5a6', '#ecf0f1', '#e67e22'])
    ax1.set_title('Accidents by Weather Conditions', fontsize=14, pad=15)
    ax1.set_xlabel('Weather Condition')
    ax1.set_ylabel('Number of Accidents')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Weather vs severity
    weather_severity = pd.crosstab(df['weather'], df['severity'])
    weather_severity.plot(kind='bar', ax=ax2, color=['#27ae60', '#f39c12', '#e74c3c', '#c0392b'])
    ax2.set_title('Accident Severity by Weather Conditions', fontsize=14, pad=15)
    ax2.set_xlabel('Weather Condition')
    ax2.set_ylabel('Number of Accidents')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Severity')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../output/weather_conditions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_road_condition_analysis(df):
    """Analyze accidents by road conditions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Road conditions
    road_counts = df['road_condition'].value_counts()
    ax1.bar(road_counts.index, road_counts.values, color=['#27ae60', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
    ax1.set_title('Accidents by Road Conditions', fontsize=14, pad=15)
    ax1.set_xlabel('Road Condition')
    ax1.set_ylabel('Number of Accidents')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Road condition vs weather
    road_weather = pd.crosstab(df['road_condition'], df['weather'])
    road_weather.plot(kind='bar', ax=ax2, color=['#f1c40f', '#3498db', '#95a5a6', '#ecf0f1', '#e67e22'])
    ax2.set_title('Road Conditions vs Weather', fontsize=14, pad=15)
    ax2.set_xlabel('Road Condition')
    ax2.set_ylabel('Number of Accidents')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Weather')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../output/road_conditions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_hotspots(df):
    """Visualize accident hotspots"""
    plt.figure(figsize=(12, 8))
    
    # Create heatmap of accident locations
    plt.hexbin(df['longitude'], df['latitude'], gridsize=30, cmap='Reds', alpha=0.8)
    plt.colorbar(label='Number of Accidents')
    
    # Add some mock landmarks for context
    plt.scatter([77.1, 77.2, 77.15], [28.5, 28.6, 28.55], c='blue', s=100, marker='^', label='Major Intersections')
    
    plt.title('Accident Hotspots Map', fontsize=16, pad=20)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('../output/accident_hotspots.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_contributing_factors(df):
    """Analyze contributing factors to accidents"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Primary causes
    cause_counts = df['cause'].value_counts()
    ax1.pie(cause_counts.values, labels=cause_counts.index, autopct='%1.1f%%',
            colors=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'])
    ax1.set_title('Primary Causes of Accidents', fontsize=14, pad=15)
    
    # Cause vs severity
    cause_severity = pd.crosstab(df['cause'], df['severity'])
    cause_severity.plot(kind='bar', ax=ax2, color=['#27ae60', '#f39c12', '#e74c3c', '#c0392b'])
    ax2.set_title('Accident Severity by Cause', fontsize=14, pad=15)
    ax2.set_xlabel('Cause')
    ax2.set_ylabel('Number of Accidents')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Severity')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../output/contributing_factors.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function"""
    print("Loading and analyzing traffic accident data...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    
    # Load data
    accident_df = load_accident_data()
    
    print(f"Analyzing {len(accident_df)} accident records...")
    
    # Generate all visualizations
    plot_time_analysis(accident_df)
    plot_weather_analysis(accident_df)
    plot_road_condition_analysis(accident_df)
    plot_hotspots(accident_df)
    plot_contributing_factors(accident_df)
    
    print("Analysis complete! Visualizations saved to /output folder")
    
    # Print key insights
    print("\nKey Insights:")
    print(f"1. Most accidents occur during: {accident_df['time_of_day'].mode().values[0]}")
    print(f"2. Most common weather condition: {accident_df['weather'].mode().values[0]}")
    print(f"3. Most common road condition: {accident_df['road_condition'].mode().values[0]}")
    print(f"4. Primary cause: {accident_df['cause'].mode().values[0]}")

if __name__ == "__main__":
    main()
