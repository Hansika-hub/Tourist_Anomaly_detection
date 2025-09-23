
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def generate_mock_data(n_samples=1000):
    """
    Generate mock tourist GPS + timestamp data with some anomalies
    """
    np.random.seed(42)
    
    # Generate base timestamps (every 5 minutes over 3 days)
    start_time = datetime.now() - timedelta(days=3)
    timestamps = [start_time + timedelta(minutes=i*5) for i in range(n_samples)]
    
    # Generate base GPS coordinates (walking around a tourist area)
    base_lat = 40.7128  # NYC coordinates
    base_lon = -74.0060
    
    # Normal movement pattern
    lat_offsets = np.cumsum(np.random.normal(0, 0.001, n_samples))
    lon_offsets = np.cumsum(np.random.normal(0, 0.001, n_samples))
    
    latitudes = base_lat + lat_offsets
    longitudes = base_lon + lon_offsets
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
    
    for idx in anomaly_indices:
        if idx < n_samples - 10:
            # Sudden location jumps
            if np.random.random() > 0.5:
                latitudes[idx:idx+3] += np.random.normal(0, 0.01, 3)
                longitudes[idx:idx+3] += np.random.normal(0, 0.01, 3)
            # Inactivity periods (same coordinates for multiple timestamps)
            else:
                latitudes[idx:idx+5] = latitudes[idx]
                longitudes[idx:idx+5] = longitudes[idx]
    
    data = {
        'timestamp': timestamps,
        'latitude': latitudes,
        'longitude': longitudes
    }
    
    return pd.DataFrame(data)


def compute_features(df):
    """
    Compute features: distance moved, speed, inactivity duration
    
    Args:
        df: DataFrame with columns ['timestamp', 'latitude', 'longitude']
    
    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate time differences in hours
    df['time_diff_hours'] = df['timestamp'].diff().dt.total_seconds() / 3600
    df['time_diff_hours'].fillna(0, inplace=True)
    
    # Calculate distance between consecutive points (Haversine formula)
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    df['distance_km'] = 0.0
    for i in range(1, len(df)):
        df.loc[i, 'distance_km'] = haversine_distance(
            df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude'],
            df.loc[i, 'latitude'], df.loc[i, 'longitude']
        )
    
    # Calculate speed (km/h)
    df['speed_kmh'] = df['distance_km'] / df['time_diff_hours']
    df['speed_kmh'].replace([np.inf, -np.inf], 0, inplace=True)
    
    # Calculate inactivity duration (consecutive zero-distance points)
    df['inactivity_duration'] = 0
    inactivity_count = 0
    
    for i in range(len(df)):
        if df.loc[i, 'distance_km'] < 0.001:  # Less than 1 meter
            inactivity_count += 1
            df.loc[i, 'inactivity_duration'] = inactivity_count * df.loc[i, 'time_diff_hours']
        else:
            inactivity_count = 0
            df.loc[i, 'inactivity_duration'] = 0
    
    # Calculate rolling statistics for context
    df['avg_speed_5h'] = df['speed_kmh'].rolling(window=5, min_periods=1).mean()
    df['max_distance_5h'] = df['distance_km'].rolling(window=5, min_periods=1).max()
    
    return df


def train_iforest(X):
    """
    Train Isolation Forest on the features
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
    
    Returns:
        Trained IsolationForest model
    """
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    # contamination=0.1 means we expect ~10% anomalies
    iforest = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    
    iforest.fit(X_scaled)
    
    return iforest, scaler


def detect_anomalies(model, scaler, X):
    """
    Detect anomalies using trained model
    
    Args:
        model: Trained IsolationForest model
        scaler: Fitted StandardScaler
        X: Feature matrix
    
    Returns:
        Array of anomaly predictions (-1 for anomaly, 1 for normal)
    """
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions


def raise_alerts(df, anomaly_column='anomaly'):
    """
    Raise alerts for detected anomalies
    
    Args:
        df: DataFrame with anomaly predictions
        anomaly_column: Name of the anomaly column
    """
    anomalies = df[df[anomaly_column] == -1]
    
    if len(anomalies) > 0:
        print(f"\n🚨 ALERT: {len(anomalies)} anomalies detected!")
        print("=" * 50)
        
        for idx, row in anomalies.iterrows():
            timestamp = row['timestamp']
            lat = row['latitude']
            lon = row['longitude']
            speed = row.get('speed_kmh', 0)
            distance = row.get('distance_km', 0)
            inactivity = row.get('inactivity_duration', 0)
            
            print(f"📍 Anomaly at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Location: ({lat:.6f}, {lon:.6f})")
            print(f"   Speed: {speed:.2f} km/h")
            print(f"   Distance: {distance:.3f} km")
            print(f"   Inactivity: {inactivity:.2f} hours")
            
            # Determine anomaly type
            if speed > 50:  # Unrealistic speed
                print("   ⚠️  Type: Abnormal high speed")
            elif distance > 1:  # Large location jump
                print("   ⚠️  Type: Sudden location jump")
            elif inactivity > 2:  # Long inactivity
                print("   ⚠️  Type: Extended inactivity")
            else:
                print("   ⚠️  Type: General anomaly")
            print()
    else:
        print("\n✅ No anomalies detected. Tourist appears safe.")


def visualize_anomalies(df, anomaly_column='anomaly'):
    """
    Create matplotlib visualization (blue=normal, red=anomaly)
    
    Args:
        df: DataFrame with anomaly predictions
        anomaly_column: Name of the anomaly column
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tourist Anomaly Detection Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data
    normal_data = df[df[anomaly_column] == 1]
    anomaly_data = df[df[anomaly_column] == -1]
    
    # 1. GPS Trajectory Plot
    ax1 = axes[0, 0]
    ax1.plot(normal_data['longitude'], normal_data['latitude'], 
             'b-', alpha=0.7, linewidth=1, label='Normal Path')
    ax1.scatter(normal_data['longitude'], normal_data['latitude'], 
               c='blue', s=10, alpha=0.6, label='Normal Points')
    
    if len(anomaly_data) > 0:
        ax1.scatter(anomaly_data['longitude'], anomaly_data['latitude'], 
                   c='red', s=50, marker='x', label='Anomalies')
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('GPS Trajectory (Blue=Normal, Red=Anomaly)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Speed over Time
    ax2 = axes[0, 1]
    ax2.plot(df['timestamp'], df['speed_kmh'], 'b-', alpha=0.7, linewidth=1)
    if len(anomaly_data) > 0:
        ax2.scatter(anomaly_data['timestamp'], anomaly_data['speed_kmh'], 
                   c='red', s=50, marker='x', label='Anomalies')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Speed (km/h)')
    ax2.set_title('Speed Over Time')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Distance Moved
    ax3 = axes[1, 0]
    ax3.plot(df['timestamp'], df['distance_km'], 'b-', alpha=0.7, linewidth=1)
    if len(anomaly_data) > 0:
        ax3.scatter(anomaly_data['timestamp'], anomaly_data['distance_km'], 
                   c='red', s=50, marker='x', label='Anomalies')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Distance (km)')
    ax3.set_title('Distance Moved Over Time')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Inactivity Duration
    ax4 = axes[1, 1]
    ax4.plot(df['timestamp'], df['inactivity_duration'], 'b-', alpha=0.7, linewidth=1)
    if len(anomaly_data) > 0:
        ax4.scatter(anomaly_data['timestamp'], anomaly_data['inactivity_duration'], 
                   c='red', s=50, marker='x', label='Anomalies')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Inactivity Duration (hours)')
    ax4.set_title('Inactivity Duration Over Time')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the complete anomaly detection pipeline
    """
    print("🏛️  Smart Tourist Safety App - Anomaly Detection Module")
    print("=" * 60)
    
    # Step 1: Generate mock data
    print("\n1. Generating mock tourist GPS data...")
    df = generate_mock_data(n_samples=1000)
    print(f"   Generated {len(df)} GPS data points")
    
    # Step 2: Compute features
    print("\n2. Computing features...")
    df = compute_features(df)
    print("   Features computed: distance, speed, inactivity duration")
    
    # Step 3: Prepare feature matrix for training
    feature_columns = ['distance_km', 'speed_kmh', 'inactivity_duration', 
                      'avg_speed_5h', 'max_distance_5h']
    X = df[feature_columns].values
    
    # Step 4: Train Isolation Forest
    print("\n3. Training Isolation Forest model...")
    model, scaler = train_iforest(X)
    print("   Model trained successfully")
    
    # Step 5: Detect anomalies
    print("\n4. Detecting anomalies...")
    predictions = detect_anomalies(model, scaler, X)
    df['anomaly'] = predictions
    
    # Count anomalies
    n_anomalies = sum(predictions == -1)
    n_normal = sum(predictions == 1)
    print(f"   Detected {n_anomalies} anomalies out of {len(df)} total points")
    print(f"   Anomaly rate: {n_anomalies/len(df)*100:.2f}%")
    
    # Step 6: Raise alerts
    print("\n5. Raising alerts...")
    raise_alerts(df)
    
    # Step 7: Create visualization
    print("\n6. Creating visualization...")
    visualize_anomalies(df)
    
    # Display summary statistics
    print("\n📊 Summary Statistics:")
    print("=" * 30)
    print(f"Total data points: {len(df)}")
    print(f"Normal points: {n_normal}")
    print(f"Anomalous points: {n_anomalies}")
    print(f"Average speed: {df['speed_kmh'].mean():.2f} km/h")
    print(f"Max speed: {df['speed_kmh'].max():.2f} km/h")
    print(f"Total distance traveled: {df['distance_km'].sum():.2f} km")
    print(f"Max inactivity duration: {df['inactivity_duration'].max():.2f} hours")
    
    return df


if __name__ == "__main__":
    # Run the complete pipeline
    result_df = main()
