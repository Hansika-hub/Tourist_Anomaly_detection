# Smart Tourist Safety – Anomaly Detection Module

## Overview

The Anomaly Detection Module is a component of the **Smart Tourist Safety Application** designed to monitor tourist movement using GPS coordinates and timestamps.

It uses **Machine Learning (Isolation Forest)** to identify unusual movement patterns, such as:
- Sudden and unexpected changes in GPS location.
- Abnormally high travel speeds.
- Extended periods of inactivity.
- Other unusual movement behaviours.

The system analyses GPS trajectories, extracts movement-related features, detects anomalies, and generates alerts to help identify potentially unsafe situations.

## Features

- **GPS Movement Tracking:** Processes latitude, longitude, and timestamp data.
- **Feature Extraction:** Calculates distance travelled, movement speed, and inactivity duration.
- **Anomaly Detection:** Uses the Isolation Forest algorithm to identify unusual movement patterns.
- **Automated Alerts:** Reports detected anomalies along with their timestamps, locations, and movement statistics.
- **Data Visualization:** Generates plots to analyse GPS trajectories, speed, distance, and inactivity patterns.
- **Summary Statistics:** Provides an overview of normal and anomalous data points.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Isolation Forest and StandardScaler |
| Matplotlib | Data visualization |

## Methodology

The anomaly detection pipeline consists of the following stages:

### 1. GPS Data Generation

The current implementation generates mock tourist GPS data for experimentation.

- Generates 1,000 GPS data points by default.
- Simulates timestamps at five-minute intervals.
- Uses geographical coordinates around New York City.
- Introduces artificial anomalies, including sudden location jumps and inactivity periods.

### 2. Feature Extraction

Movement-related features are extracted from consecutive GPS observations.

| Feature | Description |
|---|---|
| `distance_km` | Distance between consecutive GPS coordinates using the Haversine formula. |
| `speed_kmh` | Calculated movement speed in kilometres per hour. |
| `inactivity_duration` | Duration of consecutive observations with negligible movement. |
| `avg_speed_5h` | Rolling average speed over the previous five observations. |
| `max_distance_5h` | Maximum distance recorded over the previous five observations. |

### 3. Feature Scaling

The extracted features are standardized using `StandardScaler` before being passed to the machine learning model.

### 4. Anomaly Detection Using Isolation Forest

The system uses the **Isolation Forest** unsupervised machine learning algorithm.

Model configuration:
- Number of estimators: 100
- Contamination: 0.1 (expected proportion of anomalies)
- Random state: 42

Isolation Forest identifies unusual observations by isolating data points that differ significantly from normal movement patterns.

The model produces two prediction labels:

| Prediction | Meaning |
|---|---|
| `1` | Normal observation |
| `-1` | Anomalous observation |

### 5. Alert Generation

For every detected anomaly, the system displays:
- Timestamp
- GPS coordinates
- Travel speed
- Distance travelled
- Inactivity duration
- Possible anomaly category

The system categorizes anomalies using threshold-based checks:

- **Abnormal High Speed:** Speed greater than 50 km/h.
- **Sudden Location Jump:** Distance greater than 1 km.
- **Extended Inactivity:** Inactivity duration greater than 2 hours.
- **General Anomaly:** Anomaly that does not match the above conditions.

### 6. Data Visualization

The module generates a dashboard containing four visualizations:

1. **GPS Trajectory:** Displays normal and anomalous GPS locations.
2. **Speed Over Time:** Shows changes in tourist movement speed.
3. **Distance Over Time:** Displays distances travelled between consecutive observations.
4. **Inactivity Duration:** Shows periods of extended inactivity.

Normal observations are represented in blue, while anomalies are highlighted in red.

## Project Structure

```text
anomaly-detection/
│
├── anomaly_detection.py
└── README.md
