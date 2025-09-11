```python
import pandas as pd
import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
import time
import random
import os

# Define constants for the anomaly detection system
# --- Configuration Parameters ---
WINDOW_SIZE = 100  # Number of recent data points to keep for analysis
Z_SCORE_THRESHOLD = 3.0  # Threshold for statistical anomaly detection (e.g., 3 standard deviations)
IF_RETRAIN_INTERVAL = 50  # Retrain Isolation Forest every N data points
IF_CONTAMINATION = 0.05  # Expected proportion of outliers in the dataset for Isolation Forest
SIMULATION_DELAY_SECONDS = 0.1  # Delay between generating new network traffic data points
MAX_SIMULATION_ITERATIONS = 500 # Maximum iterations for the simulation

# Define numerical features that will be used for anomaly detection
NUMERICAL_FEATURES = [
    'packet_size_bytes',
    'num_packets',
    'duration_seconds',
    'byte_rate_bps' # bytes per second
]

# Define categorical features that might be interesting but not directly used by IsolationForest
CATEGORICAL_FEATURES = [
    'src_ip',
    'dst_ip',
    'src_port',
    'dst_port',
    'protocol'
]

# --- NetworkAnomalyDetector Class ---
class NetworkAnomalyDetector:
    """
    A real-time anomaly detection system for network traffic data.

    This system uses a combination of statistical (Z-score on rolling windows)
    and machine learning (Isolation Forest) methods to identify anomalous
    patterns in incoming network traffic data. It maintains a sliding window
    of recent data points and continuously updates its models and statistics.
    """

    def __init__(self, window_size: int = WINDOW_SIZE,
                 z_score_threshold: float = Z_SCORE_THRESHOLD,
                 if_retrain_interval: int = IF_RETRAIN_INTERVAL,
                 if_contamination: float = IF_CONTAMINATION):
        """
        Initializes the NetworkAnomalyDetector with specified parameters.

        Args:
            window_size (int): The maximum number of data points to keep in the rolling window.
            z_score_threshold (float): The Z-score threshold for statistical anomaly detection.
            if_retrain_interval (int): The number of new data points after which to retrain
                                       the Isolation Forest model.
            if_contamination (float): The 'contamination' parameter for Isolation Forest,
                                      representing the proportion of outliers in the data.
        """
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.if_retrain_interval = if_retrain_interval
        self.if_contamination = if_contamination

        self.data_window = deque(maxlen=window_size)  # Stores raw data dictionaries
        self.feature_window_df = pd.DataFrame(columns=NUMERICAL_FEATURES) # Stores features as DataFrame

        self.isolation_forest = None
        self.retrain_counter = 0
        self.is_if_trained = False

        print(f"NetworkAnomalyDetector initialized:")
        print(f"  Window Size: {self.window_size}")
        print(f"  Z-score Threshold: {self.z_score_threshold}")
        print(f"  IF Retrain Interval: {self.if_retrain_interval}")
        print(f"  IF Contamination: {self.if_contamination}")
        print(f"  Numerical Features: {NUMERICAL_FEATURES}")

    def _preprocess_data(self, data_point: dict) -> pd.Series:
        """
        Extracts and converts relevant numerical features from a raw data point
        into a pandas Series suitable for analysis.

        Args:
            data_point (dict): A dictionary representing a single network traffic event.

        Returns:
            pd.Series: A Series containing the numerical features for the data point.
        """
        processed_data = {}
        for feature in NUMERICAL_FEATURES:
            processed_data[feature] = data_point.get(feature, 0) # Use .get() for safety

        return pd.Series(processed_data)

    def add_data(self, data_point: dict):
        """
        Adds a new network traffic data point to the system's rolling window.

        Args:
            data_point (dict): A dictionary representing a single network traffic event.
        """
        self.data_window.append(data_point)
        processed_series = self._preprocess_data(data_point)

        # Append to DataFrame, handling initial empty state
        if self.feature_window_df.empty:
            self.feature_window_df = pd.DataFrame([processed_series])
        else:
            self.feature_window_df = pd.concat([self.feature_window_df, pd.DataFrame([processed_series])], ignore_index=True)

        # Ensure DataFrame does not exceed window size
        if len(self.feature_window_df) > self.window_size:
            self.feature_window_df = self.feature_window_df.iloc[1:].reset_index(drop=True)

        self.retrain_counter += 1

    def _retrain_isolation_forest(self):
        """
        Retrains the Isolation Forest model using the data currently in the window.
        This is done periodically to adapt to changing traffic patterns.
        """
        if len(self.feature_window_df) < max(2, self.window_size // 10): # Need minimum data points
            # print("  [INFO] Not enough data to train Isolation Forest yet.")
            return

        print(f"  [INFO] Retraining Isolation Forest with {len(self.feature_window_df)} data points...")
        try:
            # Initialize or re-initialize Isolation Forest
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=self.if_contamination,
                random_state=42,
                n_jobs=-1 # Use all available cores
            )
            # Fit the model to the current window data
            self.isolation_forest.fit(self.feature_window_df[NUMERICAL_FEATURES])
            self.is_if_trained = True
            print("  [INFO] Isolation Forest retraining complete.")
        except Exception as e:
            print(f"  [ERROR] Failed to retrain Isolation Forest: {e}")
            self.is_if_trained = False # Mark as not trained if error occurs

        self.retrain_counter = 0 # Reset counter after retraining

    def detect_anomalies(self, current_data_point: dict) -> tuple[bool, str]:
        """
        Detects anomalies in the current network traffic data point using both
        statistical Z-score and Isolation Forest.

        Args:
            current_data_point (dict): The most recent network traffic event to check.

        Returns:
            tuple[bool, str]: A tuple where the first element is True if an anomaly
                              is detected, False otherwise. The second element is
                              a string describing the reason for detection.
        """
        if len(self.feature_window_df) < 2: # Need at least two data points for statistics
            return False, "Not enough data for detection."

        # --- Statistical Anomaly Detection (Z-score) ---
        current_processed_data = self._preprocess_data(current_data_point)
        reasons = []

        for feature in NUMERICAL_FEATURES:
            if feature in self.feature_window_df.columns:
                window_data = self.feature_window_df[feature].values
                # Avoid division by zero if std dev is 0
                if np.std(window_data) == 0:
                    if current_processed_data[feature] != window_data[0]:
                        # If std is 0, all values are the same. Any different value is an anomaly.
                        reasons.append(f"Statistical: {feature} (value={current_processed_data[feature]:.2f}) is unique when window is constant.")
                        return True, "; ".join(reasons) # Immediate anomaly
                    continue # No anomaly if value is constant and same

                z_score = np.abs((current_processed_data[feature] - np.mean(window_data)) / np.std(window_data))
                if z_score > self.z_score_threshold:
                    reasons.append(f"Statistical: {feature} (value={current_processed_data[feature]:.2f}, Z-score={z_score:.2f}) is beyond {self.z_score_threshold} std devs.")

        if reasons:
            return True, "; ".join(reasons)

        # --- Isolation Forest Anomaly Detection ---
        if self.retrain_counter >= self.if_retrain_interval:
            self._retrain_isolation_forest()

        if self.is_if_trained and len(self.feature_window_df) >= self.window_size:
            # Predict the anomaly score for the current data point
            # IsolationForest expect a 2D array, even for a single sample
            current_features_df = pd.DataFrame([current_processed_data[NUMERICAL_FEATURES]])
            # The predict method returns -1 for outliers and 1 for inliers
            if_prediction = self.isolation_forest.predict(current_features_df)
            if if_prediction[0] == -1:
                return True, "Isolation Forest: Detected as an outlier."

        return False, "No anomaly detected."

# --- Data Simulation Function ---
def generate_network_traffic_data(iteration: int, introduce_anomaly: bool = False) -> dict:
    """
    Simulates a single network traffic data point.
    Can optionally introduce an anomaly for testing purposes.

    Args:
        iteration (int): The current simulation iteration, used for time and context.
        introduce_anomaly (bool): If True, a specific anomalous pattern will be generated.

    Returns:
        dict: A dictionary representing a simulated network traffic event.
    """
    timestamp = time.time()
    src_ip = f"192.168.1.{random.randint(1, 254)}"
    dst_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"
    src_port = random.choice([80, 443, 22, 21, 23, 53, 3389] + [random.randint(1024, 65535)])
    dst_port = random.choice([80, 443, 22, 21, 23, 53, 3389] + [random.randint(1024, 65535)])
    protocol = random.choice(["TCP", "UDP", "ICMP"])
    
    # Base values for packet size and number of packets
    base_packet_size = random.gauss(1500, 300) # bytes, centered around typical MTU
    base_num_packets = random.gauss(100, 30) # number of packets in a flow/interval
    base_duration = random.uniform(0.1, 5.0) # seconds

    # Ensure positive values
    packet_size_bytes = max(10, int(base_packet_size))
    num_packets = max(1, int(base_num_packets))
    duration_seconds = max(0.01, base_duration)
    
    byte_rate_bps = (packet_size_bytes * num_packets) / duration_seconds if duration_seconds > 0 else 0


    if introduce_anomaly:
        anomaly_type = random.choice(["high_packets_unusual_port", "large_packet_spike", "ddos_like_rate"])
        if anomaly_type == "high_packets_unusual_port":
            num_packets = random.randint(500, 2000)  # Very high number of packets
            dst_port = random.randint(49152, 65535) # High/ephemeral port, often unusual for server traffic
            src_ip = "192.168.1.100" # A specific source
            protocol = "UDP"
            print(f"  [SIMULATION] Injecting ANOMALY: High packets to unusual port from {src_ip}")
        elif anomaly_type == "large_packet_spike":
            packet_size_bytes = random.randint(4000, 10000) # Very large packet size
            num_packets = random.randint(1, 10) # Maybe just a few very large packets
            duration_seconds = random.uniform(0.01, 0.1) # Very short duration
            byte_rate_bps = (packet_size_bytes * num_packets) / duration_seconds if duration_seconds > 0 else 0
            print(f"  [SIMULATION] Injecting ANOMALY: Large packet size spike.")
        elif anomaly_type == "ddos_like_rate":
            num_packets = random.randint(1000, 5000) # Extremely high number of packets
            duration_seconds = random.uniform(0.01, 0.5) # Very short duration
            src_ip = f"192.168.{random.randint(1,2)}.{random.randint(1,254)}" # Varying source IPs but high rate
            dst_ip = "10.0.0.5" # Target a specific internal server
            dst_port = random.choice([80,443])
            byte_rate_bps = (packet_size_bytes * num_packets) / duration_seconds if duration_seconds > 0 else 0
            print(f"  [SIMULATION] Injecting ANOMALY: DDoS-like high packet rate.")

    return {
        "timestamp": timestamp,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol,
        "packet_size_bytes": packet_size_bytes,
        "num_packets": num_packets,
        "duration_seconds": duration_seconds,
        "byte_rate_bps": byte_rate_bps
    }

# --- Main Simulation Loop ---
def main():
    """
    Main function to run the real-time anomaly detection system simulation.
    It continuously generates simulated network traffic, feeds it to the
    detector, and prints detection results.
    """
    print("\n--- Starting Real-time Anomaly Detection System Simulation ---")
    detector = NetworkAnomalyDetector()

    # Clear screen for cleaner output
    # os.system('cls' if os.name == 'nt' else 'clear') 

    for i in range(1, MAX_SIMULATION_ITERATIONS + 1):
        # Introduce an anomaly every N iterations (e.g., every 20-30 iterations)
        # after the system has enough data to learn normal patterns.
        introduce_anomaly = (i > WINDOW_SIZE * 0.5) and (i % random.randint(20, 30) == 0)
        
        # Introduce a burst of anomalies to simulate an attack
        if i == MAX_SIMULATION_ITERATIONS // 2 and not introduce_anomaly:
             print("\n!!! SIMULATING A BURST ATTACK !!!\n")
             for _ in range(5):
                 simulated_data = generate_network_traffic_data(i, introduce_anomaly=True)
                 detector.add_data(simulated_data)
                 if len(detector.data_window) >= WINDOW_SIZE // 2: # Start detecting early if window half full
                     is_anomaly, reason = detector.detect_anomalies(simulated_data)
                     status = f"ANOMALY DETECTED ({reason})" if is_anomaly else "Normal Traffic"
                     print(f"[{i:04d}] {simulated_data['src_ip']}:{simulated_data['src_port']} -> "
                           f"{simulated_data['dst_ip']}:{simulated_data['dst_port']} | "
                           f"Pkts: {simulated_data['num_packets']} | Size: {simulated_data['packet_size_bytes']} | "
                           f"STATUS: {status}")
                 time.sleep(SIMULATION_DELAY_SECONDS * 0.5) # Faster for burst
             print("\n!!! BURST ATTACK SIMULATION ENDED !!!\n")

        simulated_data = generate_network_traffic_data(i, introduce_anomaly)
        detector.add_data(simulated_data)

        if len(detector.data_window) < WINDOW_SIZE:
            print(f"[{i:04d}] Collecting data... {len(detector.data_window)}/{WINDOW_SIZE} points in window.")
        else:
            is_anomaly, reason = detector.detect_anomalies(simulated_data)
            
            status_prefix = "ANOMALY DETECTED" if is_anomaly else "Normal Traffic"
            status_color_start = "\033[91m" if is_anomaly else "\033[92m" # Red for anomaly, Green for normal
            status_color_end = "\033[0m" # Reset color

            print(f"[{i:04d}] {simulated_data['src_ip']}:{simulated_data['src_port']} -> "
                  f"{simulated_data['dst_ip']}:{simulated_data['dst_port']} | "
                  f"Pkts: {simulated_data['num_packets']} | Size: {simulated_data['packet_size_bytes']} | "
                  f"Rate: {simulated_data['byte_rate_bps']:.0f} Bps | "
                  f"{status_color_start}STATUS: {status_prefix}{status_color_end}{(' (' + reason + ')' if is_anomaly else '')}")

        time.sleep(SIMULATION_DELAY_SECONDS)

    print("\n--- Simulation Finished ---")

if __name__ == "__main__":
    main()
```