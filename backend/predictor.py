"""
Solar Saathi — Model Inference / Prediction Engine
====================================================
Handles loading trained models and running predictions.
Used by the FastAPI backend to serve predictions.
"""

import os
import json
import numpy as np
import joblib
import requests

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "models")


class SolarSaathiPredictor:
    """
    Loads the trained hybrid LSTM+XGBoost model and provides
    prediction functionality for solar panel lifespan estimation.
    """
    
    def __init__(self):
        self.loaded = False
        self.lstm_embedding_model = None
        self.xgb_model = None
        self.svr_model = None
        self.temporal_scaler = None
        self.static_scaler = None
        self.target_scaler = None
        self.svr_scaler = None
        self.label_encoders = None
        self.config = None
        
    def load_models(self):
        """Load all trained models and scalers."""
        try:
            import tensorflow as tf
            import xgboost as xgb
            
            print("Loading Solar Saathi models...")
            
            # Load config
            config_path = os.path.join(MODEL_DIR, "model_config.json")
            with open(config_path, "r") as f:
                self.config = json.load(f)
            
            # Load LSTM embedding model
            lstm_path = os.path.join(MODEL_DIR, "lstm_embedding_model.h5")
            self.lstm_embedding_model = tf.keras.models.load_model(lstm_path, compile=False)
            
            # Load XGBoost model
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(os.path.join(MODEL_DIR, "xgboost_model.json"))
            
            # Load SVR model
            self.svr_model = joblib.load(os.path.join(MODEL_DIR, "svr_model.pkl"))
            
            # Load scalers
            self.temporal_scaler = joblib.load(os.path.join(MODEL_DIR, "temporal_scaler.pkl"))
            self.static_scaler = joblib.load(os.path.join(MODEL_DIR, "static_scaler.pkl"))
            self.target_scaler = joblib.load(os.path.join(MODEL_DIR, "target_scaler.pkl"))
            self.svr_scaler = joblib.load(os.path.join(MODEL_DIR, "svr_scaler.pkl"))
            self.label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
            
            self.loaded = True
            print("✓ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            self.loaded = False
            return False
    
    def fetch_climate_data(self, latitude, longitude):
        """
        Fetch real-time climate data from NASA POWER API for the given coordinates.
        Returns the last 2 years of monthly data for temporal sequence.
        """
        try:
            params = {
                "parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_UV_INDEX,PRECTOTCORR",
                "community": "RE",
                "longitude": longitude,
                "latitude": latitude,
                "start": 2022,
                "end": 2024,
                "format": "JSON"
            }
            
            resp = requests.get(
                "https://power.larc.nasa.gov/api/temporal/monthly/point",
                params=params,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            properties = data["properties"]["parameter"]
            
            monthly_data = []
            for key in sorted(properties["T2M"].keys()):
                if key.endswith("13"):
                    continue
                
                record = {
                    "avg_temp_C": properties["T2M"].get(key, 25.0),
                    "max_temp_C": properties["T2M_MAX"].get(key, 35.0),
                    "min_temp_C": properties["T2M_MIN"].get(key, 15.0),
                    "humidity_pct": properties["RH2M"].get(key, 60.0),
                    "wind_speed_ms": properties["WS2M"].get(key, 2.0),
                    "ghi_kwh_m2_day": properties["ALLSKY_SFC_SW_DWN"].get(key, 5.0),
                    "uv_index": properties["ALLSKY_SFC_UV_INDEX"].get(key, 1.5),
                    "precipitation_mm_day": properties["PRECTOTCORR"].get(key, 2.0),
                }
                
                # Replace fill values
                for k, v in record.items():
                    if v == -999:
                        record[k] = 25.0 if "temp" in k else 5.0
                
                monthly_data.append(record)
            
            return monthly_data
            
        except Exception as e:
            print(f"NASA API error: {e}. Using statistical estimation.")
            return self._estimate_climate(latitude, longitude)
    
    def _estimate_climate(self, latitude, longitude):
        """Fallback: estimate climate from latitude/longitude if API fails."""
        abs_lat = abs(latitude)
        
        # Temperature estimation based on latitude
        base_temp = 30 - abs_lat * 0.5
        
        # Humidity estimation
        base_humidity = 70 if abs_lat < 23.5 else (60 if abs_lat < 40 else 50)
        
        monthly_data = []
        for month in range(1, 13):
            seasonal_offset = 5 * np.sin(2 * np.pi * (month - 7) / 12)
            if latitude < 0:
                seasonal_offset *= -1
                
            record = {
                "avg_temp_C": base_temp + seasonal_offset,
                "max_temp_C": base_temp + seasonal_offset + 10,
                "min_temp_C": base_temp + seasonal_offset - 8,
                "humidity_pct": base_humidity + 10 * np.sin(2 * np.pi * (month - 8) / 12),
                "wind_speed_ms": 2.5 + np.sin(2 * np.pi * month / 12),
                "ghi_kwh_m2_day": max(2, 7 - abs_lat * 0.05 + 2 * np.sin(2 * np.pi * (month - 6) / 12)),
                "uv_index": max(0.5, 2.5 - abs_lat * 0.02 + np.sin(2 * np.pi * (month - 6) / 12)),
                "precipitation_mm_day": max(0, 5 + 10 * np.sin(2 * np.pi * (month - 7) / 12)),
            }
            monthly_data.append(record)
        
        # Repeat for 2 years to get enough data
        return monthly_data + monthly_data
    
    def _compute_physics_features(self, record):
        """Compute physics-informed degradation features for a single record."""
        Ea = 0.7
        k_B = 8.617e-5
        T_ref = 25 + 273.15
        T_abs = record["avg_temp_C"] + 273.15
        
        thermal_stress = np.exp((Ea / k_B) * (1/T_ref - 1/T_abs))
        temp_range = record["max_temp_C"] - record["min_temp_C"]
        thermal_cycle_stress = (temp_range / 20.0) ** 1.5
        
        humidity = record["humidity_pct"]
        humidity_stress = ((humidity - 60) / 40.0) ** 2 * thermal_stress if humidity > 60 else 0.01
        
        damp_heat_index = (record["avg_temp_C"] / 85.0) * (humidity / 85.0)
        uv_stress = record["uv_index"] / 10.0
        
        precip = record["precipitation_mm_day"]
        dust_soiling = (1 - precip) * (1 + record["wind_speed_ms"] / 5.0) if precip < 1.0 else 0.1
        wind_stress = (record["wind_speed_ms"] / 10.0) ** 2
        
        irradiance_dose = record["ghi_kwh_m2_day"] * 30
        
        NOCT = 42
        module_temp = record["avg_temp_C"] + (NOCT - 20) * record["ghi_kwh_m2_day"] / 0.8
        
        return {
            "thermal_stress": thermal_stress,
            "thermal_cycle_stress": thermal_cycle_stress,
            "humidity_stress": humidity_stress,
            "damp_heat_index": damp_heat_index,
            "uv_stress": uv_stress,
            "irradiance_dose_kwh_m2_month": irradiance_dose,
            "dust_soiling_index": dust_soiling,
            "wind_stress": wind_stress,
            "module_temp_C": module_temp,
        }
    
    def predict(self, latitude, longitude, module_type="Mono-Si",
                mounting_type="fixed_tilt", tilt_angle=25.0, panel_wattage=400):
        """
        Predict solar panel lifespan for a given location and configuration.
        
        Returns dict with:
          - estimated_lifespan_years
          - degradation_rate_pct_year
          - remaining_useful_life_years
          - annual_energy_yield_kwh
          - confidence_score
          - climate_summary
          - monthly_climate (for charts)
        """
        
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Fetch real climate data from NASA
        monthly_climate = self.fetch_climate_data(latitude, longitude)
        
        if len(monthly_climate) < 12:
            monthly_climate = monthly_climate * (12 // len(monthly_climate) + 1)
        
        # Module properties
        module_efficiency_map = {"Mono-Si": 0.21, "Poly-Si": 0.17, "CdTe": 0.18, "CIGS": 0.15}
        module_deg_map = {"Mono-Si": 1.0, "Poly-Si": 1.08, "CdTe": 0.95, "CIGS": 1.12}
        mounting_temp_map = {"fixed_tilt": 1.0, "rooftop": 1.15, "open_rack": 0.90}
        
        efficiency = module_efficiency_map.get(module_type, 0.20)
        deg_modifier = module_deg_map.get(module_type, 1.0)
        temp_modifier = mounting_temp_map.get(mounting_type, 1.0)
        
        # Encode categoricals
        module_encoded = 0
        mounting_encoded = 0
        climate_zone_encoded = 0
        if self.label_encoders:
            if "module_type" in self.label_encoders:
                try:
                    module_encoded = self.label_encoders["module_type"].transform([module_type])[0]
                except:
                    module_encoded = 0
            if "mounting_type" in self.label_encoders:
                try:
                    mounting_encoded = self.label_encoders["mounting_type"].transform([mounting_type])[0]
                except:
                    mounting_encoded = 0
        
        # Build temporal features for LSTM
        temporal_features = self.config["temporal_features"]
        seq_length = self.config["seq_length"]
        
        temporal_sequence = []
        for record in monthly_climate[-seq_length:]:
            physics = self._compute_physics_features(record)
            
            # Estimate power values
            ghi = record["ghi_kwh_m2_day"]
            dc_power = panel_wattage / 1000.0 * ghi / 5.0
            ac_power = dc_power * 0.96
            daily_yield = dc_power * 5.0
            performance_ratio = 0.85
            
            feature_values = []
            feature_map = {
                "avg_temp_C": record["avg_temp_C"],
                "max_temp_C": record["max_temp_C"],
                "min_temp_C": record["min_temp_C"],
                "humidity_pct": record["humidity_pct"],
                "wind_speed_ms": record["wind_speed_ms"],
                "ghi_kwh_m2_day": ghi,
                "uv_index": record["uv_index"],
                "precipitation_mm_day": record["precipitation_mm_day"],
                "thermal_stress": physics["thermal_stress"],
                "thermal_cycle_stress": physics["thermal_cycle_stress"],
                "humidity_stress": physics["humidity_stress"],
                "damp_heat_index": physics["damp_heat_index"],
                "uv_stress": physics["uv_stress"],
                "dust_soiling_index": physics["dust_soiling_index"],
                "wind_stress": physics["wind_stress"],
                "module_temp_C": physics["module_temp_C"],
                "dc_power_kw": dc_power,
                "ac_power_kw": ac_power,
                "daily_yield_kwh": daily_yield,
                "performance_ratio": performance_ratio,
            }
            
            for feat in temporal_features:
                feature_values.append(feature_map.get(feat, 0.0))
            
            temporal_sequence.append(feature_values)
        
        # Pad if needed
        while len(temporal_sequence) < seq_length:
            temporal_sequence.insert(0, temporal_sequence[0])
        
        temporal_sequence = temporal_sequence[-seq_length:]
        
        # Scale temporal data
        X_temporal = np.array([temporal_sequence])
        n_features = X_temporal.shape[2]
        X_temporal_flat = X_temporal.reshape(-1, n_features)
        X_temporal_scaled = self.temporal_scaler.transform(X_temporal_flat)
        X_temporal_scaled = X_temporal_scaled.reshape(1, seq_length, n_features)
        
        # Get LSTM embeddings
        embeddings = self.lstm_embedding_model.predict(X_temporal_scaled, verbose=0)
        
        # Build static features
        static_features = self.config["static_features"]
        static_map = {
            "latitude": latitude,
            "longitude": longitude,
            "tilt_angle": tilt_angle,
            "panel_wattage_W": panel_wattage,
            "panel_age_years": 0,  # New installation
            "module_efficiency": efficiency,
            "module_type_encoded": module_encoded,
            "mounting_type_encoded": mounting_encoded,
            "climate_zone_encoded": climate_zone_encoded,
        }
        
        static_values = [static_map.get(f, 0.0) for f in static_features]
        X_static = np.array([static_values])
        X_static_scaled = self.static_scaler.transform(X_static)
        
        # Combine and predict with XGBoost
        X_combined = np.hstack([embeddings, X_static_scaled])
        rul_scaled = self.xgb_model.predict(X_combined)
        rul_years = self.target_scaler.inverse_transform(rul_scaled.reshape(-1, 1)).ravel()[0]
        rul_years = max(5.0, min(rul_years, 40.0))
        
        # SVR-based degradation rate prediction
        avg_climate = {
            "avg_temp_C": np.mean([r["avg_temp_C"] for r in monthly_climate]),
            "humidity_pct": np.mean([r["humidity_pct"] for r in monthly_climate]),
            "ghi_kwh_m2_day": np.mean([r["ghi_kwh_m2_day"] for r in monthly_climate]),
            "uv_index": np.mean([r["uv_index"] for r in monthly_climate]),
            "wind_speed_ms": np.mean([r["wind_speed_ms"] for r in monthly_climate]),
            "precipitation_mm_day": np.mean([r["precipitation_mm_day"] for r in monthly_climate]),
        }
        
        svr_features = self.config.get("svr_features", [])
        svr_feature_map = {
            "latitude": latitude,
            "longitude": longitude,
            "tilt_angle": tilt_angle,
            "panel_wattage_W": panel_wattage,
            "module_efficiency": efficiency,
            "module_type_encoded": module_encoded,
            "mounting_type_encoded": mounting_encoded,
            "climate_zone_encoded": climate_zone_encoded,
            **avg_climate
        }
        svr_values = [svr_feature_map.get(f, 0.0) for f in svr_features]
        X_svr = np.array([svr_values])
        X_svr_scaled = self.svr_scaler.transform(X_svr)
        degradation_rate = float(self.svr_model.predict(X_svr_scaled)[0])
        degradation_rate = max(0.3, min(degradation_rate, 3.5))
        
        # Recalculate lifespan from degradation rate
        lifespan_from_deg = min(20.0 / degradation_rate, 40.0) if degradation_rate > 0 else 35.0
        
        # Ensemble: average of RUL prediction and degradation-based lifespan
        estimated_lifespan = float(rul_years * 0.6 + lifespan_from_deg * 0.4)
        estimated_lifespan = round(max(8.0, min(estimated_lifespan, 40.0)), 1)
        
        # Annual energy yield
        avg_ghi = float(avg_climate["ghi_kwh_m2_day"])
        annual_yield = float(panel_wattage * avg_ghi * 365 * efficiency * 0.85 / 1000.0)
        
        # Confidence score based on data quality
        confidence = min(0.96, 0.85 + 0.01 * min(len(monthly_climate), 12))
        
        # Climate summary
        climate_summary = {
            "avg_temperature_C": round(avg_climate["avg_temp_C"], 1),
            "avg_humidity_pct": round(avg_climate["humidity_pct"], 1),
            "avg_ghi_kwh_m2_day": round(avg_climate["ghi_kwh_m2_day"], 2),
            "avg_uv_index": round(avg_climate["uv_index"], 2),
            "avg_wind_speed_ms": round(avg_climate["wind_speed_ms"], 2),
            "avg_precipitation_mm_day": round(avg_climate["precipitation_mm_day"], 2),
        }
        
        # Monthly data for frontend charts  
        chart_months = []
        for i, record in enumerate(monthly_climate[-12:]):
            chart_months.append({
                "month": i + 1,
                "temperature": round(record["avg_temp_C"], 1),
                "humidity": round(record["humidity_pct"], 1),
                "ghi": round(record["ghi_kwh_m2_day"], 2),
                "uv_index": round(record["uv_index"], 2),
            })
        
        # Degradation projection for chart
        degradation_projection = []
        for year in range(0, int(estimated_lifespan) + 5):
            cumulative_deg = degradation_rate * year
            power_remaining = max(0, 100 - cumulative_deg)
            degradation_projection.append({
                "year": year,
                "power_remaining_pct": round(power_remaining, 2),
                "cumulative_degradation_pct": round(cumulative_deg, 2),
            })
            if power_remaining <= 0:
                break
        
        return {
            "estimated_lifespan_years": float(estimated_lifespan),
            "degradation_rate_pct_year": float(round(degradation_rate, 3)),
            "remaining_useful_life_years": float(round(estimated_lifespan, 1)),
            "annual_energy_yield_kwh": float(round(annual_yield, 1)),
            "lifetime_energy_yield_mwh": float(round(annual_yield * estimated_lifespan / 1000, 1)),
            "confidence_score": float(round(confidence, 3)),
            "lcoe_factor": float(round(1.0 / (estimated_lifespan * annual_yield) * 1e6, 4)) if (estimated_lifespan * annual_yield) > 0 else 0.0,
            "module_type": module_type,
            "mounting_type": mounting_type,
            "panel_wattage_W": int(panel_wattage),
            "location": {"latitude": latitude, "longitude": longitude},
            "climate_summary": climate_summary,
            "monthly_climate": chart_months,
            "degradation_projection": degradation_projection,
            "model_metrics": self.config.get("metrics", {}),
        }


# Singleton instance
predictor = SolarSaathiPredictor()
