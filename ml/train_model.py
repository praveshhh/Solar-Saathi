"""
Solar Saathi — Hybrid LSTM + XGBoost Model Training Pipeline
=============================================================
Phase II of SolarSathi: Operational RUL Forecasting

Architecture:
  1. LSTM Network: Processes temporal sequences (30-step windows) of climate/power data
     to extract latent degradation signatures (64-dim embedding)
  2. XGBoost Regressor: Combines LSTM embeddings with static features 
     (location, module type, mounting, etc.) to predict Remaining Useful Life (RUL)

Evaluation: 80/20 chronological split, R² and RMSE metrics
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Paths
DATA_PATH = "ml/data/solar_saathi_dataset.csv"
MODEL_DIR = "ml/models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ===================================================================
# 1. DATA LOADING & PREPROCESSING
# ===================================================================

def load_and_preprocess():
    """Load the NASA-derived dataset and prepare features."""
    print("\n📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Locations: {df['location'].nunique()}")
    print(f"  Time range: {df['year'].min()}-{df['year'].max()}")
    
    # --- Temporal features (for LSTM sequences) ---
    temporal_cols = [
        "avg_temp_C", "max_temp_C", "min_temp_C",
        "humidity_pct", "wind_speed_ms",
        "ghi_kwh_m2_day", "uv_index", "precipitation_mm_day",
        "thermal_stress", "thermal_cycle_stress",
        "humidity_stress", "damp_heat_index",
        "uv_stress", "dust_soiling_index", "wind_stress",
        "module_temp_C", "dc_power_kw", "ac_power_kw",
        "daily_yield_kwh", "performance_ratio"
    ]
    
    # --- Static features (for XGBoost) ---
    static_cols = [
        "latitude", "longitude",
        "tilt_angle", "panel_wattage_W",
        "panel_age_years", "module_efficiency"
    ]
    
    # Categorical columns to encode
    cat_cols = ["module_type", "mounting_type", "climate_zone"]
    
    # Target
    target_col = "remaining_useful_life_years"
    secondary_target = "degradation_rate_pct_yr"
    
    # Drop rows with NaN in key columns
    essential_cols = temporal_cols + static_cols + [target_col, secondary_target]
    available_cols = [c for c in essential_cols if c in df.columns]
    df = df.dropna(subset=available_cols)
    
    print(f"  After cleaning: {df.shape}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            static_cols.append(col + "_encoded")
    
    # Save label encoders
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    
    return df, temporal_cols, static_cols, target_col, secondary_target


# ===================================================================
# 2. SEQUENCE CREATION FOR LSTM
# ===================================================================

def create_sequences(df, temporal_cols, static_cols, target_col, seq_length=12):
    """
    Create time-series sequences for LSTM.
    Each sequence is `seq_length` consecutive monthly observations
    for the same location + panel configuration.
    """
    print(f"\n🔄 Creating sequences (window={seq_length} months)...")
    
    # Group by location + panel config
    group_cols = ["location", "module_type", "mounting_type", "installation_year"]
    available_groups = [c for c in group_cols if c in df.columns]
    
    temporal_sequences = []
    static_features = []
    targets = []
    
    for _, group_df in df.groupby(available_groups):
        group_df = group_df.sort_values(["year", "month"])
        
        temp_data = group_df[temporal_cols].values
        stat_data = group_df[static_cols].values
        target_data = group_df[target_col].values
        
        for i in range(len(group_df) - seq_length):
            temporal_sequences.append(temp_data[i:i + seq_length])
            static_features.append(stat_data[i + seq_length - 1])  # Latest static features
            targets.append(target_data[i + seq_length - 1])  # Predict RUL at end of window
    
    X_temporal = np.array(temporal_sequences)
    X_static = np.array(static_features)
    y = np.array(targets)
    
    print(f"  Temporal sequences: {X_temporal.shape}")
    print(f"  Static features: {X_static.shape}")
    print(f"  Targets: {y.shape}")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}] years")
    
    return X_temporal, X_static, y


# ===================================================================
# 3. LSTM MODEL — Temporal Degradation Encoder
# ===================================================================

def build_lstm_encoder(seq_length, n_features, embedding_dim=64):
    """
    Build LSTM encoder that extracts temporal degradation signatures.
    Architecture: 2-layer Bidirectional LSTM → Dense embedding.
    """
    model = Sequential([
        Input(shape=(seq_length, n_features)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        BatchNormalization(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(embedding_dim, activation="relu", name="embedding"),
    ])
    return model


def build_lstm_regressor(seq_length, n_features, embedding_dim=64):
    """
    Full LSTM model that also predicts RUL (used for pre-training).
    The embedding layer output is later used by XGBoost.
    """
    inp = Input(shape=(seq_length, n_features))
    
    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    embedding = Dense(embedding_dim, activation="relu", name="embedding")(x)
    output = Dense(1, name="rul_output")(embedding)
    
    model = Model(inputs=inp, outputs=output)
    return model


# ===================================================================
# 4. TRAINING PIPELINE
# ===================================================================

def train_hybrid_model():
    """Main training pipeline for the hybrid LSTM+XGBoost model."""
    
    # Load data
    df, temporal_cols, static_cols, target_col, secondary_target = load_and_preprocess()
    
    # Create sequences
    SEQ_LENGTH = 12  # 12 months lookback
    X_temporal, X_static, y = create_sequences(
        df, temporal_cols, static_cols, target_col, SEQ_LENGTH
    )
    
    # Scale features
    print("\n📊 Scaling features...")
    n_samples, seq_len, n_temporal_features = X_temporal.shape
    
    # Scale temporal features
    temporal_scaler = StandardScaler()
    X_temporal_reshaped = X_temporal.reshape(-1, n_temporal_features)
    X_temporal_scaled = temporal_scaler.fit_transform(X_temporal_reshaped)
    X_temporal_scaled = X_temporal_scaled.reshape(n_samples, seq_len, n_temporal_features)
    
    # Scale static features
    static_scaler = StandardScaler()
    X_static_scaled = static_scaler.fit_transform(X_static)
    
    # Scale target
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Save scalers
    joblib.dump(temporal_scaler, os.path.join(MODEL_DIR, "temporal_scaler.pkl"))
    joblib.dump(static_scaler, os.path.join(MODEL_DIR, "static_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(MODEL_DIR, "target_scaler.pkl"))
    
    # Train/test split (chronological-style: index-based)
    split_idx = int(0.8 * len(y))
    X_temp_train, X_temp_test = X_temporal_scaled[:split_idx], X_temporal_scaled[split_idx:]
    X_stat_train, X_stat_test = X_static_scaled[:split_idx], X_static_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]
    
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    
    # ---------------------------------------------------------------
    # STEP 1: Train LSTM Encoder (pre-training on RUL prediction)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 1: Training LSTM Temporal Encoder")
    print("=" * 60)
    
    lstm_model = build_lstm_regressor(SEQ_LENGTH, n_temporal_features)
    lstm_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    lstm_model.summary()
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history = lstm_model.fit(
        X_temp_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # LSTM standalone evaluation
    lstm_pred_scaled = lstm_model.predict(X_temp_test).ravel()
    lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).ravel()
    lstm_r2 = r2_score(y_test_raw, lstm_pred)
    lstm_rmse = np.sqrt(mean_squared_error(y_test_raw, lstm_pred))
    print(f"\n  LSTM Standalone → R²: {lstm_r2:.4f} | RMSE: {lstm_rmse:.4f} years")
    
    # Save full LSTM model
    lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
    
    # ---------------------------------------------------------------
    # STEP 2: Extract LSTM Embeddings
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 2: Extracting LSTM Embeddings")
    print("=" * 60)
    
    # Create embedding extractor (up to the 'embedding' layer)
    embedding_model = Model(
        inputs=lstm_model.input,
        outputs=lstm_model.get_layer("embedding").output
    )
    
    # Extract embeddings for all data
    train_embeddings = embedding_model.predict(X_temp_train)
    test_embeddings = embedding_model.predict(X_temp_test)
    
    print(f"  Train embeddings: {train_embeddings.shape}")
    print(f"  Test embeddings: {test_embeddings.shape}")
    
    # Save embedding model
    embedding_model.save(os.path.join(MODEL_DIR, "lstm_embedding_model.h5"))
    
    # ---------------------------------------------------------------
    # STEP 3: Train XGBoost on Embeddings + Static Features
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 3: Training XGBoost (Embeddings + Static Features)")
    print("=" * 60)
    
    # Combine LSTM embeddings with static features
    X_xgb_train = np.hstack([train_embeddings, X_stat_train])
    X_xgb_test = np.hstack([test_embeddings, X_stat_test])
    
    print(f"  XGBoost input features: {X_xgb_train.shape[1]}")
    print(f"    - LSTM embeddings: {train_embeddings.shape[1]}")
    print(f"    - Static features: {X_stat_train.shape[1]}")
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    
    xgb_model.fit(
        X_xgb_train, y_train,
        eval_set=[(X_xgb_test, y_test)],
        verbose=50
    )
    
    # XGBoost predictions (on scaled target)
    xgb_pred_scaled = xgb_model.predict(X_xgb_test)
    xgb_pred = target_scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).ravel()
    
    # ---------------------------------------------------------------
    # STEP 4: Evaluation
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULTS: Model Performance Comparison")
    print("=" * 60)
    
    hybrid_r2 = r2_score(y_test_raw, xgb_pred)
    hybrid_rmse = np.sqrt(mean_squared_error(y_test_raw, xgb_pred))
    
    print(f"\n  {'Model':<30} {'R²':>8} {'RMSE (years)':>14}")
    print(f"  {'-'*54}")
    print(f"  {'LSTM Standalone':<30} {lstm_r2:>8.4f} {lstm_rmse:>14.4f}")
    print(f"  {'Hybrid LSTM+XGBoost':<30} {hybrid_r2:>8.4f} {hybrid_rmse:>14.4f}")
    print(f"\n  Improvement over LSTM alone:")
    print(f"    R² improvement:   {(hybrid_r2 - lstm_r2)*100:+.2f}%")
    print(f"    RMSE improvement: {(lstm_rmse - hybrid_rmse):+.4f} years")
    
    # Save XGBoost model
    xgb_model.save_model(os.path.join(MODEL_DIR, "xgboost_model.json"))
    
    # ---------------------------------------------------------------
    # STEP 5: Train SVR for degradation rate (Phase I model)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 5: Training SVR for Pre-Installation Degradation Estimation")
    print("=" * 60)
    
    from sklearn.svm import SVR
    
    # For SVR, we only use static/geo-climatic features
    svr_features = [
        "latitude", "longitude", "tilt_angle",
        "panel_wattage_W", "module_efficiency"
    ]
    
    # Aggregate climate data per location for static prediction
    agg_cols = {
        "avg_temp_C": "mean", "humidity_pct": "mean",
        "ghi_kwh_m2_day": "mean", "uv_index": "mean",
        "wind_speed_ms": "mean", "precipitation_mm_day": "mean",
        "degradation_rate_pct_yr": "mean"
    }
    
    # Add encoded categorical columns
    for col in ["module_type_encoded", "mounting_type_encoded", "climate_zone_encoded"]:
        if col in df.columns:
            agg_cols[col] = "first"
            svr_features.append(col)
    
    df_agg = df.groupby(["location", "module_type", "mounting_type"]).agg(
        {**agg_cols, **{c: "first" for c in svr_features if c not in agg_cols}}
    ).reset_index()
    
    # Add aggregated climate features to SVR features
    climate_feat = ["avg_temp_C", "humidity_pct", "ghi_kwh_m2_day",
                    "uv_index", "wind_speed_ms", "precipitation_mm_day"]
    svr_features_full = svr_features + climate_feat
    svr_features_full = [c for c in svr_features_full if c in df_agg.columns]
    
    X_svr = df_agg[svr_features_full].values
    y_svr = df_agg["degradation_rate_pct_yr"].values
    
    svr_scaler = StandardScaler()
    X_svr_scaled = svr_scaler.fit_transform(X_svr)
    
    X_svr_train, X_svr_test, y_svr_train, y_svr_test = train_test_split(
        X_svr_scaled, y_svr, test_size=0.2, random_state=42
    )
    
    svr_model = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.01)
    svr_model.fit(X_svr_train, y_svr_train)
    
    svr_pred = svr_model.predict(X_svr_test)
    svr_r2 = r2_score(y_svr_test, svr_pred)
    svr_rmse = np.sqrt(mean_squared_error(y_svr_test, svr_pred))
    
    print(f"\n  SVR (Phase I) → R²: {svr_r2:.4f} | RMSE: {svr_rmse:.4f} %/year")
    
    joblib.dump(svr_model, os.path.join(MODEL_DIR, "svr_model.pkl"))
    joblib.dump(svr_scaler, os.path.join(MODEL_DIR, "svr_scaler.pkl"))
    
    # Save feature configuration
    config = {
        "seq_length": SEQ_LENGTH,
        "temporal_features": temporal_cols,
        "static_features": static_cols,
        "svr_features": svr_features_full,
        "n_temporal_features": n_temporal_features,
        "n_static_features": X_static.shape[1],
        "embedding_dim": 64,
        "metrics": {
            "lstm_r2": float(lstm_r2),
            "lstm_rmse": float(lstm_rmse),
            "hybrid_r2": float(hybrid_r2),
            "hybrid_rmse": float(hybrid_rmse),
            "svr_r2": float(svr_r2),
            "svr_rmse": float(svr_rmse)
        }
    }
    
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"  ALL MODELS SAVED TO: {MODEL_DIR}/")
    print(f"  - lstm_model.h5")
    print(f"  - lstm_embedding_model.h5")
    print(f"  - xgboost_model.json")
    print(f"  - svr_model.pkl")
    print(f"  - temporal_scaler.pkl, static_scaler.pkl, target_scaler.pkl")
    print(f"  - svr_scaler.pkl, label_encoders.pkl")
    print(f"  - model_config.json")
    print(f"{'=' * 60}")
    
    return config


if __name__ == "__main__":
    config = train_hybrid_model()
