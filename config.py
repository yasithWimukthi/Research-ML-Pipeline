# ─────────────────────────────────────────────
#  ML Pipeline Configuration
# ─────────────────────────────────────────────

# Elasticsearch
ES_HOST  = "http://localhost:9200"
ES_INDEX = "microservices-logs-*"

# Time window size in seconds
# Each window = one row in your dataset
WINDOW_SIZE_SECONDS = 30

# Prediction horizon in seconds (5 minutes)
# Windows within this period BEFORE a failure = label 1
PREDICTION_HORIZON_SECONDS = 300   # 5 minutes

# Minimum logs required in a window to be included
# Windows with too few logs are not reliable features
MIN_LOGS_PER_WINDOW = 3

# Services to extract features for
SERVICES = [
    "order-service",
    "inventory-service",
    "payment-service",
    "api-gateway",
]

# Output paths
DATASET_PATH        = "output/dataset.csv"
FEATURE_INFO_PATH   = "output/feature_info.json"
SCALER_PATH         = "output/scaler.pkl"
XGBOOST_MODEL_PATH  = "output/xgboost_model.pkl"
LSTM_MODEL_PATH     = "output/lstm_model.keras"
RESULTS_PATH        = "output/evaluation_results.json"
PLOTS_PATH          = "output/plots/"
