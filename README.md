# 🏎️ Formula 1 Race Predictions

A production-grade **serverless ML pipeline** that automatically predicts Formula 1 race finishing positions using real-time weather data, historical performance metrics, and XGBoost machine learning models deployed on AWS.

This is a public showcase of how to build a fully automated, event-driven data pipeline that ingests F1 race data, engineers features, trains ML models, and generates predictions—all running serverless on AWS Lambda with zero operational overhead.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Installation & Setup](#installation--setup)
- [AWS Configuration](#aws-configuration)
- [Usage](#usage)
- [Requirements](#requirements)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

### What This Project Does

This repository demonstrates a **complete end-to-end ML pipeline** for Formula 1 race outcome prediction:

1. **Automatically ingests** qualifying and race data via the [FastF1](https://docs.fastf1.dev/) API
2. **Fetches real-time weather** forecasts from [Open-Meteo](https://open-meteo.com/)
3. **Engineers features** using historical race data, driver form, circuit history, and weather conditions
4. **Trains an XGBoost model** with multi-stage preprocessing and exports to ONNX format for fast inference
5. **Generates predictions** automatically when qualifying is complete—predicting each driver's finishing position
6. **Runs entirely serverless** on AWS Lambda, triggered by EventBridge on a 2-hour schedule

The pipeline runs autonomously throughout F1 race weekends, continuously updating predictions as new data arrives. Predictions are stored in DynamoDB and S3 for easy access.

---

## Key Features

✅ **Fully Automated** - EventBridge-triggered pipeline that runs 24/7 during race season
✅ **Serverless Architecture** - No servers to manage; pay only for compute used
✅ **Real-Time Predictions** - Updates predictions when race forecasts change
✅ **Comprehensive Features** - Qualifying position, lap pace, weather, circuit history, season form
✅ **Production-Ready** - Handles missing data, multiple race formats (conventional & sprint), error recovery
✅ **Portable Models** - ONNX export enables predictions on any platform
✅ **Data Lineage Tracking** - DynamoDB tracks every pipeline stage for debugging
✅ **Scalable Design** - Handles all 25 races/year with minimal infrastructure

---

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     AWS Serverless Pipeline                      │
└─────────────────────────────────────────────────────────────────┘

    EventBridge Scheduler (2-hourly)
            │
            ▼
    ┌──────────────────────┐
    │  Orchestrator        │ (Lambda)
    │  - Checks for new    │
    │    race weekends     │
    │  - Coordinates all   │
    │    pipeline stages   │
    └──────────────────────┘
            │
    ┌───────┴─────────────────────────────────┐
    │                                           │
    ▼                                           ▼
┌──────────────────┐              ┌──────────────────────────┐
│  Data Ingestion  │              │   Weather Data Fetch     │
│  (Lambda)        │              │   (Lambda)               │
│                  │              │                          │
│ • FastF1 API     │              │ • Open-Meteo Forecast    │
│ • Session Data   │              │ • Historical Archive     │
│ • → S3 Bronze    │              │ • → S3 Bronze            │
└──────────────────┘              └──────────────────────────┘
    │                                   │
    ▼                                   ▼
┌──────────────────────────────────────────────┐
│    Data Transformation (Lambdas)             │
│                                              │
│  • Clean & normalize session data            │
│  • Process weather observations              │
│  • → S3 Silver Layer                         │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Feature Engineering (Lambda + DuckDB)       │
│                                              │
│  • Pre-race driver features                  │
│  • Circuit history & pace metrics            │
│  • Weather-adjusted statistics               │
│  • → S3 Gold Layer                           │
└──────────────────────────────────────────────┘
    │
    ├─ (Qualifying Complete) ─────────────────┐
    │                                          │
    ▼                                          ▼
┌─────────────────┐           ┌──────────────────────┐
│  Model Training │           │    Inference         │
│  (Lambda)       │           │    (Lambda)          │
│                 │           │                      │
│ • Load Gold     │           │ • Load ONNX model    │
│   Features      │           │ • Fetch race weather │
│ • Train         │           │ • Preprocess features│
│   XGBoost       │           │ • Generate predictions
│ • Export ONNX   │           │ • → DynamoDB + S3    │
│ • → S3 Models   │           │                      │
└─────────────────┘           └──────────────────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │  Results Dashboard   │
                            │  (Lambda)            │
                            │                      │
                            │  • Compile metrics   │
                            │  • Generate HTML     │
                            │  • → S3 Reports      │
                            └──────────────────────┘

Data Flow: FastF1 & Open-Meteo → S3 Bronze → S3 Silver → S3 Gold → Model & Inference
Metadata: DynamoDB tracks each race weekend's lineage
```

### Data Layer Architecture (Medallion Pattern)

```
S3 Bucket Structure:
├── bronze/                          # Raw ingestion
│   ├── sessions/                    # FastF1 session data
│   └── weather/                     # Open-Meteo API responses
├── silver/                          # Cleaned & normalized
│   ├── sessions/                    # Deduplicated, typed
│   └── weather/                     # Standardized format
├── gold/                            # Feature-engineered
│   ├── gold_driver_event_features/  # Training features
│   │   └── run_id=<timestamp>/
│   └── training_dataset.parquet     # Labeled dataset
├── models/                          # ML artifacts
│   └── run_id=<timestamp>/
│       ├── xgb_model.ubj
│       ├── preprocessor.joblib
│       ├── model_metadata.json
│       └── onnx_model.onnx
├── predictions/                     # Race predictions
│   └── 2026/
│       └── <event>_predictions.parquet
└── reports/                         # Dashboards
    ├── run_summary.json
    └── run_summary.html

DynamoDB Table: f1_session_tracking
├── Partition Key: event_partition_key (e.g., "2025_Australian_Grand_Prix")
├── Sort Key: session_name_abr (Q, P1, P2, P3, R, R1, R2, etc.)
└── Attributes: bronze_path, silver_path, gold_path, status, coordinates
```

---

## Project Structure

```
Formula_1_Race_Predictions/
├── README.md                                 # This file
├── src/
│   ├── orchestrator/                         # Pipeline orchestration (EventBridge trigger)
│   │   ├── lambda_function.py                # Main orchestrator logic
│   │   ├── run_local.py                      # Local testing
│   │   └── test_payload.json
│   │
│   ├── download_session_data/                # FastF1 data ingestion
│   │   ├── lambda_function.py
│   │   ├── run_local.py
│   │   └── test_payload.json
│   │
│   ├── combine_data_into_silver/             # Session data transformation
│   │   ├── lambda_function.py
│   │   ├── run_local.py
│   │   └── test_payload.json
│   │
│   ├── fetch_historical_weather/             # Open-Meteo historical archive API
│   │   ├── lambda_function.py
│   │   ├── run_local.py
│   │   └── test_payload.json
│   │
│   ├── fetch_weather_forecast/               # Open-Meteo 16-day forecast API
│   │   ├── lambda_function.py
│   │   ├── run_local.py
│   │   └── test_payload.json
│   │
│   ├── weather_to_silver/                    # Weather data transformation
│   │   ├── lambda_function.py
│   │   ├── run_local.py
│   │   └── test_payload.json
│   │
│   ├── create_tracking_table/                # DynamoDB table initialization
│   │   ├── lambda_function.py
│   │   ├── run_local.py
│   │   └── test_payload.json
│   │
│   ├── export_dim_sessions/                  # Session dimension export
│   │   ├── lambda_function.py
│   │   ├── run_local.py
│   │   └── test_payload.json
│   │
│   ├── gold_features_duckdb/                 # Feature engineering with DuckDB
│   │   ├── gold_driver_event_features.sql    # Main feature query (420 lines)
│   │   ├── gold_training_dataset.sql         # Labeled dataset
│   │   ├── validate_rolling_features.py      # Validation
│   │   ├── lambda_function.py
│   │   ├── run_local.py
│   │   └── test_payload.json
│   │
│   ├── train/                                # Model training & ONNX export
│   │   ├── config.py                         # Pydantic config models
│   │   ├── train.py                          # XGBoost training
│   │   ├── onnx_export.py                    # ONNX conversion
│   │   ├── lambda_function.py
│   │   ├── requirements.txt                  # Python dependencies
│   │   ├── run_local.py
│   │   └── test_payload.json
│   │
│   ├── inference/                            # Prediction generation
│   │   ├── predict.py                        # ONNX inference logic
│   │   ├── lambda_function.py
│   │   ├── run_local.py
│   │   └── test_payload.json
│   │
│   ├── build_run_summary/                    # Dashboard generation
│   │   ├── lambda_function.py
│   │   ├── run_local.py
│   │   └── test_payload.json
│   │
│   ├── weather/                              # Weather utilities
│   │   ├── forecast.py                       # Open-Meteo API client
│   │   └── historical_backfill.py
│   │
│   ├── common/                               # Shared utilities
│   │   ├── circuit_coordinates.py            # 25 F1 circuits (2022-2025)
│   │   ├── delete_table.py                   # DynamoDB management
│   │   └── recreate_table.py
│   │
│   └── pipeline/                             # Legacy orchestration
│       └── ...
│
├── predictions/2026/                         # Generated predictions
│   └── *.parquet
│
├── docs/                                     # Documentation (extensible)
│   └── README.md
│
└── .gitignore                                # Excludes .env, __pycache__, *.zip
```

---

## How It Works

### Pipeline Flow (Simplified)

1. **EventBridge Trigger** (every 2 hours)
   - Invokes the Orchestrator Lambda

2. **Check for Active Race Weekends**
   - Query DynamoDB for upcoming F1 races
   - Determine if new session data available

3. **Ingest Session Data** (Parallel)
   - Download qualifying/race results via FastF1
   - Store raw data in S3 `bronze/sessions/`
   - Create DynamoDB entry tracking the session

4. **Fetch Weather Data** (Parallel)
   - Historical: Open-Meteo archive API for past qualifying days
   - Forecast: 16-day forecast for race day
   - Store in S3 `bronze/weather/`

5. **Transform to Silver Layer**
   - Normalize data types and column names
   - Handle missing values
   - Deduplicate records
   - Output to S3 `silver/`

6. **Engineer Gold Features**
   - Load qualifying results, past race performance
   - Calculate rolling statistics (3-year circuit history, season form)
   - Fetch weather conditions
   - Build pre-race feature vectors using DuckDB SQL
   - Output to S3 `gold/gold_driver_event_features/`

7. **Train Model** (On-demand or post-gold-build)
   - Load gold features + race outcomes
   - Fit sklearn preprocessing pipeline (imputation, scaling)
   - Train XGBoost regressor (200 trees, depth 3)
   - Export model + preprocessor to S3 `models/run_id=<timestamp>/`
   - Convert to ONNX for inference speed

8. **Generate Predictions** (When qualifying complete)
   - Load latest ONNX model from S3
   - Fetch live race forecast weather
   - Apply preprocessing pipeline to drivers' feature vectors
   - Run inference: predict finishing position for each driver
   - Write predictions to DynamoDB + S3 `predictions/`

9. **Generate Dashboard**
   - Discover all training runs
   - Compile metrics (MAE, RMSE, R²)
   - Generate JSON + interactive HTML report
   - Upload to S3 `reports/`

### Feature Set

The model predicts race finishing position using:

**Pre-Race Driver Metrics:**
- Qualifying grid position
- Average lap pace in practice
- Sprint race result (if applicable)
- Points total year-to-date

**Historical Performance:**
- Last 3 races finishing position & points
- 3-year circuit history (avg position, DNF rate)
- Head-to-head vs. teammates

**Environmental Conditions:**
- Forecasted race day temperature, humidity, precipitation
- Track condition (dry/wet/mixed)
- Wind speed & direction

**Target Variable:**
- Race finishing position (1-20, labeled from historical data)

---

## Quick Start

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Formula_1_Race_Predictions.git
   cd Formula_1_Race_Predictions
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r src/train/requirements.txt
   ```

3. **Set up AWS credentials:**
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_REGION=us-east-1
   ```

4. **Set up environment variables:**
   ```bash
   export S3_BUCKET_NAME=f1-race-prediction
   export DYNAMODB_TABLE_NAME=f1_session_tracking
   ```

5. **Test a Lambda function locally:**
   ```bash
   cd src/download_session_data
   python run_local.py
   ```

### Deploy to AWS

See [AWS Configuration](#aws-configuration) section below for detailed setup instructions.

---

## Installation & Setup

### Prerequisites

- **Python 3.9+** (3.11+ recommended for performance)
- **AWS CLI** configured with appropriate credentials
- **pip** and virtual environment support
- **AWS Account** with permissions for Lambda, S3, DynamoDB, EventBridge, CloudWatch

### Development Environment

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r src/train/requirements.txt
   ```

3. **Create a `.env` file for local development:**
   ```bash
   cat > .env << EOF
   AWS_REGION=us-east-1
   S3_BUCKET_NAME=f1-race-prediction
   DYNAMODB_TABLE_NAME=f1_session_tracking
   S3_BRONZE_PATH=bronze
   S3_SILVER_PATH=silver
   S3_GOLD_PATH=gold
   S3_MODEL_PATH=models
   S3_REPORTS_PATH=reports
   PREDICTIONS_S3_PREFIX=predictions
   EOF
   ```

### Python Dependencies

Key libraries (see `src/train/requirements.txt` for complete list):

| Package | Version | Purpose |
|---------|---------|---------|
| `boto3` | ≥1.40.0 | AWS SDK |
| `duckdb` | ≥1.4.0 | SQL feature engineering |
| `xgboost` | ≥2.1.0 | ML model training |
| `onnxruntime` | ≥1.20.0 | Fast inference |
| `pandas` | ≥2.0.0 | Data manipulation |
| `scikit-learn` | ≥1.5.0 | Preprocessing pipelines |
| `pyarrow` | ≥15.0.0 | Parquet I/O |

---

## AWS Configuration

### Prerequisites

- **AWS Account** with appropriate IAM permissions
- **AWS CLI** installed and configured: `aws configure`
- **S3 Bucket** for data lake (recommend: `f1-race-prediction`)
- **DynamoDB Table** for session tracking (recommend: `f1_session_tracking`)

### Step 1: Create S3 Bucket

```bash
aws s3 mb s3://f1-race-prediction --region us-east-1
```

### Step 2: Create DynamoDB Table

```bash
aws dynamodb create-table \
  --table-name f1_session_tracking \
  --attribute-definitions \
    AttributeName=event_partition_key,AttributeType=S \
    AttributeName=session_name_abr,AttributeType=S \
  --key-schema \
    AttributeName=event_partition_key,KeyType=HASH \
    AttributeName=session_name_abr,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

Or initialize via Lambda:
```bash
cd src/create_tracking_table
python run_local.py
```

### Step 3: Create IAM Role for Lambda

```bash
# Create trust policy
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name f1-lambda-execution-role \
  --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name f1-lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/AWSLambdaBasicExecutionRole

# Create inline policy for S3 & DynamoDB
cat > permissions.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": "arn:aws:s3:::f1-race-prediction/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:*:table/f1_session_tracking"
    }
  ]
}
EOF

aws iam put-role-policy \
  --role-name f1-lambda-execution-role \
  --policy-name f1-inline-policy \
  --policy-document file://permissions.json
```

### Step 4: Deploy Lambda Functions

For each Lambda function, create a deployment package:

```bash
cd src/download_session_data
pip install -r requirements.txt -t package/
cd package
zip -r ../function.zip .
cd ..
zip -r function.zip lambda_function.py

# Deploy
aws lambda create-function \
  --function-name f1-download-session-data \
  --runtime python3.11 \
  --role arn:aws:iam::<ACCOUNT_ID>:role/f1-lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip \
  --timeout 300 \
  --memory-size 1024 \
  --environment Variables={S3_BUCKET_NAME=f1-race-prediction,DYNAMODB_TABLE_NAME=f1_session_tracking}
```

Repeat for each Lambda in the pipeline.

### Step 5: Create EventBridge Schedule

```bash
aws scheduler create-schedule \
  --name f1-orchestrator-2hourly \
  --schedule-expression "rate(2 hours)" \
  --target "arn:aws:scheduler:::aws-sdk:lambda:invoke" \
  --flexible-time-window '{"Mode": "OFF"}' \
  --target-role-arn arn:aws:iam::<ACCOUNT_ID>:role/f1-eventbridge-invoke-role \
  --payload '{"FunctionName": "f1-orchestrator"}'
```

### Step 6: Environment Variables

Set environment variables for all Lambda functions:

```bash
aws lambda update-function-configuration \
  --function-name f1-download-session-data \
  --environment Variables='{
    S3_BUCKET_NAME=f1-race-prediction,
    DYNAMODB_TABLE_NAME=f1_session_tracking,
    S3_BRONZE_PATH=bronze,
    S3_SILVER_PATH=silver,
    S3_GOLD_PATH=gold,
    AWS_REGION=us-east-1
  }'
```

---

## Usage

### Run Orchestrator Manually

Trigger the pipeline at any time:

```bash
aws lambda invoke \
  --function-name f1-orchestrator \
  --payload '{}' \
  response.json

cat response.json
```

### Test Individual Lambda Functions

Each module includes `run_local.py` for testing:

```bash
# Test data ingestion
cd src/download_session_data
python run_local.py

# Test feature engineering
cd src/gold_features_duckdb
python run_local.py

# Test model training
cd src/train
python run_local.py
```

### Check CloudWatch Logs

```bash
# View recent orchestrator logs
aws logs tail /aws/lambda/f1-orchestrator --follow

# View specific function logs
aws logs tail /aws/lambda/f1-inference --follow
```

### Query DynamoDB for Session Status

```bash
aws dynamodb scan --table-name f1_session_tracking

# Or query specific race
aws dynamodb get-item \
  --table-name f1_session_tracking \
  --key '{"event_partition_key":{"S":"2025_Australian_Grand_Prix"},"session_name_abr":{"S":"Q"}}'
```

### Download Predictions from S3

```bash
# List all predictions
aws s3 ls s3://f1-race-prediction/predictions/ --recursive

# Download specific race predictions
aws s3 cp s3://f1-race-prediction/predictions/2026/Australian_Grand_Prix_predictions.parquet ./predictions/
```

### View Model Dashboard

```bash
# Download HTML report
aws s3 cp s3://f1-race-prediction/reports/run_summary.html ./

# Open in browser
open run_summary.html
```

---

## Requirements

### Core Dependencies

```
boto3>=1.40.0
duckdb>=1.4.0
joblib>=1.4.0
numpy>=2.0.0
pandas>=2.0.0
pyarrow>=15.0.0
pyyaml>=6.0.0
pydantic>=2.0.0
scikit-learn>=1.5.0,<2.0.0
skl2onnx>=1.17.0
onnxmltools>=1.12.0
onnxruntime>=1.20.0
xgboost>=2.1.0,<3.0.0
fastf1>=3.0.0
requests>=2.31.0
```

### External APIs (Free & Public)

- **FastF1** - F1 official data API (no authentication required)
- **Open-Meteo** - Weather forecast & historical API (free, no authentication)

### AWS Services

- **Lambda** - Serverless compute
- **S3** - Data lake storage
- **DynamoDB** - Session metadata
- **EventBridge** - Scheduled triggers
- **CloudWatch** - Logging & monitoring

---

## Technical Details

### Model Architecture

**XGBoost Regressor:**
- **Objective:** `reg:squarederror` (predict finishing position)
- **Estimators:** 200 trees
- **Max Depth:** 3
- **Learning Rate:** 0.05
- **Subsample:** 0.9
- **Colsample by Tree:** 0.9
- **Input Features:** 50+ pre-race + historical features
- **Output:** Float (1-20, finishing position)

**Preprocessing Pipeline:**
- SimpleImputer for missing values (strategy: mean)
- StandardScaler for numerical features
- Fitted on training data, applied to inference features

**ONNX Export:**
- Converted using `onnxmltools` for cross-platform inference
- Targets opset 12 for compatibility
- Inference speed: ~1ms per prediction (vs. XGBoost library)

### Feature Engineering (DuckDB)

Main query: `src/gold_features_duckdb/gold_driver_event_features.sql`

**Feature Categories:**

1. **Qualifying Performance** (5 features)
   - Grid position, Q3 progression, practice pace

2. **Race Pace Metrics** (8 features)
   - Average lap time, pit stop count, tyre strategy

3. **Rolling History** (12 features)
   - Last 3 races: position, points, DNF
   - 3-year circuit average: position, points, DNF rate

4. **Season Form** (5 features)
   - Cumulative points, races completed, point-scoring races

5. **Weather Conditions** (6 features)
   - Temperature, humidity, precipitation, wind, track type

6. **Event Context** (4 features)
   - Constructor, driver experience, sprint race indicator

### Data Quality & Validation

- **Missing Data:** Handled via SimpleImputer (mean strategy)
- **Outliers:** Capped to 3-sigma using sklearn robust scalers
- **Validation:** Rolling window test on historical races
- **Monitoring:** CloudWatch metrics for inference latency, prediction variance

### Scalability & Cost

- **Concurrent Drivers:** 20 predictions per race (λ runtime: 200-500ms)
- **Storage:** ~500MB/season (bronze + silver + gold)
- **Cost Estimate:** ~$20-50/month for full season automation (Lambda + DynamoDB + S3)
- **Scaling:** Auto-scales to zero when no races active

---

## Contributing

This is a **public showcase repository**. While contributions are appreciated, the primary goal is to demonstrate best practices in serverless ML pipelines.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/your-feature`
3. **Make your changes** with clear commit messages
4. **Add tests** for new functionality
5. **Submit a pull request** with a description of your changes

### Areas for Contribution

- ✅ Model improvements (feature engineering, hyperparameter tuning)
- ✅ Performance optimizations (Lambda memory/timeout tuning)
- ✅ Documentation enhancements
- ✅ Bug reports & fixes
- ✅ Additional weather data sources
- ✅ Visualization dashboards

### Code Style

- Python 3.9+ compatible
- PEP 8 formatting
- Type hints where practical
- Clear variable names & docstrings

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Data Attribution

- **FastF1:** Copyright © FastF1 contributors. Data sourced from official F1 API.
- **Open-Meteo:** Free weather data. Attribution appreciated but not required.
- **Formula 1:** All F1 data is proprietary to Formula 1. This project uses public APIs for educational purposes.

---

## Acknowledgments

- **FastF1** - Excellent F1 data API wrapper
- **Open-Meteo** - Free & comprehensive weather data API
- **XGBoost** - Industry-standard gradient boosting library
- **AWS** - Serverless infrastructure enabling cost-effective automation
- **Formula 1** - For providing the most exciting motorsport on the planet 🏁

---

## Contact & Support

For questions, issues, or feedback:

- **GitHub Issues:** [Open an issue](https://github.com/yourusername/Formula_1_Race_Predictions/issues)
- **GitHub Discussions:** [Join the conversation](https://github.com/yourusername/Formula_1_Race_Predictions/discussions)

---

## Disclaimer

**Predictions are for entertainment purposes only.** This model is trained on historical data and cannot account for all real-world variables (pit crew performance, mechanical failures, driver skill variance, safety cars, etc.). Never rely solely on these predictions for betting, financial decisions, or serious analysis.

---

**Last Updated:** February 2026
**Status:** Active development during F1 season (March-December)

