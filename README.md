# Matchoracle Prediction App

Welcome to the **Matchoracle Prediction App**! This FastAPI-based machine learning app is designed to offer predictions and model training functionalities for match data analysis. It provides endpoints to train models, make predictions, and retrieve performance metrics.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
  - [POST /train](#post-train)
  - [POST /predict](#post-predict)
  - [GET /metrics](#get-metrics)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features
- **Model Training**: Train various models for match prediction.
- **Model Prediction**: Generate predictions for match outcomes.
- **Metrics Calculation**: Evaluate model performance with detailed metrics.
- **Authentication**: Integrate with user authentication for secure access.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/felixkpt/matchoracle-predictions.git
    cd matchoracle-predictions
    ```

2. Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate     # For Windows
    pip install -r requirements.txt
    ```

3. Set up environment variables. Create a `.env` file based on the `.env.example` file:
    ```bash
    cp .env.example .env
    ```

4. Run the application:
    ```bash
    uvicorn app.main:app --reload
    ```

## Usage

### POST /train
This endpoint initiates the model training process.

#### Example Request Body
```json
{
  "competition_id": 12345,
  "prediction_type": "regular",
  "target": null,
  "prefer_saved_matches": true,
  "is_grid_search": false,
  "retrain_if_last_train_is_before": "2024-08-01",
  "ignore_trained": true,
  "per_page": 1000,
  "job_id": 1234
}
```

#### Response
```json
{
  "message": "Model training started successfully."
}
```

### POST /predict
This endpoint generates predictions for match outcomes based on the trained models.

#### Example Request Body
```json
{
  "target": null,
  "competition_id": 12345,
  "last_predict_date": null,
  "prediction_type": "regular",
  "from_date": "2024-11-01",
  "to_date": "2024-11-30",
  "target_match": null,
  "job_id": 1234
}
```

#### Response
```json
{
  "message": "Predictions started successfully."
}
```

### GET /metrics
This endpoint retrieves the metrics for a specific model or prediction, this is handy during initial app models setup.

#### Response
```json
{
  "accuracy": 0.95,
  "precision": 0.93,
  "recall": 0.91
}
```

## Project Structure

```
├── app
│   ├── auth                 # Authentication logic
│   ├── configs              # Configuration files (e.g., settings, logger)
│   ├── helpers              # Utility functions and helper modules
│   ├── metrics.py           # Metrics calculation logic
│   ├── model_metrics.py     # Model performance evaluation
│   ├── predictions          # Prediction logic for different match types
│   ├── predictions_normalizers  # Data normalization for predictions
│   ├── predict.py           # Prediction API endpoint
│   ├── requests             # Request schema validation
│   ├── train.py             # Model training logic
│   └── trained_models       # Folder storing trained models
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
└── startup.sh               # Script for starting the app
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit them (`git commit -am 'Add feature'`).
4. Push to your forked branch (`git push origin feature-name`).
5. Open a pull request.

---

This app integrates multiple machine learning models and provides a seamless API for prediction and training workflows. Make sure to adjust the `.env` file for authentication and customize the features for your own needs.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## Author

**Felix Kiptoo Biwott**  
[GitHub Repository (Laravel Backend)](https://github.com/felixkpt/matchoracle-be)  
[GitHub Repository (FastAPI Predictions)](https://github.com/felixkpt/matchoracle-predictions)

---
