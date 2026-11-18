# PathSnap – GPS Trajectory Map Matching

## About the Project

PathSnap is a machine learning-based GPS map-matching project that identifies the most likely road segments followed by a vehicle.

GPS locations may contain errors or may not lie exactly on roads. PathSnap uses machine learning and a Hidden Markov Model (HMM) to identify suitable road segments and reconstruct a likely route.

## Objectives

- Process GPS trajectory and road-network data.
- Generate candidate road segments for GPS observations.
- Extract useful geographic, temporal and directional features.
- Train and compare multiple machine learning models.
- Perform hyperparameter tuning using Bayesian optimisation.
- Reconstruct routes using an HMM and the Viterbi algorithm.
- Compare reconstructed routes with reference routes.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Optuna
- Joblib
- Folium
- Jupyter Notebook

## Machine Learning Models

The following five models are implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours (KNN)
4. Random Forest
5. Gradient Boosting

Random Forest was selected for the HMM route-reconstruction stage based on its initial F1-score.

## Project Workflow

1. Dataset collection and organisation.
2. GPS and road-network preprocessing.
3. Candidate road-segment generation.
4. Feature engineering.
5. Machine learning model training.
6. Model comparison and evaluation.
7. Bayesian hyperparameter optimisation.
8. HMM-based route reconstruction.
9. Route evaluation and visualisation.

## Project Structure

```text
PathSnap/
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── student_test/
│   ├── processed/
│   └── features/
│
├── notebooks/
│   ├── 01_dataset_inventory.ipynb
│   ├── 02_gps_eda.ipynb
│   ├── 03_road_network_eda.ipynb
│   ├── 04_route_analysis.ipynb
│   └── 05_folium_visualization.ipynb
│
├── src/
│   ├── candidates/
│   ├── feature_engineering/
│   ├── models/
│   ├── preprocessing/
│   └── visualizations/
│
├── outputs/
│   └── model_results/
│
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SHREYAS-606/PathSnap.git
```

### 2. Open the project directory

```bash
cd PathSnap
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

## Running the Project

Run the following commands from the project root directory.

### 1. Generate candidates

```bash
python -m src.candidates.generate_candidates
```

### 2. Build features

```bash
python -m src.feature_engineering.build_features
```

### 3. Train and compare models

```bash
python -m src.models.train_all_models
```

### 4. Perform Bayesian optimisation

```bash
python -m src.models.tune_random_forest
```

### 5. Perform HMM route reconstruction

```bash
python -m src.models.hmm_route_reconstruction
```

**Note:** Run each stage after its required input files have been generated. The commands above follow the project workflow; check the individual scripts for their exact input and output requirements.

## Results

- Five machine learning models were trained and compared.
- Random Forest achieved an initial F1-score of approximately 0.730.
- Gradient Boosting achieved an initial accuracy of approximately 0.915.
- Bayesian optimisation was performed using Optuna.
- HMM route reconstruction was completed for 10 student-testing trips.
- A total of 19,160 GPS observations were reconstructed.
- Average unique-arc overlap precision: approximately 82.99%.
- Average reference arc coverage: approximately 89.32%.

## Output Files

The project generates:

- Trained machine learning models.
- Model comparison results.
- Bayesian optimisation trial results.
- Student-testing predictions.
- Reconstructed route CSV files.
- HMM route-reconstruction summary.
- Route visualisation outputs.

## Team Members

- S SHREYAS KUMAR – PES2UG24CS656
- R SAI PARANAV REDDY – PES2UG24CS651
- KEERTHANA M – PES2UG24CS638
- RAMYA – PES2UG24CS653
