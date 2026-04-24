# Titanic Dataset Analysis (AI Assignment 2)

## Project Objective
Build a predictive pipeline for Titanic survival by performing:
- Data cleaning
- Feature engineering
- Feature selection

This project is organized for GitHub submission and follows the assignment rubric.

## Project Structure

```text
titanic_assignment/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── feature_selection.py
├── README.md
└── requirements.txt
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place Kaggle Titanic files in `data/`:
- `train.csv`
- `test.csv` (optional but recommended)

## How To Run

From inside `titanic_assignment/scripts/`:

```bash
python data_cleaning.py
python feature_engineering.py
python feature_selection.py --top-k 20 --run-rfe
```

Generated files are written to `../data/` (e.g., `train_cleaned.csv`, `train_features.csv`, `feature_importance.csv`).

## Approach Summary

### Part 1: Data Cleaning
- Missing value handling:
  - `Age`: imputed using median grouped by `Sex` and `Pclass`
  - `Fare`: imputed using median grouped by `Pclass`
  - `Embarked`: imputed using mode
  - Missing indicator flags created: `AgeWasMissing`, `FareWasMissing`, `CabinWasMissing`, `EmbarkedWasMissing`
- Outlier handling:
  - `Age` and `Fare` capped using IQR-based clipping
- Data consistency:
  - Standardized `Sex` values to `male`/`female`/`unknown`
  - Removed duplicates
- Deliverable:
  - `data/train_cleaned.csv`

### Part 2: Feature Engineering
Engineered features:
- `FamilySize = SibSp + Parch + 1`
- `IsAlone = 1 if FamilySize == 1 else 0`
- `Title` extracted from `Name` and grouped (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`)
- `Deck` extracted from `Cabin`
- `AgeGroup` (`Child`, `Teen`, `Adult`, `Senior`)
- `FarePerPerson = Fare / FamilySize`
- Interaction features: `Pclass_Fare`, `Age_Pclass`
- Transformations: `Fare_log`, `Age_log`
- Encoding:
  - One-hot encoding for `Sex`, `Embarked`, `Title`, `Deck`, `AgeGroup`
  - `Pclass` kept numeric (ordinal)
- Scaling:
  - StandardScaler applied to numeric feature columns

### Part 3: Feature Selection
- Correlation analysis removes highly correlated columns (threshold configurable, default `0.9`)
- Random Forest ranks remaining features
- Top-K selected features exported (`selected_features.txt`)
- Optional RFE (`--run-rfe`) for extra credit

## Notebook Expectations
The notebook `notebooks/Titanic_Feature_Engineering.ipynb` should include:
- Missing value analysis and cleaning decisions
- Visualizations (distribution, boxplots, correlation heatmap)
- Feature creation steps with sample rows
- Feature selection justification
- Final list of selected features and key observations

## Key Findings (Fill After Running)
- Most predictive features:
  - (Update from `data/feature_importance.csv`)
- High-impact engineered features:
  - (Update after analysis)
- Correlation-based drops:
  - (Update from `data/selected_features.txt`)

## Suggested Git Commit Flow
1. `chore: scaffold titanic assignment project structure`
2. `feat: add data cleaning pipeline`
3. `feat: add feature engineering pipeline`
4. `feat: add feature selection pipeline`
5. `docs: add assignment README and usage guide`
6. `notebook: add analysis and visualizations`
