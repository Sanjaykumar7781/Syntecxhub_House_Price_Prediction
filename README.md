# House Price Prediction

This project trains a simple machine learning model to predict house prices from a Kaggle dataset. The workflow is contained in `app.ipynb`, and the trained model artifact is saved as `house_price_model.pkl`.

## Project Files

- `app.ipynb` - Jupyter notebook for downloading data, preprocessing, training, evaluation, visualization, and model export.
- `house_price_model.pkl` - saved `sklearn.linear_model.LinearRegression` model trained on log-transformed prices.
- `.venu` - local environment notes for Python version, dependencies, dataset, and model file.

## Dataset

The notebook downloads this Kaggle dataset through `kagglehub`:

```text
sheemazain/house-price-predication
```

The raw data includes fields such as price, bedrooms, bathrooms, square footage, location, year built, renovation year, and other house attributes.

## What The Notebook Does

1. Imports the required Python libraries.
2. Downloads the dataset using `kagglehub`.
3. Cleans the data by removing invalid prices, high-price outliers, missing values, and selected text columns.
4. Adds engineered features:
   - `renovated`
   - `total_sqft`
   - `bed_bath_ratio`
5. One-hot encodes categorical columns such as city/state fields.
6. Selects features based on correlation with price.
7. Splits the data into train and test sets.
8. Scales features using `StandardScaler`.
9. Trains a `LinearRegression` model on log-transformed prices.
10. Evaluates the model and saves it with `joblib`.

## Requirements

Use Python 3.13 or a compatible Python 3 version.

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn kagglehub joblib scikit-learn
```

## How To Run

1. Open the project folder.
2. Install the dependencies listed above.
3. Open `app.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
4. Run the notebook cells from top to bottom.
5. After training, the model will be saved as:

```text
house_price_model.pkl
```

## Current Model

The saved model is a scikit-learn `LinearRegression` model with 81 coefficients. In the latest notebook output, the model reported:

- Train R2 on log price: `0.8036`
- Test R2 on actual price: `0.7213`
- RMSE: `120951.24`

## Notes

- The model is trained on log-transformed prices, so predictions are converted back with `np.exp(...)`.
- The saved pickle currently stores only the regression model. For production-style reuse, also save the scaler and feature column list so new inputs can be transformed exactly like the training data.
- The notebook includes visualization cells for feature correlation, actual vs predicted price, perfect prediction comparison, and error distribution.
