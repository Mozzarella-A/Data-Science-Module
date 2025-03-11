import numpy as np
import pandas as pd
import dask.dataframe as dd
import time
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import os
from PIL import Image

# Load datasets with error handling
try:
    data1 = pd.read_csv('Trips_by_Distance.csv')
    data2 = pd.read_csv('Trips_Full_Data.csv')
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
    exit(1)

# Data cleaning: Remove duplicates and handle missing values
data1 = data1.drop_duplicates().dropna()
data2 = data2.drop_duplicates().dropna()

# Standardising column names and convert date columns to datetime
data1.columns = data1.columns.str.strip().str.lower().str.replace(' ', '_')
data2.columns = data2.columns.str.strip().str.lower().str.replace(' ', '_')
data1['date'] = pd.to_datetime(data1['date'], errors='coerce')
data2['date'] = pd.to_datetime(data2['date'], errors='coerce')

# Optimise memory usage by converting numeric columns to appropriate types
for col in data1.select_dtypes(include=['int64', 'float64']).columns:
    data1[col] = pd.to_numeric(data1[col], downcast='integer')
for col in data2.select_dtypes(include=['int64', 'float64']).columns:
    data2[col] = pd.to_numeric(data2[col], downcast='integer')

# ----------------------- Processing Time Comparison -----------------------

def measure_processing_time():
    """
    Compare processing time between Pandas (sequential) and Dask (parallel) for tasks a and b.
    Task a: Calculate weekly average of people staying at home.
    Task b: Identify peak travel dates with >10M trips for 10-25 and 50-100 miles.
    """
    # Sequential processing with Pandas
    start_time = time.time()
    staying_home_total, traveling_population = analyze_home_vs_travel(silent=True)
    high_travel_10_25, high_travel_50_100 = identify_peak_travel_dates(silent=True)
    pandas_time = time.time() - start_time

    # Parallel processing with Dask across different CPU counts
    processor_counts = [2, 4, 8, 10, 20, 30, 40]
    dask_times = {}

    for num_workers in processor_counts:
        dask_df = dd.from_pandas(data1, npartitions=num_workers)
        dask_df_full = dd.from_pandas(data2, npartitions=num_workers)
        start_time = time.time()
        staying_home_mean = dask_df.groupby('week')['population_staying_at_home'].mean().compute()
        peak_travel_10_25 = dask_df_full[dask_df_full['trips_10-25_miles'] > 10_000_000][['date', 'trips_10-25_miles']].compute()
        peak_travel_50_100 = dask_df_full[dask_df_full['trips_50-100_miles'] > 10_000_000][['date', 'trips_50-100_miles']].compute()
        dask_times[num_workers] = time.time() - start_time

    # Visualize processing times
    labels = ['Pandas (Sequential)'] + [f'Dask ({w} CPUs)' for w in processor_counts]
    times = [pandas_time] + [dask_times[w] for w in processor_counts]
    
    plt.figure(figsize=(14, 8))
    colors = ['royalblue', 'forestgreen', 'mediumseagreen', 'dodgerblue', 'darkcyan', 'cadetblue', 'teal']
    bars = plt.bar(labels, times, color=colors, alpha=0.85, edgecolor='black')
    plt.xticks(rotation=20, fontsize=13, ha='right', fontweight='bold')
    plt.xlabel("Processing Method", fontsize=15, fontweight="bold", labelpad=12)
    plt.ylabel("Execution Time (seconds)", fontsize=15, fontweight="bold", labelpad=12)
    plt.title("Comparison of Processing Time: Pandas vs Dask", fontsize=18, fontweight="bold", pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("execution_time.png", dpi=300, bbox_inches='tight')
    plt.close()

# ----------------------- Question 1 (Home vs. Travel Distance) -----------------------

def analyze_home_vs_travel(silent=False):
    """
    Calculate the weekly average of people staying at home and the mean traveling population
    using data from Trips_Full_Data.csv with Dask for parallel computation.
    """
    potential_columns = {
        'week': ['week', 'week_of_date'],
        'population_staying_at_home': ['population_staying_at_home', 'staying_at_home_population', 'home_population'],
        'people_not_staying_at_home': ['people_not_staying_at_home', 'traveling_population', 'away_population']
    }
    selected_cols = {}

    for key, options in potential_columns.items():
        for option in options:
            if option in data2.columns:
                selected_cols[key] = option
                break

    dask_df = dd.from_pandas(data2, npartitions=4)
    staying_home_weekly = dask_df.groupby(selected_cols['week'])[selected_cols['population_staying_at_home']].mean().compute()
    traveling_population = dask_df[selected_cols['people_not_staying_at_home']].mean().compute()

    if not silent:
        return staying_home_weekly.mean(), traveling_population
    return staying_home_weekly.mean(), traveling_population

# ----------------------- Question 2 (Identifying Peak Travel Dates) -----------------------

def identify_peak_travel_dates(silent=False):
    """
    Identify dates with more than 10 million trips for 10-25 miles and 50-100 miles
    using data from Trips_Full_Data.csv.
    """
    required_columns = ['date', 'trips_10-25_miles', 'trips_50-100_miles']
    if not all(col in data2.columns for col in required_columns):
        return pd.DataFrame(), pd.DataFrame()

    data2['date'] = pd.to_datetime(data2['date'], errors='coerce')

    high_travel_10_25 = data2[data2['trips_10-25_miles'] > 10_000_000][['date', 'trips_10-25_miles']]
    high_travel_50_100 = data2[data2['trips_50-100_miles'] > 10_000_000][['date', 'trips_50-100_miles']]

    if not silent:
        return high_travel_10_25, high_travel_50_100
    return high_travel_10_25, high_travel_50_100

# ----------------------- Generate Graphs -----------------------

def Travelling_vs_Distance():
    """
    Generate a histogram of average trips per week by distance category
    using data from Trips_Full_Data.csv.
    """
    df = data2.copy()
    df = df.drop_duplicates(subset=['week_of_date'])

    distance_columns = [
        'trips_<1_mile', 'trips_1-3_miles', 'trips_3-5_miles', 'trips_5-10_miles',
        'trips_10-25_miles', 'trips_25-50_miles', 'trips_50-100_miles',
        'trips_100-250_miles', 'trips_250-500_miles', 'trips_500+_miles'
    ]

    df[distance_columns] = df[distance_columns].apply(pd.to_numeric, errors='coerce')

    travel_totals = df.groupby('week_of_date')[distance_columns].mean()
    labels = ["<1 mile", "1-3 miles", "3-5 miles", "5-10 miles", "10-25 miles",
              "25-50 miles", "50-100 miles", "100-250 miles", "250-500 miles", "500+ miles"]

    plt.figure(figsize=(12, 6))
    plt.hist(range(len(labels)), bins=len(labels), weights=travel_totals.mean().values, color='royalblue', alpha=0.8, edgecolor='black')
    plt.xlabel("Distance Traveled")
    plt.ylabel("Average Number of Trips per Week")
    plt.title("Histogram of Trips vs Distance")
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("Histogram_Travel_vs_Distance.png", dpi=300, bbox_inches='tight')
    plt.close()

def identify_peak_travel_dates_graphs():
    """
    Generate scatter plots for peak travel days (>10M trips) for 10-25 and 50-100 miles
    using data from Trips_Full_Data.csv.
    """
    high_travel_10_25, high_travel_50_100 = identify_peak_travel_dates()

    plt.figure(figsize=(12, 6))
    plt.scatter(high_travel_10_25['date'], high_travel_10_25['trips_10-25_miles'] / 1_000_000, 
                color='blue', alpha=0.7, label="Trips (10-25 miles)")
    plt.axhline(y=10, color='gray', linestyle='--', label='10M threshold')
    plt.xlabel("Date")
    plt.ylabel("Trips (Millions)")
    plt.title("Peak Travel Days (10-25 miles)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("s_peak_travel_10_25.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.scatter(high_travel_50_100['date'], high_travel_50_100['trips_50-100_miles'] / 1_000_000, 
                color='red', alpha=0.7, label="Trips (50-100 miles)")
    plt.axhline(y=10, color='gray', linestyle='--', label='10M threshold')
    plt.xlabel("Date")
    plt.ylabel("Trips (Millions)")
    plt.title("Peak Travel Days (50-100 miles)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("s_peak_travel_50_100.png", dpi=300, bbox_inches='tight')
    plt.close()

def Staying_Home_vs_Week():
    """
    Generate a histogram of average people staying at home per week
    using data from Trips_by_Distance.csv with Pandas.
    """
    # Use data1 (Trips_by_Distance.csv)
    if 'week' not in data1.columns or 'population_staying_at_home' not in data1.columns:
        return

    # Convert columns to numeric, handling potential issues
    data1['week'] = pd.to_numeric(data1['week'], errors='coerce')
    data1['population_staying_at_home'] = pd.to_numeric(data1['population_staying_at_home'], errors='coerce')

    # Group by week and calculate mean (national average per week), drop NaN
    weekly_mean = data1.groupby('week')['population_staying_at_home'].mean().dropna()

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(weekly_mean.index, bins=len(weekly_mean.index), weights=weekly_mean.values / 1_000, 
             color='royalblue', alpha=0.8, edgecolor='black')
    plt.xlabel("Week")
    plt.ylabel("Average People Staying at Home (Thousands)")
    plt.title("Histogram of Average People Staying at Home by Week")
    plt.xticks(ticks=weekly_mean.index, labels=[f"W{int(w)}" for w in weekly_mean.index], rotation=90)
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("Histogram_Staying_Home_vs_Week.png", dpi=300, bbox_inches='tight')
    plt.close()

def travel_frequency_prediction():
    """
    Develop and visualize a polynomial regression model to predict travel frequency for 5-10 mile trips in Week 32.
    """
    data1 = pd.read_csv('Trips_by_Distance.csv').drop_duplicates().dropna()
    data2 = pd.read_csv('Trips_Full_Data.csv').drop_duplicates().dropna()

    data1.columns = data1.columns.str.strip().str.lower().str.replace(' ', '_')
    data2.columns = data2.columns.str.strip().str.lower().str.replace(' ', '_')

    if 'week_of_date' in data2.columns:
        data2['week_of_date'] = data2['week_of_date'].str.replace('Week ', '', regex=True).astype(float)

    df_week_32_1 = data1[data1['week'] == 32]
    df_week_32_2 = data2[data2['week_of_date'] == 32].groupby('week_of_date').agg({
        'trips_1-25_miles': 'sum',
        'trips_25-100_miles': 'sum'
    }).reset_index()

    if df_week_32_1.empty or df_week_32_2.empty:
        return

    df_merged = df_week_32_1.copy()
    df_merged['trips_1-25_miles'] = df_week_32_2['trips_1-25_miles'].iloc[0]
    df_merged['trips_25-100_miles'] = df_week_32_2['trips_25-100_miles'].iloc[0]

    X = df_merged[['trips_1-25_miles', 'trips_25-100_miles', 'number_of_trips_1-3', 'number_of_trips_10-25']]
    y = df_merged['number_of_trips_5-10']

    if X.empty or y.empty or len(X) != len(y):
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)

    mse_poly = mean_squared_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    poly_cv_scores = cross_val_score(poly_model, X_train_poly, y_train, cv=5)

    # Visualize Polynomial Regression results
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_test, color='green', alpha=0.7, label="Actual", marker='o')
    plt.scatter(y_test, y_pred_poly, color='orange', alpha=0.7, label="Predicted (Polynomial)", marker='x')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linewidth=2, linestyle='dashed', label="Ideal Fit (y=x)")
    plt.xlabel('Actual Number of Trips (5-10 miles)')
    plt.ylabel('Predicted Number of Trips')
    plt.title('Actual vs Predicted Travel Frequency (Week 32) with Polynomial Features')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("Travel_Frequency_Prediction_Week32_Poly_Enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()

def total_trips_by_distance():
    """
    Generate a bar plot of total trips by distance category, normalized by weeks
    using data from Trips_Full_Data.csv.
    """
    trip_columns = [
        'trips_<1_mile', 'trips_1-3_miles', 'trips_3-5_miles', 'trips_5-10_miles',
        'trips_10-25_miles', 'trips_25-50_miles', 'trips_50-100_miles',
        'trips_100-250_miles', 'trips_250-500_miles', 'trips_500+_miles'
    ]

    existing_columns = [col for col in trip_columns if col in data2.columns]
    df_selected = data2[existing_columns]

    num_weeks = data2["date"].nunique()
    total_trips = df_selected.sum()

    if num_weeks > 1:
        total_trips = total_trips / num_weeks

    plt.figure(figsize=(12, 6))
    bars = plt.bar(total_trips.index, total_trips.values, width=0.6, color='royalblue', alpha=0.8)
    plt.xlabel("Distance Traveled")
    plt.ylabel("Total Number of Trips (Averaged per Week if Needed)")
    plt.title("Total Number of Trips Per Category")
    plt.xticks(rotation=45, ha='right')
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("Total_Trips_By_Distance.png", dpi=150, bbox_inches='tight', format='png', transparent=False)
    plt.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    measure_processing_time()
    Travelling_vs_Distance()
    identify_peak_travel_dates_graphs()
    Staying_Home_vs_Week()
    total_trips_by_distance()
    travel_frequency_prediction()