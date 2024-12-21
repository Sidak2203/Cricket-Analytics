import requests
import sns
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import panel as pn
import gc
import os
import hvplot.pandas  # For Panel-compatible visualizations
import mplcursors
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans


def get_player_names(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    tables = pd.read_html(page.text)
    player_table = tables[2]

    player_names = player_table['Player'].tolist()

    cleaned_player_names = [name.split(' (')[0] for name in player_names]

    return cleaned_player_names

def search_player(player_name):
    search_url = 'https://search.espncricinfo.com/ci/content/site/search.html?search=' + player_name.lower().replace(' ', '%20')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'TE': 'Trailers'
    }

    page = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    player_links = [link for link in soup.find_all('a', href=True) if '/cricketers/' in link['href']]

    return player_links

def extract_name_from_link(link):
    return link['href'].split('/')[-1].rsplit('-', 1)[0]

def match_name(player_name, extracted_name):
    player_name_parts = player_name.lower().split()
    extracted_name_parts = extracted_name.split('-')

    if len(player_name_parts) < 2 or len(extracted_name_parts) < 2:
        return False

    player_first_initial = player_name_parts[0][0]
    player_last_name = player_name_parts[-1]

    extracted_first_initial = extracted_name_parts[0][0]
    extracted_last_name = extracted_name_parts[-1]

    return (player_first_initial == extracted_first_initial and player_last_name == extracted_last_name)

def get_player_id(player_name):
    player_links = search_player(player_name)

    if not player_links:
        last_name = player_name.split()[-1]
        player_links = search_player(last_name)

    if player_links:
        for link in player_links:
            extracted_name = extract_name_from_link(link)
            if match_name(player_name, extracted_name):
                profile_url = link['href']
                if not profile_url.startswith('https://'):
                    profile_url = 'https://www.espncricinfo.com' + profile_url
                player_id = profile_url.split('/')[-1].split('-')[-1]
                return player_id, profile_url

        link = player_links[0]
        profile_url = link['href']
        if not profile_url.startswith('https://'):
            profile_url = 'https://www.espncricinfo.com' + profile_url
        player_id = profile_url.split('/')[-1].split('-')[-1]
        return player_id, profile_url

    return f'Player id for {player_name} not found', None

def get_player_details(profile_url):
    print(f"Fetching player details from URL: {profile_url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'TE': 'Trailers'
    }

    page = requests.get(profile_url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    parent_tag = soup.find('div', {'class': 'ds-p-0'})
    details = {}
    if parent_tag:
        full_name_tag = parent_tag.find('h1', {'class': 'ds-text-title-l ds-font-bold'})
        country_and_role_tags = parent_tag.find_all('span', {'class': 'ds-text-comfortable-s'})

        details['Full Name'] = full_name_tag.text.strip() if full_name_tag else "N/A"
        #Limit loop to first element (country)
        for idx, tag in enumerate(country_and_role_tags):
            if idx > 1:  # Skip the first element (index 0)
                continue
            text = tag.text.strip()
            if text in ["Batter", "Top order Batter", "Middle order Batter", "Opening Batter", "Bowler", "Allrounder", "Wicketkeeper Batter", "Batting Allrounder", "Bowling Allrounder"]:
                details['Playing Role'] = text
            else:
                details['Country'] = text

    info_labels = ["Age", "Batting Style", "Bowling Style", "Fielding Position"]
    for label in info_labels:
        label_tag = soup.find('p', string=label)
        if label_tag:
            value_tag = label_tag.find_next('span')
            if value_tag:
                details[label] = value_tag.text.strip()
    return details

def get_players_info(all_players):
    data = []
    for player_name in all_players:
        player_id, profile_url = get_player_id(player_name)
        if player_id:
            player_details = get_player_details(profile_url)
            player_details['Player Name'] = player_name
            player_details['Player ID'] = player_id
            player_details['Profile URL'] = profile_url
            data.append(player_details)
        else:
            data.append({'Player Name': player_name, 'Player ID': None, 'Profile URL': None, 'Full Name': None, 'Country': None, 'Playing Role': None, 'Age': None, 'Batting Style': None, 'Bowling Style': None, 'Fielding Position': None})
    print(data)
    return pd.DataFrame(data)


from requests.exceptions import ConnectTimeout, HTTPError, RequestException


def get_player_stats(player_id, type_):
    """
    Fetch player statistics based on player ID and type (batting or bowling).
    Includes error handling for connection issues and HTTP errors.
    """
    url = f'https://stats.espncricinfo.com/ci/engine/player/{player_id}.html?class=3;template=results;type={type_};view=innings'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    print(f"Fetching data from URL: {url}")

    # Create a session with headers
    session = requests.Session()
    session.headers.update(headers)

    try:
        # Attempt to retrieve the page with a timeout
        page = session.get(url, timeout=10)

        # Check if the request was successful
        if page.status_code == 200:
            df_list = pd.read_html(page.text)
            df = df_list[3]
            return df
        else:
            print(f"Failed to retrieve page. Status code: {page.status_code}")
            return None

    except ConnectTimeout:
        print(f"Connection timed out for URL: {url}")
        return None

    except HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None

    except RequestException as e:
        print(f"Request exception occurred: {e}")
        return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def get_and_save_player_stats(player_name, player_types):
    print(player_name,player_types)
    player_id, _ = get_player_id(player_name)  # Only grab the player_id from the tuple
    not_found_players = []  # List to store players not found

    if player_id == (f'Player id for {player_name} not found'):
        # Retry with last name only if full name fails
        player_id, _ = get_player_id(player_name.split()[-1])  # Last name
    if player_id != (f'Player id for {player_name} not found'):
        for type_ in player_types:
            df = get_player_stats(player_id, type_)
            if df is not None:
                df_name = f"{player_name.replace(' ', '_')}_{type_}"  # Replace spaces with underscores
                globals()[df_name] = df
                print(f"Saved stats for {player_name} ({type_})")
            else:
                print(f"Could not retrieve {type_} stats for {player_name}")
    else:
        print(f"Player ID for {player_name} not found even after retrying with last name.")
        not_found_players.append(player_name)
    return not_found_players

def clean_batter_data(df):
    df = df.copy()

    # Replace unwanted strings with NaN
    df.replace(['DNB', 'TDNB', '-', 'NA'], np.nan, inplace=True)

    # Clean numeric columns
    if df['Runs'].dtype != 'object':
        df['Runs'] = df['Runs'].astype(str)
    df['Runs'] = df['Runs'].str.replace('*', '', regex=False)
    df['Runs'] = pd.to_numeric(df['Runs'], errors='coerce')
    df['SR'] = pd.to_numeric(df['SR'], errors='coerce')
    df['BF'] = pd.to_numeric(df['BF'], errors='coerce')
    df['4s'] = pd.to_numeric(df['4s'], errors='coerce')
    df['6s'] = pd.to_numeric(df['6s'], errors='coerce')
    df['Pos'] = pd.to_numeric(df['Pos'], errors='coerce')

    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')

    df.dropna(axis=1, how='all', inplace=True)

    return df

def clean_bowler_data(df):
    df = df.copy()

    # Clean numeric columns
    df['Econ'] = pd.to_numeric(df['Econ'], errors='coerce')
    df['Wkts'] = pd.to_numeric(df['Wkts'], errors='coerce')
    df['Mdns'] = pd.to_numeric(df['Mdns'], errors='coerce')
    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')

    # Drop any column where all values are NaN
    df.dropna(axis=1, how='all', inplace=True)

    return df

def calculate_batter_impact(row):
    position = row['Pos']
    if pd.isna(position) or pd.isna(row['Runs']):
        return np.nan

    runs = row['Runs']
    strike_rate = row['SR']
    fours = row['4s']
    sixes = row['6s']
    balls_faced = row['BF']

    # Maximum possible impact (assuming 100 runs at 200 strike rate)
    max_impact = (0.70 * 100) + (0.30 * 200)

    if position in [1, 2]:  # Opener
        impact = (0.80 * runs) + (0.40 * strike_rate) + (0.50 * (fours + sixes))
    elif position == 3:  # Number 3
        impact = (0.90 * runs) + (0.25 * strike_rate) + (0.25 * balls_faced)
    elif position in [4, 5]:  # Middle-order
        impact = (0.70 * runs) + (0.40 * strike_rate) + (0.60 * (fours + sixes))
    elif position in [6, 7, 8, 9, 10]:  # Lower-order
        impact = (0.30 * runs) + (0.50 * strike_rate) + (3 * sixes) + (1 * fours)
    else:
        impact = np.nan

    # Normalize impact to a scale of 0-100
    normalized_impact = (impact / max_impact) * 100

    # Cap the impact at 100
    impact_rating = min(normalized_impact, 100)
    return impact_rating

def calculate_bowler_impact(row):
    if pd.isna(row['Wkts']) or pd.isna(row['Econ']) or pd.isna(row['Mdns']):
        return np.nan

    wickets = row['Wkts']
    economy_rate = row['Econ']
    maidens = row['Mdns']

    # Maximum possible impact is achieved when bowler takes 5 wickets and has an economy rate of 3
    max_impact = 0.70 * 5 + 0.30 * (10 - 3)

    if maidens > 0:
        impact = (0.65 * wickets) + (0.50 * (10 - economy_rate)) + (0.50 * maidens)
    else:
        impact = (0.65 * wickets) + (0.50 * (10 - economy_rate))

    # Set impact to 0 if it’s negative
    impact = max(impact, 0)

    # Normalize impact to a scale of 0-100
    normalized_impact = (impact / max_impact) * 100
    # Cap the impact at 100
    impact_rating = min(normalized_impact, 100)
    return impact_rating

def clean_and_calculate_impact(batter_df, bowler_df):
    # Clean the data
    batter_df_clean = clean_batter_data(batter_df)
    bowler_df_clean = clean_bowler_data(bowler_df)

    # Calculate impact for batter
    batter_df_clean['Impact'] = batter_df_clean.apply(calculate_batter_impact, axis=1)

    # Calculate impact for bowler
    bowler_df_clean['Impact'] = bowler_df_clean.apply(calculate_bowler_impact, axis=1)

    # Remove rows with NaN values in 'Impact' column
    batter_df_clean = batter_df_clean.dropna(subset=['Impact'])
    bowler_df_clean = bowler_df_clean.dropna(subset=['Impact'])
    return batter_df_clean, bowler_df_clean

# Create a dictionary to map playing roles to corresponding types
playing_role_map = {
    "Batter": ["batting"],
    "Top order Batter": ["batting"],
    "Middle order Batter": ["batting"],
    "Opening Batter": ["batting"],
    "Wicketkeeper Batter": ["batting"],
    "Bowler": ["bowling"],
    "Spin Bowler": ["bowling"],
    "Pace Bowler": ["bowling"],
    "Batting Allrounder": ["batting", "bowling"],
    "Bowling Allrounder": ["batting", "bowling"],
    "Allrounder": ["batting", "bowling"]
}



def add_ema_impact(df, span=10):
    df['EMA_Impact'] = df['Impact'].ewm(span=span, adjust=False).mean()
    return df


def filter_recent_data(df, min_records=30):
    if df is None or df.empty:
        return pd.DataFrame()  # Return an empty DataFrame if input is None or empty

    # Ensure 'Start Date' is a valid datetime type
    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')

    # Filter for data after the specified start year
    # df_filtered = df[df['Start Date'] >= pd.Timestamp(f'{start_year}-01-01')]
    # Check if the filtered data meets the minimum record count
    if len(df) < min_records:
        print(f"Not enough data ({len(df)} records) after 2019 for this player.")
        return pd.DataFrame()  # Return an empty DataFrame if there aren't enough records
    return df  # Return the filtered DataFrame


def split_time_series(df):
    # Ensure data is sorted by 'Start Date'
    df = df.sort_values('Start Date')

    # Calculate the number of rows for the test set (25% of the data)
    test_size = int(0.30 * len(df))

    # Randomly sample indices for the test set based on 'EMA_Impact' values
    test_indices = np.random.choice(df['EMA_Impact'].index, size=test_size, replace=False)

    # Separate training and test data based on the random indices and extract 'EMA_Impact'
    train_data = df[~df.index.isin(test_indices)]['EMA_Impact']
    test_data = df[df.index.isin(test_indices)][['Start Date', 'EMA_Impact']]  # Keep 'Start Date' for plotting
    return train_data, test_data


# Trained ARIMA model with enforce_stationarity and enforce_invertibility disabled
def train_arima(train_data):
    model = ARIMA(train_data, order=(5, 1, 0), enforce_stationarity=False, enforce_invertibility=False)  # ARIMA(p,d,q)
    model_fit = model.fit()
    return model_fit


# Predict future impact using the ARIMA model
def predict_arima(model, steps=5):
    predictions = model.forecast(steps=steps)
    return predictions


def evaluate_model(test_data, predictions, metrics_list):
    dates = test_data['Start Date']
    test_data_np = test_data['EMA_Impact'].values.flatten()

    # Ensure that the predictions and test data have the same length
    min_len = min(len(test_data_np), len(predictions))
    predictions = predictions[:min_len]  # Trim predictions if necessary
    test_data_np = test_data_np[:min_len]  # Trim test data if necessary
    dates = dates[:min_len]  # Trim dates to the same length

    # Apply quantile filtering to exclude top 5% of extreme errors
    absolute_errors = np.abs(test_data_np - predictions)
    filtered_indices = absolute_errors < np.quantile(absolute_errors, 0.90)  # Exclude top 5%
    filtered_test_data = test_data_np[filtered_indices]
    filtered_predictions = predictions[filtered_indices]

    # Core metrics calculations on filtered data
    mse = mean_squared_error(filtered_test_data, filtered_predictions)
    mae = mean_absolute_error(filtered_test_data, filtered_predictions)
    rmse = np.sqrt(mse)
    r2 = (r2_score(filtered_test_data, filtered_predictions))

    # Additional metrics calculations on filtered data
    mape = np.mean(np.abs((filtered_test_data - filtered_predictions) / filtered_test_data)) * 100
    smape = 100 / len(filtered_test_data) * np.sum(2 * np.abs(filtered_predictions - filtered_test_data) /
                                                   (np.abs(filtered_test_data) + np.abs(filtered_predictions)))
    mbd = abs(np.mean(filtered_predictions - filtered_test_data))
    explained_variance = abs(explained_variance_score(filtered_test_data, filtered_predictions))

    # Append only non-NaN values to metrics_list
    metrics = {}
    if not np.isnan(mse):
        metrics["MSE"] = mse
    if not np.isnan(mae):
        metrics["MAE"] = mae
    if not np.isnan(rmse):
        metrics["RMSE"] = rmse
    if not np.isnan(r2):
        metrics["R-squared"] = r2
    if not np.isnan(mape):
        metrics["MAPE"] = mape
    if not np.isnan(smape):
        metrics["SMAPE"] = smape
    if not np.isnan(mbd):
        metrics["MBD"] = mbd
    if not np.isnan(explained_variance):
        metrics["Explained Variance"] = explained_variance
    metrics_list.append(metrics)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R-squared: {r2:.4f}")
    print(f"Test MAPE: {mape:.4f}")
    print(f"Test SMAPE: {smape:.4f}")
    print(f"Test MBD: {mbd:.4f}")
    print(f"Test Explained Variance: {explained_variance:.4f}")

    # # Plot Actual vs Predicted
    # plt.figure(figsize=(10, 5))
    # plt.plot(dates, test_data_np, label="Actual", marker="o")
    # plt.plot(dates, predictions, label="Predicted", marker="x")
    # plt.title("Actual vs Predicted Impact Values")
    # plt.xlabel("Date")
    # plt.ylabel("Impact Value")
    # plt.xticks(rotation=45)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Return the metrics dictionary
    return metrics


# # Function to calculate average metrics across all players
def calculate_average_metrics(all_metrics):
    avg_metrics = {}
    metric_keys = ["MSE", "MAE", "RMSE", "R-squared", "MAPE", "SMAPE", "MBD", "Explained Variance"]

    # Initialize each metric key to calculate average
    for key in metric_keys:
        avg_metrics[key] = np.mean([metrics.get((key), np.nan) for metrics in all_metrics if key in metrics])

    return avg_metrics

def get_most_common_position(batter_df):
    if 'Pos' in batter_df.columns:
        return int(batter_df['Pos'].mode()[0])  # Get the most frequent position
    else:
        return None  # If no position data, return None

def bowling_performance(predicted_impact):
    max_impact = 0.70 * 5 + 0.30 * (10 - 3)  # Max impact is 5 wickets and economy rate of 3

    # Scale predicted impact back to match performance
    scaled_impact = (predicted_impact / 100) * max_impact

    wickets = 0
    economy_rate = 10

    # Conditional logic for wickets and economy rate based on impact
    if 1 <= predicted_impact <= 25:
        wickets = np.random.randint(0, 2)  # Wickets between 0 and 1
        economy_rate = 10 - ((scaled_impact - (0.65 * wickets)) / 0.50)
        if not (8 <= economy_rate <= 20):  # Economy rate between 8 and 20
            economy_rate = np.random.randint(8, 21)

    elif 26 <= predicted_impact <= 50:
        wickets = np.random.randint(0, 4)  # Wickets between 0 and 3
        economy_rate = 10 - ((scaled_impact - (0.65 * wickets)) / 0.50)
        if not (7 <= economy_rate <= 15):  # Economy rate between 7 and 15
            economy_rate = np.random.randint(7, 16)

    elif 51 <= predicted_impact <= 75:
        wickets = np.random.randint(1, 5)  # Wickets between 1 and 5
        economy_rate = 10 - ((scaled_impact - (0.65 * wickets)) / 0.50)
        if not (4 <= economy_rate <= 10):  # Economy rate between 4 and 10
            economy_rate = np.random.randint(4, 11)

    elif 76 <= predicted_impact <= 100:
        wickets = np.random.randint(2, 8)  # Wickets between 2 and 7
        economy_rate = 10 - ((scaled_impact - (0.65 * wickets)) / 0.50)
        if not (3 <= economy_rate <= 8):  # Economy rate between 3 and 8
            economy_rate = np.random.randint(3, 9)

    else:
        wickets = 0
        economy_rate = np.random.randint(10, 21)  # ER between 10 and 20 for impact outside the range

    economy_rate = round(economy_rate, 2)

    # Calculate total runs conceded
    runs_conceded = int(economy_rate * 4)  # Assuming 4 overs bowled
    overs = 4

    return f"{wickets}-{runs_conceded} in {overs}"

def batting_performance(predicted_impact, position, max_iterations=30):
    runs = 0
    sr = 0
    iterations = 0

    # Define weights for positions
    if position in [1, 2]:  # Opener
        run_weight = 0.75
        sr_weight = 0.30
    elif position == 3:  # Number 3
        run_weight = 0.95
        sr_weight = 0.15
    elif position in [4, 5]:  # Middle-order
        run_weight = 0.70
        sr_weight = 0.35
    elif position in [6, 7, 8, 9, 10]:  # Lower-order
        run_weight = 0.30
        sr_weight = 0.70
    else:
        raise ValueError("Invalid position")

    # Maximum possible impact (normalization reference)
    max_impact = (run_weight * 100) + (sr_weight * 200)
    normalized_impact = (predicted_impact / 100) * max_impact

    while iterations < max_iterations:
        iterations += 1
        if position in [1, 2, 3, 4]:
            # Logic for batters 1-4
            if 1 <= predicted_impact <= 25:
                runs = np.random.randint(5, 16)
                sr = int((normalized_impact - (run_weight * runs)) / sr_weight)
                if 5 <= sr <= 150:
                    break
            elif 26 <= predicted_impact <= 50:
                runs = np.random.randint(10, 46)
                sr = int((normalized_impact - (run_weight * runs)) / sr_weight)
                if 50 <= sr <= 200:
                    break
            elif 51 <= predicted_impact <= 75:
                runs = np.random.randint(25, 81)
                sr = int((normalized_impact - (run_weight * runs)) / sr_weight)
                if 90 <= sr <= 250:
                    break
            elif 76 <= predicted_impact <= 100:
                runs = np.random.randint(40, 101)
                sr = int((normalized_impact - (run_weight * runs)) / sr_weight)
                if 130 <= sr <= 300:
                    break
            else:
                runs = 0
                sr = 0
                break
        else:
            # Logic for lower-order batters
            if 1 <= predicted_impact <= 25:
                sr = np.random.randint(50, 101)
                runs = int(abs((normalized_impact - (sr_weight * sr)) / run_weight))
                if 1 <= runs <= 20:
                    break
            elif 26 <= predicted_impact <= 50:
                sr = np.random.randint(100, 141)
                runs = int(abs((normalized_impact - (sr_weight * sr)) / run_weight))
                if 10 <= runs <= 40:
                    break
            elif 51 <= predicted_impact <= 75:
                sr = np.random.randint(120, 171)
                runs = int(abs((normalized_impact - (sr_weight * sr)) / run_weight))
                if 15 <= runs <= 60:
                    break
            elif 76 <= predicted_impact <= 100:
                sr = np.random.randint(150, 251)
                runs = int(abs((normalized_impact - (sr_weight * sr)) / run_weight))
                if 20 <= runs <= 100:
                    break
            else:
                runs = 0
                sr = 0
                break

        runs = max(0, int(runs))
        sr = max(10, int(sr))

    if iterations >= max_iterations:
        print(f"Unable to find a correct prediction, try again.")
        return None, None

    return runs, sr


#Analysis for all players

# Function to plot top players in each category
def plot_top_players(position_categories, top_n=10):
    for category, players in position_categories.items():
        if players:
            # Sort players by impact in descending order and select the top N
            top_players = sorted(players, key=lambda x: x[1], reverse=True)[:top_n]
            player_names, impact_values = zip(*top_players)

            # Plot the top players for the current category
            plt.figure(figsize=(10, 6))
            plt.barh(player_names, impact_values, color='skyblue')
            plt.xlabel("Impact Rating")
            plt.ylabel("Player Name")
            plt.title(f"Top {top_n} Batters - {category.capitalize()}")
            plt.gca().invert_yaxis()  # Invert y-axis to have the highest impact at the top
            plt.show()






def top_player_analysis(all_players):
    # After ensuring all_players is populated, proceed to analyze batters
    print("Analyzing batter impact ratings based on positions...")

    # Populate position_categories with player impact ratings by position
    position_categories = {
        'overall': [],
        'opener': [],
        'position_3': [],
        'position_4': [],
        'finisher': []
    }

    # Loop through each batter to categorize their impact based on position
    for player_name, (df_prefix, roles) in all_players.items():
        if 'batting' in roles:
            formatted_name = df_prefix.replace(" ", "_")
            batter_df_name = f"{formatted_name}_batting"
            print(batter_df_name)
            try:
                batter_df = globals()[batter_df_name]
                batter_df_clean = clean_batter_data(batter_df)
                batter_df_clean = filter_recent_data(batter_df_clean)
                if batter_df_clean.empty:
                    continue

                batter_df_clean['Impact'] = batter_df_clean.apply(calculate_batter_impact, axis=1)
                batter_df_clean = batter_df_clean.dropna(subset=['Impact'])

                # Determine the player's most common batting position
                most_common_position = get_most_common_position(batter_df_clean)
                avg_impact = batter_df_clean['Impact'].mean()

                # Classify by position
                position_categories['overall'].append((player_name, avg_impact))
                if most_common_position in [1, 2]:
                    position_categories['opener'].append((player_name, avg_impact))
                elif most_common_position == 3:
                    position_categories['position_3'].append((player_name, avg_impact))
                elif most_common_position == 4:
                    position_categories['position_4'].append((player_name, avg_impact))
                elif most_common_position in [5, 6]:
                    position_categories['finisher'].append((player_name, avg_impact))

            except KeyError:
                print(f"Data for {batter_df_name} not found.")


    # Execute the visualizations
    plot_top_players(position_categories, top_n=10)

P_Test_list = ["Mahedi Hasan", "Andre Russell", "Liam Livingstone", "Charith Asalanka", "James Neesham","Marcus Stoinis", "Daryl Mitchell", "Ben Stokes", "Akeal Hosein", "Romario Shepherd","Mitchell Marsh", "Glenn Maxwell", "Gulbadin Naib", "Kieron Pollard", "Afif Hossain","Sam Curran", "Axar Patel", "Hardik Pandya", "Kagiso Rabada", "Sheldon Cottrell", 'Danushka Gunathilaka','Lockie Ferguson', "Washington Sundar", "Lungi Ngidi", "Asghar Afghan", "Colin de Grandhomme", "Taskin Ahmed","Lasith Malinga", "Andile Phehlukwayo", "Shoriful Islam", "Jason Holder", "Chris Jordan","Dasun Shanaka", "Andre Russell", "Angelo Mathews", "Wayne Parnell", "Soumya Sarkar","Chris Gayle", "Adam Milne", "Mohammad Nawaz", "Colin de Grandhomme", "Faheem Ashraf","Josh Hazlewood", "Ravindra Jadeja", "Shoaib Malik", "Azmatullah Omarzai", "Dwayne Bravo","David Willey", "Andile Phehlukwayo", "Chamika Karunaratne", "Mahmudullah", "Tim Southee","Mustafizur Rahman", "Mohammad Nabi", "Mitchell Starc", "Tabraiz Shamsi", "Ish Sodhi","Anrich Nortje", "Pat Cummins", "Chris Gayle", "Shreyas Iyer", "Najibullah Zadran"]

def countrywise_analysis(all_players):

    # Define the list of top cricketing countries
    top_countries = ["India", "England", "Australia", "New Zealand", "West Indies",
                     "Sri Lanka", "Bangladesh", "Pakistan", "South Africa", "Afghanistan"]

    # Initialize empty DataFrames for each country and store them in dictionaries
    batters_dfs = {country: pd.DataFrame(columns=['Player', 'Average_Impact', 'Opponent Country', 'Batting Style']) for
                   country in top_countries}
    bowlers_dfs = {country: pd.DataFrame(columns=['Player', 'Average_Impact', 'Opponent Country', 'Bowling Style']) for
                   country in top_countries}
    overall_dfs = {country: pd.DataFrame(columns=['Player', 'Average_Impact', 'Opponent Country', 'Role']) for country
                   in top_countries}

    # Loop through each player in all_players dictionary
    for player_name, (df_prefix, roles) in all_players.items():
        formatted_name = df_prefix.replace(" ", "_")
        player_info = players_info_df.loc[players_info_df['Full Name'] == player_name]

        # Check if player information is available in players_info_df
        if not player_info.empty:
            player_country = player_info['Country'].values[0]

            # Check if player is a batter
            if 'batting' in roles:
                batter_df_name = f"{formatted_name}_batting"

                try:
                    batter_df = globals()[batter_df_name]
                    batter_df_clean = clean_batter_data(batter_df)

                    if batter_df_clean.empty:
                        print(f"No data after filtering for {player_name} as batter. Skipping...")
                        continue

                    batter_df_clean['Impact'] = batter_df_clean.apply(calculate_batter_impact, axis=1)
                    batter_df_clean = batter_df_clean.dropna(subset=['Impact'])

                    batting_style = player_info['Batting Style'].values[0]
                    nationality = player_info['Country'].values[0]

                    # Loop through each country and add data for batters
                    for country in top_countries:
                        country_matches = batter_df_clean[
                            batter_df_clean['Opposition'].str.contains(country, case=False, na=False)]

                        if len(country_matches) >= 3:
                            avg_impact = country_matches['Impact'].mean()
                            new_row = pd.DataFrame({
                                'Player': [player_name],
                                'Average_Impact': [avg_impact],
                                'Opponent Country': [country],
                                'Batting Style': [batting_style],
                                'Nationality': [nationality],
                                'Role': ['Batter']
                            })
                            # Add data to batters and overall DataFrames
                            batters_dfs[country] = pd.concat([batters_dfs[country], new_row[
                                ['Player', 'Average_Impact', 'Opponent Country', 'Batting Style', 'Nationality']]],
                                                             ignore_index=True)
                            overall_dfs[country] = pd.concat([overall_dfs[country], new_row[
                                ['Player', 'Average_Impact', 'Opponent Country', 'Role', 'Nationality']]],
                                                             ignore_index=True)
                        else:
                            print(f"{player_name} has less than 3 matches against {country} as batter. Skipping...")

                except KeyError as e:
                    print(f"KeyError for {batter_df_name}: {e}. Skipping {player_name} as batter.")
                except Exception as e:
                    print(f"An error occurred for {player_name} as batter: {e}")

            # Check if player is a bowler
            if 'bowling' in roles:
                bowler_df_name = f"{formatted_name}_bowling"

                try:
                    bowler_df = globals()[bowler_df_name]
                    bowler_df_clean = clean_bowler_data(bowler_df)

                    if bowler_df_clean.empty:
                        print(f"No data after filtering for {player_name} as bowler. Skipping...")
                        continue

                    bowler_df_clean['Impact'] = bowler_df_clean.apply(calculate_bowler_impact, axis=1)
                    bowler_df_clean = bowler_df_clean.dropna(subset=['Impact'])

                    bowling_style = player_info['Bowling Style'].values[0]
                    nationality = player_info['Country'].values[0]

                    # Loop through each country and add data for bowlers
                    for country in top_countries:
                        country_matches = bowler_df_clean[
                            bowler_df_clean['Opposition'].str.contains(country, case=False, na=False)]

                        if len(country_matches) >= 3:
                            avg_impact = country_matches['Impact'].mean()
                            new_row = pd.DataFrame({
                                'Player': [player_name],
                                'Average_Impact': [avg_impact],
                                'Opponent Country': [country],
                                'Bowling Style': [bowling_style],
                                'Nationality': [nationality],
                                'Role': ['Bowler']
                            })
                            # Add data to bowlers and overall DataFrames
                            bowlers_dfs[country] = pd.concat([bowlers_dfs[country], new_row[
                                ['Player', 'Average_Impact', 'Opponent Country', 'Bowling Style', 'Nationality']]],
                                                             ignore_index=True)
                            overall_dfs[country] = pd.concat([overall_dfs[country], new_row[
                                ['Player', 'Average_Impact', 'Opponent Country', 'Role', 'Nationality']]],
                                                             ignore_index=True)
                        else:
                            print(f"{player_name} has less than 3 matches against {country} as bowler. Skipping...")

                except KeyError as e:
                    print(f"KeyError for {bowler_df_name}: {e}. Skipping {player_name} as bowler.")
                except Exception as e:
                    print(f"An error occurred for {player_name} as bowler: {e}")

    print("\nAll players processed.")

    # Example of viewing results for India
    print("\nBatters against India:")
    print(batters_dfs['India'])

    print("\nBowlers against India:")
    print(bowlers_dfs['India'])

    print("\nOverall against India:")
    print(overall_dfs['India'])
    player_vs_countries(batters_dfs, bowlers_dfs)


def player_vs_countries(batters_dfs,bowlers_dfs):
    # Initialize empty dictionaries to store best performers for each country
    best_batters = {}
    best_bowlers = {}

    # Loop through each opponent in batters_dfs
    for opponent, batter_df in batters_dfs.items():
        # Initialize an empty DataFrame to store best batter per nationality for each opponent
        best_batter_df = pd.DataFrame(
            columns=['Player', 'Average_Impact', 'Opponent Country', 'Batting Style', 'Nationality'])

        # Group by Nationality and find the batter with the highest Average_Impact
        for nationality, group in batter_df.groupby('Nationality'):
            best_batter = group.loc[group['Average_Impact'].idxmax()]
            best_batter_df = pd.concat([best_batter_df, pd.DataFrame([best_batter])], ignore_index=True)

        # Save the results for this opponent
        best_batters[opponent] = best_batter_df

    # Repeat the same for bowlers_dfs to get best bowler per nationality
    for opponent, bowler_df in bowlers_dfs.items():
        # Initialize an empty DataFrame to store best bowler per nationality for each opponent
        best_bowler_df = pd.DataFrame(
            columns=['Player', 'Average_Impact', 'Opponent Country', 'Bowling Style', 'Nationality'])

        # Group by Nationality and find the bowler with the highest Average_Impact
        for nationality, group in bowler_df.groupby('Nationality'):
            best_bowler = group.loc[group['Average_Impact'].idxmax()]
            best_bowler_df = pd.concat([best_bowler_df, pd.DataFrame([best_bowler])], ignore_index=True)

        # Save the results for this opponent
        best_bowlers[opponent] = best_bowler_df

    # Print example output for each opponent's best batter and bowler per nationality
    for opponent in best_batters:
        print(f"Best Batters vs {opponent}:")
        print(best_batters[opponent])
        print("\n")

    for opponent in best_bowlers:
        print(f"Best Bowlers vs {opponent}:")
        print(best_bowlers[opponent])
        print("\n")


def analysis_all_players():
    try:
        # Check if all_players already has data
        if 'all_players' in globals():
            # If all_players exists and is not empty, skip further processing
            print("Players already exist in the dictionary, skipping extraction.")
        else:
            # If all_players exists but is empty, initialize and populate it
            raise NameError("Dictionary 'all_players' is empty.")
    except NameError:
        # Get the current date in the required format (e.g., "1+Dec+2024")
        current_date = datetime.now().strftime("%d+%b+%Y")  # Example: "13+Dec+2024"

        # Update the URLs dynamically with the current date
        batting_url = f'https://stats.espncricinfo.com/ci/engine/stats/index.html?batting_positionmax2=7;batting_positionmin2=1;batting_positionval2=batting_position;class=3;filter=advanced;orderby=runs;qualmin1=5;qualval1=innings;size=200;spanmax1={current_date};spanmin1=1+Dec+2019;spanval1=span;team=1;team=2;team=25;team=3;team=4;team=40;team=5;team=6;team=7;team=8;template=results;type=batting'
        bowling_url = f'https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;filter=advanced;orderby=wickets;qualmin2=5;qualval2=innings_bowled;size=200;spanmax1={current_date};spanmin1=1+Dec+2019;spanval1=span;team=1;team=2;team=25;team=3;team=4;team=40;team=5;team=6;team=7;team=8;template=results;type=bowling'

        # Get player names
        batting_players = get_player_names(batting_url)
        bowling_players = get_player_names(bowling_url)

        # Combine both lists and remove duplicates
        all_players_list = list(set(batting_players + bowling_players))
        print(all_players_list)
        # Get player information
        global players_info_df
        players_info_df = get_players_info(all_players_list)
        print(players_info_df)
        print('This is done')
        # Create a dictionary to map playing roles to corresponding types
        playing_role_map = {
            "Batter": ["batting"],
            "Top order Batter": ["batting"],
            "Middle order Batter": ["batting"],
            "Opening Batter": ["batting"],
            "Wicketkeeper Batter": ["batting"],
            "Bowler": ["bowling"],
            "Spin Bowler": ["bowling"],
            "Pace Bowler": ["bowling"],
            "Batting Allrounder": ["batting", "bowling"],
            "Bowling Allrounder": ["batting", "bowling"],
            "Allrounder": ["batting", "bowling"]
        }
        # Create the dictionary of players with their types
        all_players = {}
        for index, row in players_info_df.iterrows():
            player_name = row["Full Name"]
            playing_role = row["Playing Role"]
            player_type = playing_role_map.get(playing_role, None)
            if player_type:
                all_players[player_name] = (player_name, player_type)

        # Loop through all player categories and fetch stats
        not_found_players = []
        for player_name, (player_name, types) in all_players.items():
            not_found_players.extend(get_and_save_player_stats(player_name, types))

        if not_found_players:
            print("\nPlayers for whom data was not found:")
            for player in not_found_players:
                print(player)

    top_player_analysis(all_players)
    countrywise_analysis(all_players)
    return("Analysis for all players completed.")
def calculate_recent_form(player_df, months=12):
    # Filter for recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    recent_df = player_df[player_df['Start Date'] >= start_date].copy()

    # Handle case where no data is available for the recent period
    if recent_df.empty:
        start_date = end_date - timedelta(days=365)  # Extend period to 1 year
        recent_df = player_df[player_df['Start Date'] >= start_date].copy()
        if recent_df.empty:  # If still empty, return a message
            return None, pn.pane.Markdown("#### Not enough data for the selected time period.")

    # Calculate EMA Impact
    recent_df['EMA_Impact'] = recent_df['Impact'].ewm(span=5).mean()

    # Format 'Start Date' for x-axis in dd-mm-yy format
    recent_df['Formatted Date'] = recent_df['Start Date'].dt.strftime('%d-%m-%y')

    # Generate Matplotlib plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot EMA Impact (only this line)
    ax.plot(recent_df['Formatted Date'], recent_df['EMA_Impact'],
            label='EMA Impact', marker='o', color='darkorange', linewidth=3, linestyle='-', markersize=8)

    # Add labels and grid
    ax.set_title(f"Recent Form over Last {months} Months", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date (dd-mm-yy)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Impact", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)  # Fixed y-axis range
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add hover functionality
    cursor = mplcursors.cursor(ax.lines, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"Date: {recent_df.iloc[sel.index]['Formatted Date']}\n"
        f"EMA Impact: {sel.artist.get_ydata()[sel.index]:.2f}"))

    # Rotate date labels on x-axis to prevent overlap
    plt.xticks(rotation=45, ha='right')

    # Save plot to a buffer for rendering in Panel
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    # Remove unwanted columns
    columns_to_remove = ['EMA_Impact','Formatted Date']
    if 'Mins' in recent_df.columns:
        columns_to_remove.append('Mins')
    if 'Unnamed: 13' in recent_df.columns:
        columns_to_remove.append('Unnamed: 13')
    recent_df = recent_df.drop(columns=columns_to_remove, errors='ignore')

    return recent_df, pn.pane.PNG(buf, width=600)


# Function to get Favorite and Worst Opponent
# Function to get Favorite and Worst Opponent
def get_favorite_and_worst_opponent(player_df):
    # Total matches played by the player
    total_matches = len(player_df)

    # Group by opponent and count number of matches played against each opponent
    opponent_match_count = player_df['Opposition'].value_counts()

    # Filter opponents with at least 1/20th of the player's total matches
    valid_opponents = opponent_match_count[opponent_match_count >= total_matches / 20].index

    # Filter the player_df to consider only valid opponents
    filtered_df = player_df[player_df['Opposition'].isin(valid_opponents)]

    # Calculate average impact by opponent
    opponent_impact = filtered_df.groupby('Opposition')['Impact'].mean()

    # Get the favorite and worst opponent based on impact
    favorite_opponent = opponent_impact.idxmax()
    worst_opponent = opponent_impact.idxmin()

    # Generate plot
    fig, ax = plt.subplots(figsize=(12, 6))  # Increase figure size for better spacing
    sns.barplot(x=opponent_impact.index, y=opponent_impact.values, palette='coolwarm', ax=ax)

    # Highlight favorite and worst opponent with dashed lines
    ax.axhline(y=opponent_impact[favorite_opponent], color='green', linestyle='--',
               label=f'Favorite: {favorite_opponent}')
    ax.axhline(y=opponent_impact[worst_opponent], color='red', linestyle='--', label=f'Worst: {worst_opponent}')

    # Title and labels with larger font sizes
    ax.set_title('Average Impact by Opponent', fontsize=16, fontweight='bold')
    ax.set_xlabel('Opponents', fontsize=14)
    ax.set_ylabel('Average Impact', fontsize=14)

    # Improve x-axis label readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',
                       fontsize=12)  # Rotate labels for better readability
    ax.set_xticks(ax.get_xticks())  # Ensure correct spacing

    # Set y-axis limits
    ax.set_ylim(0, opponent_impact.max() + 10)

    # Improve plot aesthetics
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    plot_pane = pn.pane.PNG(buf, width=1000)

    # Return the result with favorite and worst opponent impact
    result = {
        "Favorite": favorite_opponent,
        "Favorite Impact": opponent_impact[favorite_opponent],
        "Worst": worst_opponent,
        "Worst Impact": opponent_impact[worst_opponent]
    }

    return result, plot_pane


# Function to get Favorite Venue
def get_favorite_venue(player_df):
    """Get and visualize the favorite venue based on average impact."""

    # Calculate the total number of matches played at each venue
    venue_match_count = player_df['Ground'].value_counts()

    # Consider only the top 10 venues the player has played the most
    top_venues = venue_match_count.head(10).index

    # Filter the player_df to consider only these top venues
    filtered_df = player_df[player_df['Ground'].isin(top_venues)]

    # Calculate average impact by venue for the filtered top 10 venues
    venue_impact = filtered_df.groupby('Ground')['Impact'].mean()

    # Identify the favorite venue (highest average impact)
    favorite_venue = venue_impact.idxmax()

    # Generate plot
    fig, ax = plt.subplots(figsize=(12, 6))  # Increase figure size for better spacing
    sns.barplot(x=venue_impact.index, y=venue_impact.values, palette='viridis', ax=ax)

    # Highlight the favorite venue with a dashed line
    ax.axhline(y=venue_impact[favorite_venue], color='blue', linestyle='--', label=f'Favorite: {favorite_venue}')

    # Set labels and title with larger font sizes
    ax.set_title('Average Impact by Venue', fontsize=16, fontweight='bold')
    ax.set_xlabel('Venue', fontsize=14)
    ax.set_ylabel('Average Impact', fontsize=14)

    # Improve x-axis label readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax.set_xticks(ax.get_xticks())  # Ensure correct spacing

    # Set y-axis limits
    ax.set_ylim(0, venue_impact.max() + 10)

    # Improve plot aesthetics
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    plot_pane = pn.pane.PNG(buf, width=1000)

    # Return the result with favorite venue impact
    result = {
        "Favorite": favorite_venue,
        "Impact": venue_impact[favorite_venue]
    }

    return result, plot_pane


import plotly.express as px
from io import BytesIO

# Extension for Panel
pn.extension()


# Function to get Top 5 Performances with interactive plot
# Function to get Top 5 Performances with interactive plot

# Function to get Top 5 Performances with Matplotlib plot
# Function to get Top 5 Performances and visualize them with Scatter Plot

def get_top_performances(player_df, top_n=5):
    """
    Get and visualize the top 5 performances using Plotly for interactivity.
    Handles both batters and bowlers dynamically.
    """
    import plotly.express as px

    # Determine the role: Batter or Bowler
    is_batter = "SR" in player_df.columns

    if is_batter:
        # Columns for batters
        columns_to_plot = ['Runs', 'SR', '4s', '6s', 'Impact', 'Opposition', 'Ground', 'Start Date']
        x_axis = 'Runs'
        y_axis = 'SR'
        x_label = "Runs"
        y_label = "Strike Rate"
        title = "Top 5 Performances (Runs vs SR)"
    else:
        # Columns for bowlers
        columns_to_plot = ['Overs', 'Mdns', 'Runs', 'Wkts', 'Econ', 'Impact', 'Opposition', 'Ground', 'Start Date']
        x_axis = 'Wkts'
        y_axis = 'Econ'
        x_label = "Wickets"
        y_label = "Economy"
        title = "Top 5 Performances (Wickets vs Economy)"

    # Sort the player data by impact and take the top n performances
    top_performances = player_df.sort_values(by='Impact', ascending=False).head(top_n)

    # Ensure only relevant columns are used
    performance_data = top_performances[columns_to_plot]

    # Create a Plotly scatter plot
    fig = px.scatter(
        performance_data,
        x=x_axis,
        y=y_axis,
        size='Impact',  # Bubble size based on Impact
        color='Impact',  # Color by Impact value
        hover_name='Opposition',  # Hover details for the opposition
        hover_data={col: True for col in columns_to_plot if col != 'Impact'},
        color_continuous_scale='Viridis',  # Color palette for the Impact values
        title=title,
        labels={x_axis: x_label, y_axis: y_label},
        size_max=60,  # Increase the size of the bubbles
        template='plotly_dark'  # Optional: Dark theme for a more polished look
    )

    # Update layout for better presentation
    fig.update_layout(
        width=800,
        height=500,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False
    )

    # Return the performance data and the Plotly plot
    return performance_data, pn.pane.Plotly(fig)



def particular_player_workflow(player_name, finalize_button):
    main_display.clear()
    output, confirmed_player_dict = particular_playerdf(player_name, finalize_button)

    def perform_analysis(event):
        if not confirmed_player_dict:
            main_display.append(pn.pane.Markdown("**Error: No player confirmed. Please confirm the player first.**"))
            return

        # Perform the analysis logic
        not_found_players = []
        for player_name, (df_prefix, roles) in confirmed_player_dict.items():
            not_found_players.extend(get_and_save_player_stats(df_prefix, roles))

        if not_found_players:
            main_display.append(pn.pane.Markdown("**Players for whom data was not found:**"))
            for player in not_found_players:
                main_display.append(pn.pane.Markdown(f"- {player}"))
            return  # Skip analysis if data is missing

        for player_name, (df_prefix, roles) in confirmed_player_dict.items():
            main_display.append(pn.pane.Markdown(f"### Analyzing {player_name}"))
            formatted_name = df_prefix.replace(" ", "_")

            if "batting" in roles:
                try:
                    batter_df_name = f"{formatted_name}_batting"
                    batter_df = globals().get(batter_df_name)
                    if batter_df is None or batter_df.empty:
                        main_display.append(pn.pane.Markdown(f"**No batting data available for {player_name}.**"))
                        continue

                    batter_df_clean = clean_batter_data(batter_df)
                    batter_df_clean["Impact"] = batter_df_clean.apply(calculate_batter_impact, axis=1)
                    analysis_output = display_analysis_results(batter_df_clean, "Batting")
                    main_display.append(analysis_output)

                except Exception as e:
                    main_display.append(pn.pane.Markdown(f"**Error during batting analysis for {player_name}: {e}**"))

            if "bowling" in roles:
                try:
                    bowler_df_name = f"{formatted_name}_bowling"
                    bowler_df = globals().get(bowler_df_name)
                    if bowler_df is None or bowler_df.empty:
                        main_display.append(pn.pane.Markdown(f"**No bowling data available for {player_name}.**"))
                        continue

                    bowler_df_clean = clean_bowler_data(bowler_df)
                    bowler_df_clean["Impact"] = bowler_df_clean.apply(calculate_bowler_impact, axis=1)
                    analysis_output = display_analysis_results(bowler_df_clean, "Bowling")
                    main_display.append(analysis_output)

                except Exception as e:
                    main_display.append(pn.pane.Markdown(f"**Error during bowling analysis for {player_name}: {e}**"))

    finalize_button.on_click(perform_analysis)

    output.append(pn.pane.Markdown("### Confirm player(s) before proceeding."))
    main_display.append(output)

def particular_playerdf(player_name, finalize_button):
    output = pn.Column()
    player_name_input = pn.widgets.TextInput(name="Player Name", value=player_name)
    search_button = pn.widgets.Button(name="Search Player", button_type="primary")
    finalize_button.visible = False  # Initially hide finalize button
    reenter_button = pn.widgets.Button(name="Re-enter Player Name", button_type="warning")

    playing_role_map = {
        "Batter": ["batting"],
        "Top order Batter": ["batting"],
        "Middle order Batter": ["batting"],
        "Opening Batter": ["batting"],
        "Wicketkeeper Batter": ["batting"],
        "Bowler": ["bowling"],
        "Spin Bowler": ["bowling"],
        "Pace Bowler": ["bowling"],
        "Batting Allrounder": ["batting", "bowling"],
        "Bowling Allrounder": ["batting", "bowling"],
        "Allrounder": ["batting", "bowling"]
    }

    player_dict = {}
    players_info_df = None

    def search_player(event):
        nonlocal players_info_df, player_dict

        players_info_df = get_players_info([player_name_input.value.strip()])
        output.clear()
        player_dict.clear()

        if players_info_df.empty:
            output.append(pn.pane.Markdown(f"**No information found for: {player_name_input.value}.**"))
        else:
            output.append(pn.pane.DataFrame(players_info_df, width=600))
            output.append(pn.pane.Markdown("**Is this the correct player?**"))
            finalize_button.visible = True
            finalize_button.disabled = False  # Enable finalize button after a valid search
            output.append(finalize_button)
            output.append(reenter_button)

    def reenter_player_name(event):
        player_name_input.value = ""  # Allow editing
        finalize_button.visible = False
        finalize_button.disabled = True  # Disable finalize button again
        output.clear()
        output.append(pn.Column(player_name_input, search_button))

    def finalize_confirmation(event):
        nonlocal players_info_df, player_dict

        if players_info_df.empty:
            output.clear()
            output.append(pn.pane.Markdown("**Error: No player data available. Please search first.**"))
            return

        for index, row in players_info_df.iterrows():
            full_name = row.get("Full Name", "Unknown")
            role = playing_role_map.get(row.get("Playing Role", ""), None)
            if role:
                player_dict[full_name] = (full_name, role)

        output.clear()
        if player_dict:
            output.append(pn.pane.Markdown("**Players confirmed:**"))
            for player in player_dict:
                output.append(pn.pane.Markdown(f"- {player}"))
        else:
            output.append(pn.pane.Markdown("**Error: No players confirmed.**"))

    search_button.on_click(search_player)
    reenter_button.on_click(reenter_player_name)
    finalize_button.on_click(finalize_confirmation)

    output.append(pn.Column(player_name_input, search_button))
    return output, player_dict



def display_analysis_results(player_df_clean, role):
    analysis_output = pn.Column()
    # Calculate recent form and generate plot
    recent_form_df, recent_form_plot = calculate_recent_form(player_df_clean)
    # Append to Panel layout
    analysis_output.append(pn.pane.Markdown(f"### {role} Recent Form (EMA Impact over last 12 months/2 years)"))
    if recent_form_df is None:
        analysis_output.append(recent_form_plot)  # Display no-data message
    else:
        analysis_output.append(pn.pane.DataFrame(recent_form_df, width=800, height=300))
        analysis_output.append(recent_form_plot)


    # Favorite/Worst Opponent
    # Generate opponent analysis output
    opponent_impact_df, opponent_impact_plot = get_favorite_and_worst_opponent(player_df_clean)
    # Displaying opponent analysis
    analysis_output.append(pn.pane.Markdown(f"### Opponent Analysis"))
    analysis_output.append(opponent_impact_plot)
    # Highlight the Favorite and Worst Opponents
    favorite_text = f"**<span style='color:#AFEEEE; font-size: 16px; font-weight: bold;'>Favorite Opponent: {opponent_impact_df['Favorite']} (Impact: {opponent_impact_df['Favorite Impact']:.2f})</span>**"
    worst_text = f"**<span style='color:red; font-size: 16px; font-weight: bold;'>Worst Opponent: {opponent_impact_df['Worst']} (Impact: {opponent_impact_df['Worst Impact']:.2f})</span>**"
    # Add the formatted markdown with colors and styles
    analysis_output.append(pn.pane.Markdown(favorite_text))
    analysis_output.append(pn.pane.Markdown(worst_text))

    # Favorite Venue
    # Generate venue analysis output
    venue_impact_df, venue_impact_plot = get_favorite_venue(player_df_clean)
    # Displaying venue analysis
    analysis_output.append(pn.pane.Markdown(f"### Venue Analysis"))
    analysis_output.append(venue_impact_plot)
    # Highlight the Favorite Venue with formatted markdown
    favorite_text = f"**<span style='color:#AFEEEE; font-size: 18px; font-weight: bold;'>Favorite Venue: {venue_impact_df['Favorite']} (Impact: {venue_impact_df['Impact']:.2f})</span>**"
    # Add the formatted markdown with colors and styles
    analysis_output.append(pn.pane.Markdown(favorite_text))


    # Get the top performances and the plot
    top_performances_df, top_performances_plot = get_top_performances(player_df_clean)
    # Add the Markdown title for the analysis
    analysis_output.append(pn.pane.Markdown("### Top 5 Performances"))
    # Display the top performances table
    analysis_output.append(pn.pane.DataFrame(top_performances_df, width=800, height=300))
    # Display the interactive plot
    analysis_output.append(top_performances_plot)

    return analysis_output


def gather_player_data_from_names(player_names):
    """Fetch data for multiple players based on names."""
    player_dict = {}
    for player_name in player_names:
        player_data = particular_playerdf(player_name)
        if player_data:
            player_dict.update(player_data)
        else:
            main_display.append(pn.pane.Markdown(f"**Error: No data found for {player_name}.**"))
            return None
    return player_dict



# Helper function: Get user's choice for stats comparison
def get_compare_choice():
    return int(input("Which stats would you like to compare?\n1. Batting\n2. Bowling\n3. Both\nEnter choice: "))


# Function to analyze a specific player with input field
def show_predict_player():
    main_display.clear()
    player_input = pn.widgets.TextInput(name="Enter Player Name")
    analyze_button = pn.widgets.Button(name="Analyze Player", button_type="primary")

    def on_analyze(event):
        player_name = player_input.value.strip()
        if not player_name:
            main_display.clear()
            main_display.append(pn.pane.Markdown("**Error: Please enter a player name.**"))
            return

        # Proceed to the next step
        prediction_particular_player_workflow(player_name)

    analyze_button.on_click(on_analyze)
    main_display.append(pn.Column(player_input, analyze_button))

def prediction_particular_player_workflow(player_name):
    """
    Interactive workflow for player prediction.
    """
    main_display.clear()
    main_display.append(pn.pane.Markdown(f"# Prediction Workflow for {player_name}"))

    player_dict = {}
    playing_role_map = {
        "Batter": ["batting"],
        "Top order Batter": ["batting"],
        "Middle order Batter": ["batting"],
        "Opening Batter": ["batting"],
        "Wicketkeeper Batter": ["batting"],
        "Bowler": ["bowling"],
        "Spin Bowler": ["bowling"],
        "Pace Bowler": ["bowling"],
        "Batting Allrounder": ["batting", "bowling"],
        "Bowling Allrounder": ["batting", "bowling"],
        "Allrounder": ["batting", "bowling"],
    }

    # Step 1: Fetch player information
    player_info = get_players_info([player_name.strip()])
    if player_info.empty:
        main_display.append(pn.pane.Markdown(f"**Error: No information found for player: {player_name}.**"))
        return main_display

    # Display player data and confirmation interface
    main_display.append(pn.pane.DataFrame(player_info, width=600))
    main_display.append(pn.pane.Markdown(f"**Confirm the data for Player: {player_name}**"))

    # Confirmation and re-entry buttons
    confirm_button = pn.widgets.Button(name="Confirm Player", button_type="success")
    reenter_button = pn.widgets.Button(name="Re-enter Player Name", button_type="warning")

    def confirm_player(event):
        for _, row in player_info.iterrows():
            full_name = row.get("Full Name", "Unknown")
            playing_role = row.get("Playing Role", None)
            player_type = playing_role_map.get(playing_role, [])
            if player_type:
                player_dict[full_name] = (full_name, player_type)

        if player_dict:
            main_display.clear()
            main_display.append(pn.pane.Markdown(f"## Predictions for {player_name}"))
            perform_prediction(player_dict)
        else:
            main_display.append(pn.pane.Markdown("**Error: Unable to confirm player role.**"))

    def reenter_player(event):
        main_display.clear()
        main_display.append(pn.pane.Markdown("**Re-enter the player name and try again.**"))
        player_name_input = pn.widgets.TextInput(name="Player Name", placeholder="Enter Player Name")
        reenter_submit_button = pn.widgets.Button(name="Submit", button_type="primary")

        def on_reenter_submit(event):
            new_name = player_name_input.value.strip()
            if not new_name:
                main_display.append(pn.pane.Markdown("**Error: Player name cannot be empty.**"))
                return
            main_display.clear()
            workflow = prediction_particular_player_workflow(new_name)
            main_display.append(workflow)

        reenter_submit_button.on_click(on_reenter_submit)
        main_display.extend([player_name_input, reenter_submit_button])

    confirm_button.on_click(confirm_player)
    reenter_button.on_click(reenter_player)

    main_display.extend([confirm_button, reenter_button])

    return main_display


def parse_bowling_performance(performance_string):
    """
    Parse the performance string of format 'wickets-runs in overs'.
    Example: '3-25 in 4' -> (3, 25, 4)
    """
    try:
        # Extract the components
        wickets_runs, overs = performance_string.split(" in ")
        wickets, runs = wickets_runs.split("-")

        return int(wickets), int(runs), int(overs)
    except Exception as e:
        # Handle any parsing errors gracefully
        print(f"Error parsing bowling performance string: {e}")
        return 0, 0, 0  # Default fallback

def perform_prediction(player_dict):
    """
    Perform ARIMA-based prediction analysis for players (batting and bowling).
    """
    not_found_players = []

    for player_name, (df_prefix, roles) in player_dict.items():
        not_found_players.extend(get_and_save_player_stats(df_prefix, roles))

    if not_found_players:
        main_display.append(pn.pane.Markdown("**Players for whom data was not found:**"))
        for player in not_found_players:
            main_display.append(pn.pane.Markdown(f"- {player}"))
        return

    for player_name, (df_prefix, roles) in player_dict.items():
        formatted_name = df_prefix.replace(" ", "_")

        # Batting Predictions Section in perform_prediction
        if "batting" in roles:
            try:
                batter_df_name = f"{formatted_name}_batting"
                batter_df = globals().get(batter_df_name)

                if batter_df is None or batter_df.empty:
                    main_display.append(pn.pane.Markdown(f"**No batting data available for {player_name}.**"))
                    continue

                batter_df_clean = clean_batter_data(batter_df)
                batter_df_clean = filter_recent_data(batter_df_clean)

                if batter_df_clean.empty:
                    main_display.append(
                        pn.pane.Markdown(f"**Insufficient data for batting analysis of {player_name}.**"))
                    continue

                batter_df_clean['Impact'] = batter_df_clean.apply(calculate_batter_impact, axis=1)
                batter_df_clean = add_ema_impact(batter_df_clean, span=10)

                train, test = split_time_series(batter_df_clean)
                model = train_arima(train)
                predictions = predict_arima(model, steps=len(test['EMA_Impact']))

                most_common_position=get_most_common_position(batter_df_clean)
                # Fetch batting performance details
                batting_details = []
                for idx, predicted_value in enumerate(predictions[:3]):
                    runs, sr = batting_performance(predicted_value, most_common_position)
                    balls_faced = int(runs / (sr / 100)) if sr > 0 else 0  # Ensure no divide-by-zero errors

                    batting_details.append({
                        "Match Number": f"Match {idx + 1}",
                        "Predicted Impact": f"{predicted_value:.2f}",
                        "Runs": runs,
                        "Strike Rate": sr,
                        "Balls Faced": balls_faced,
                        "Comment": (
                            "Exceptional Performance Expected" if predicted_value > 70 else
                            "Good Performance Expected" if predicted_value > 50 else
                            "Average Performance Expected" if predicted_value > 30 else
                            "Poor Performance Expected"
                        )
                    })

                # Create Tabulator for batting predictions
                batting_prediction_df = pd.DataFrame(batting_details)
                batting_tabulator = pn.widgets.Tabulator(
                    batting_prediction_df, show_index=False, theme="modern", pagination="local", page_size=5
                )

                main_display.append(pn.pane.Markdown(f"### Predictions for {player_name}"))
                main_display.append(pn.pane.Markdown("#### Batting Predictions"))
                main_display.append(batting_tabulator)

            except Exception as e:
                main_display.append(pn.pane.Markdown(f"**Error in batting prediction for {player_name}: {e}**"))

        # Bowling Prediction
        if "bowling" in roles:
            try:
                bowler_df_name = f"{formatted_name}_bowling"
                bowler_df = globals().get(bowler_df_name)

                if bowler_df is None or bowler_df.empty:
                    main_display.append(pn.pane.Markdown(f"**No bowling data available for {player_name}.**"))
                    continue

                bowler_df_clean = clean_bowler_data(bowler_df)
                bowler_df_clean = filter_recent_data(bowler_df_clean)

                if bowler_df_clean.empty:
                    main_display.append(pn.pane.Markdown(f"**Insufficient data for bowling analysis of {player_name}.**"))
                    continue

                bowler_df_clean['Impact'] = bowler_df_clean.apply(calculate_bowler_impact, axis=1)
                bowler_df_clean = add_ema_impact(bowler_df_clean, span=10)

                train, test = split_time_series(bowler_df_clean)
                model = train_arima(train)
                predictions = predict_arima(model, steps=len(test['EMA_Impact']))

                most_common_position = get_most_common_position(bowler_df_clean)

                # Fetch bowling performance details
                bowling_details = []
                for idx, predicted_value in enumerate(predictions[:3]):
                    performance_string = bowling_performance(predicted_value)
                    wickets, runs_conceded, overs = parse_bowling_performance(performance_string)

                    bowling_details.append({
                        "Match Number": f"Match {idx + 1}",
                        "Predicted Impact": f"{predicted_value:.2f}",
                        "Wickets": wickets,
                        "Runs Conceded": runs_conceded,
                        "Overs": overs,
                        "Comment": (
                            "Exceptional Performance Expected" if predicted_value > 70 else
                            "Good Performance Expected" if predicted_value > 50 else
                            "Average Performance Expected" if predicted_value > 30 else
                            "Poor Performance Expected"
                        )
                    })

                # Create Tabulator for bowling predictions
                bowling_prediction_df = pd.DataFrame(bowling_details)
                bowling_tabulator = pn.widgets.Tabulator(
                    bowling_prediction_df, show_index=False, theme="modern", pagination="local", page_size=5
                )

                main_display.append(pn.pane.Markdown("#### Bowling Predictions"))
                main_display.append(bowling_tabulator)

            except Exception as e:
                main_display.append(pn.pane.Markdown(f"**Error in bowling prediction for {player_name}: {e}**"))

        # Add metrics explanation
        main_display.append(pn.pane.Markdown("""
        **Metrics Used for Batting Analysis:**
        - Predicted Impact: A calculated metric based on past performances.
        - Runs: The expected runs to be scored in the match.
        - Strike Rate: Expected strike rate during the innings.
        - Balls Faced: Calculated as Runs / (Strike Rate / 100).
        - Comments:
          - **Exceptional Performance Expected**: Impact > 70.
          - **Good Performance Expected**: 50 < Impact <= 70.
          - **Average Performance Expected**: 30 < Impact < 50.
          - **Average Performance Expected**: Impact <= 30.
        """))



#Callback function for "Prediction for a particular player"
def show_prediction():
    # Clear main display content
    main_display.clear()
    # Call the prediction function and display output
    prediction_output = prediction_particular_player()
    main_display.append(pn.pane.Markdown(prediction_output))


def show_predictions_all():
    main_display.clear()
    # Get the current date in the required format (e.g., "1+Dec+2024")
    current_date = datetime.now().strftime("%d+%b+%Y")  # Example: "13+Dec+2024"

    # Update the URLs dynamically with the current date
    batting_url = f'https://stats.espncricinfo.com/ci/engine/stats/index.html?batting_positionmax2=7;batting_positionmin2=1;batting_positionval2=batting_position;class=3;filter=advanced;orderby=runs;qualmin1=5;qualval1=innings;size=200;spanmax1={current_date};spanmin1=1+Dec+2019;spanval1=span;team=1;team=2;team=25;team=3;team=4;team=40;team=5;team=6;team=7;team=8;template=results;type=batting'
    bowling_url = f'https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;filter=advanced;orderby=wickets;qualmin2=5;qualval2=innings_bowled;size=200;spanmax1={current_date};spanmin1=1+Dec+2019;spanval1=span;team=1;team=2;team=25;team=3;team=4;team=40;team=5;team=6;team=7;team=8;template=results;type=bowling'
    print(batting_url)
    print(bowling_url)
    # Fetch player names
    batting_players = get_player_names(batting_url)
    bowling_players = get_player_names(bowling_url)
    all_players_list = list(set(batting_players + bowling_players))
    print(f"Fetched player names: {all_players_list}")

    # Get player info
    players_info_df = get_players_info(all_players_list)
    print(f"Player information DataFrame:\n{players_info_df.head()}")

    # Define playing role map
    playing_role_map = {
        "Batter": ["batting"], "Top order Batter": ["batting"], "Middle order Batter": ["batting"],
        "Opening Batter": ["batting"], "Wicketkeeper Batter": ["batting"], "Bowler": ["bowling"],
        "Spin Bowler": ["bowling"], "Pace Bowler": ["bowling"], "Batting Allrounder": ["batting", "bowling"],
        "Bowling Allrounder": ["batting", "bowling"], "Allrounder": ["batting", "bowling"]
    }

    # Create player-role dictionary
    all_players = {}
    for index, row in players_info_df.iterrows():
        player_name = row["Full Name"]
        playing_role = row["Playing Role"]
        player_type = playing_role_map.get(playing_role, None)
        if player_type:
            all_players[player_name] = (player_name, player_type)

    print(f"Player-role mapping:\n{all_players}")

    not_found_players = []
    for player_name, (player_name, types) in all_players.items():
        not_found_players.extend(get_and_save_player_stats(player_name, types))
    all_iplayers = {name: details for name, details in all_players.items() if name not in P_Test_list}
    if not_found_players:
        print(f"Data not found for these players: {not_found_players}")
    # Initialize separate DataFrames for batters and bowlers
    batters_df = pd.DataFrame(columns=[
        'Player Name', 'Type', 'Position', 'Predicted Impact',
        'Prediction1 (Runs/SR)', 'Prediction2 (Runs/SR)', 'Prediction3 (Runs/SR)'
    ])
    bowlers_df = pd.DataFrame(columns=[
        'Player Name', 'Type', 'Position', 'Predicted Impact',
        'Prediction1 (Wkts/Econ)', 'Prediction2 (Wkts/Econ)', 'Prediction3 (Wkts/Econ)'
    ])
    all_metrics = []
    for player_name, (df_prefix, roles) in all_iplayers.items():
        print(f"\nProcessing {player_name}...")
        formatted_name = df_prefix.replace(" ", "_")

        # Batting Predictions Section
        if "batting" in roles:
            try:
                batter_df_name = f"{formatted_name}_batting"
                batter_df = globals().get(batter_df_name)

                if batter_df is None or batter_df.empty:
                    print((f"**No batting data available for {player_name}.**"))
                    continue

                batter_df_clean = clean_batter_data(batter_df)
                batter_df_clean = filter_recent_data(batter_df_clean)

                if batter_df_clean.empty:
                    print((f"**Insufficient data for batting analysis of {player_name}.**"))
                    continue

                batter_df_clean['Impact'] = batter_df_clean.apply(calculate_batter_impact, axis=1)
                batter_df_clean = add_ema_impact(batter_df_clean, span=10)

                train, test = split_time_series(batter_df_clean)
                model = train_arima(train)
                predictions = predict_arima(model, steps=len(test['EMA_Impact']))

                most_common_position = get_most_common_position(batter_df_clean)

                prediction_strings = []
                for idx, predicted_value in enumerate(predictions[:3]):
                    runs, sr = batting_performance(predicted_value, most_common_position)
                    balls_faced = int(runs / (sr / 100)) if sr > 0 else 0
                    prediction_strings.append(f"{runs} runs @ {sr:.2f} SR")
                #                 # Evaluate model and store metrics
                metrics = evaluate_model(test, predictions, all_metrics)
                batting_row = {
                    "Player Name": player_name,
                    "Type": "Batter",
                    "Position": most_common_position,
                    "Predicted Impact": np.abs(predictions).mean(),
                    "Prediction1 (Runs/SR)": prediction_strings[0] if len(prediction_strings) > 0 else None,
                    "Prediction2 (Runs/SR)": prediction_strings[1] if len(prediction_strings) > 1 else None,
                    "Prediction3 (Runs/SR)": prediction_strings[2] if len(prediction_strings) > 2 else None
                }
                batters_df = pd.concat([batters_df, pd.DataFrame([batting_row])], ignore_index=True)

            except Exception as e:
                print((f"**Error in batting prediction for {player_name}: {e}**"))

        # Bowling Predictions Section
        if "bowling" in roles:
            try:
                bowler_df_name = f"{formatted_name}_bowling"
                bowler_df = globals().get(bowler_df_name)

                if bowler_df is None or bowler_df.empty:
                    print((f"**No bowling data available for {player_name}.**"))
                    continue

                bowler_df_clean = clean_bowler_data(bowler_df)
                bowler_df_clean = filter_recent_data(bowler_df_clean)

                if bowler_df_clean.empty:
                    print((f"**Insufficient data for bowling analysis of {player_name}.**"))
                    continue

                bowler_df_clean['Impact'] = bowler_df_clean.apply(calculate_bowler_impact, axis=1)
                bowler_df_clean = add_ema_impact(bowler_df_clean, span=10)

                train, test = split_time_series(bowler_df_clean)
                model = train_arima(train)
                predictions = predict_arima(model, steps=len(test['EMA_Impact']))

                most_common_position = get_most_common_position(bowler_df_clean)

                prediction_strings = []
                for idx, predicted_value in enumerate(predictions[:3]):
                    performance_string = bowling_performance(predicted_value)
                    wickets, runs_conceded, overs = parse_bowling_performance(performance_string)
                    prediction_strings.append(f"{wickets} wkts, {runs_conceded} runs in {overs} overs")
                # Evaluate model and store metrics
                metrics = evaluate_model(test, predictions, all_metrics)
                bowling_row = {
                    "Player Name": player_name,
                    "Type": "Bowler",
                    "Position": most_common_position,
                    "Predicted Impact": np.abs(predictions).mean(),
                    "Prediction1 (Wkts/Econ)": prediction_strings[0] if len(prediction_strings) > 0 else None,
                    "Prediction2 (Wkts/Econ)": prediction_strings[1] if len(prediction_strings) > 1 else None,
                    "Prediction3 (Wkts/Econ)": prediction_strings[2] if len(prediction_strings) > 2 else None
                }
                bowlers_df = pd.concat([bowlers_df, pd.DataFrame([bowling_row])], ignore_index=True)

            except Exception as e:
                print((f"**Error in bowling prediction for {player_name}: {e}**"))
        # Calculate and print the average metrics across all players
        avg_metrics = calculate_average_metrics(all_metrics)
        print("\nAverage Metrics across all players:")
        print(f"Average MAE: {avg_metrics['MAE']:.4f}")
        print(f"Average RMSE: {avg_metrics['RMSE']:.4f}")
        print(f"Average MAPE: {avg_metrics['MAPE']:.4f}")

    # Append to dashboard
    if not batters_df.empty:
        batters_tabulator = pn.widgets.Tabulator(
            batters_df, show_index=False, theme="modern", pagination="local", page_size=5
        )
        main_display.append(pn.pane.Markdown("# Batting Predictions"))
        main_display.append(batters_tabulator)

    if not bowlers_df.empty:
        bowlers_tabulator = pn.widgets.Tabulator(
            bowlers_df, show_index=False, theme="modern", pagination="local", page_size=5
        )
        main_display.append(pn.pane.Markdown("# Bowling Predictions"))
        main_display.append(bowlers_tabulator)

    if batters_df.empty and bowlers_df.empty:
        return pn.pane.Markdown("**No predictions available. Please check the input data.**")





# Function to analyze a specific player with input field
def show_analyze_player():
    main_display.clear()
    player_input = pn.widgets.TextInput(name="Enter Player Name")
    search_button = pn.widgets.Button(name="Search Player", button_type="primary")

    # Persistent Perform Analysis button (disabled by default)
    finalize_button = pn.widgets.Button(name="Perform Analysis", button_type="success", disabled=True)

    def on_search(event):
        player_name = player_input.value.strip()
        if not player_name:
            main_display.clear()
            main_display.append(pn.pane.Markdown("**Error: Please enter a player name.**"))
            main_display.append(finalize_button)  # Ensure button persists
            return

        # Proceed to player workflow
        particular_player_workflow(player_name, finalize_button)

    search_button.on_click(on_search)
    main_display.append(pn.Column(player_input, search_button, finalize_button))



# Function to compare multiple players
def show_compare_players():
    """
    Unified function to:
    1. Collect player names and comparison type.
    2. Search and confirm players interactively.
    3. Perform analysis and display results.
    """
    # Clear the main display
    main_display.clear()

    # Step 1: Widgets for user inputs
    num_players_input = pn.widgets.IntSlider(name="Number of Players", start=2, end=5, step=1, value=2)
    player_name_inputs = [pn.widgets.TextInput(name=f"Player {i + 1} Name") for i in range(5)]  # Max 5 inputs
    compare_choice_input = pn.widgets.RadioButtonGroup(
        name="Comparison Type",
        options={"Batting": 1, "Bowling": 2, "Both": 3},
        value=1
    )
    submit_button = pn.widgets.Button(name="Submit", button_type="primary")

    # Step 2: Dynamically update player input visibility
    @pn.depends(num_players_input.param.value, watch=True)
    def update_player_inputs(num_players):
        for i, widget in enumerate(player_name_inputs):
            widget.visible = i < num_players

    update_player_inputs(num_players_input.value)  # Initial setup

    # Step 3: Handle submission
    def on_submit(event):
        num_players = num_players_input.value
        player_names = [w.value.strip() for w in player_name_inputs[:num_players] if w.visible and w.value.strip()]

        if len(player_names) != num_players:
            main_display.clear()
            main_display.append(pn.pane.Markdown("**Error: Please provide valid names for all players.**"))
            return

        # Debugging - Log collected inputs
        print(f"Collected Player Names: {player_names}")
        print(f"Comparison Choice: {compare_choice_input.value}")

        # Gather and validate player data
        player_dict = {}  # To store confirmed player data
        playing_role_map = {
            "Batter": ["batting"],
            "Top order Batter": ["batting"],
            "Middle order Batter": ["batting"],
            "Opening Batter": ["batting"],
            "Wicketkeeper Batter": ["batting"],
            "Bowler": ["bowling"],
            "Spin Bowler": ["bowling"],
            "Pace Bowler": ["bowling"],
            "Batting Allrounder": ["batting", "bowling"],
            "Bowling Allrounder": ["batting", "bowling"],
            "Allrounder": ["batting", "bowling"]
        }

        current_index = [0]  # Track the current player being processed

        def process_next_player():
            if current_index[0] >= len(player_names):
                # All players processed, proceed with comparison
                if not player_dict:
                    main_display.clear()
                    main_display.append(pn.pane.Markdown("**Error: No valid players confirmed.**"))
                    return

                main_display.clear()
                comparison_choice = compare_choice_input.value

                # Perform analysis and visualization
                comparison_data = {"batting": [], "bowling": []}

                # Fetch player stats and analyze
                not_found_players = []
                for player_name, (df_prefix, roles) in player_dict.items():
                    not_found_players.extend(get_and_save_player_stats(df_prefix, roles))

                if not_found_players:
                    main_display.append(pn.pane.Markdown("**Players for whom data was not found:**"))
                    for player in not_found_players:
                        main_display.append(pn.pane.Markdown(f"- {player}"))
                    return  # Skip analysis if data is missing

                # Proceed with cleaning and analysis for confirmed players
                for player_name, (df_prefix, roles) in player_dict.items():
                    formatted_name = df_prefix.replace(" ", "_")

                    # Analyze Batting (if needed)
                    if "batting" in roles and comparison_choice in [1, 3]:
                        try:
                            batter_df_name = f"{formatted_name}_batting"
                            batter_df = globals().get(batter_df_name)
                            if batter_df is None or batter_df.empty:
                                main_display.append(
                                    pn.pane.Markdown(f"**No batting data available for {player_name}.**"))
                                continue

                            batter_df_clean = clean_batter_data(batter_df)
                            batter_df_clean["Impact"] = batter_df_clean.apply(calculate_batter_impact, axis=1)

                            # Append the player name and cleaned data to comparison_data
                            comparison_data["batting"].append({"player_name": player_name, "data": batter_df_clean})
                        except Exception as e:
                            main_display.append(
                                pn.pane.Markdown(f"**Error during batting analysis for {player_name}: {e}**"))

                    # Analyze Bowling (if needed)
                    if "bowling" in roles and comparison_choice in [2, 3]:
                        try:
                            bowler_df_name = f"{formatted_name}_bowling"
                            bowler_df = globals().get(bowler_df_name)
                            if bowler_df is None or bowler_df.empty:
                                main_display.append(
                                    pn.pane.Markdown(f"**No bowling data available for {player_name}.**"))
                                continue

                            bowler_df_clean = clean_bowler_data(bowler_df)
                            bowler_df_clean["Impact"] = bowler_df_clean.apply(calculate_bowler_impact, axis=1)

                            # Append the player name and cleaned data to comparison_data
                            comparison_data["bowling"].append({"player_name": player_name, "data": bowler_df_clean})
                        except Exception as e:
                            main_display.append(
                                pn.pane.Markdown(f"**Error during bowling analysis for {player_name}: {e}**"))

                # Call the plotting function with structured data
                # Clear the display and add the comparison results
                main_display.clear()
                main_display.append("## Comparison Results")
                plots = perform_comparisons_with_plots(comparison_data["batting"], comparison_data["bowling"],comparison_choice)
                for plot_component in plots:
                    main_display.append(plot_component)  # Append each component (Markdown or PNG) separately


            player_name = player_names[current_index[0]]
            player_info = get_players_info([player_name.strip()])

            if player_info.empty:
                main_display.clear()
                main_display.append(pn.pane.Markdown(f"**Error: No information found for player: {player_name}.**"))
                return

            main_display.clear()
            main_display.append(pn.pane.DataFrame(player_info, width=600))
            main_display.append(
                pn.pane.Markdown(f"**Confirm the data for Player {current_index[0] + 1}: {player_name}**"))

            confirm_button = pn.widgets.Button(name="Confirm", button_type="success")
            reenter_button = pn.widgets.Button(name="Re-enter Name", button_type="warning")

            def confirm_player(event):
                for _, row in player_info.iterrows():
                    full_name = row.get("Full Name", "Unknown")
                    playing_role = row.get("Playing Role", None)
                    player_type = playing_role_map.get(playing_role, [])
                    if player_type:
                        player_dict[full_name] = (full_name, player_type)

                current_index[0] += 1
                process_next_player()

            def reenter_player(event):
                main_display.clear()
                main_display.append(pn.pane.Markdown(f"**Re-enter the name for Player {current_index[0] + 1}:**"))
                player_name_input = pn.widgets.TextInput(name="Player Name", value=player_name)

                def resubmit_name(event):
                    player_names[current_index[0]] = player_name_input.value.strip()
                    process_next_player()

                resubmit_button = pn.widgets.Button(name="Submit", button_type="primary")
                resubmit_button.on_click(resubmit_name)

                main_display.append(player_name_input)
                main_display.append(resubmit_button)

            confirm_button.on_click(confirm_player)
            reenter_button.on_click(reenter_player)

            main_display.append(confirm_button)
            main_display.append(reenter_button)

        process_next_player()

    submit_button.on_click(on_submit)

    # Add input components to the main display
    main_display.extend([
        pn.Column(
            "### Compare and Confirm Players",
            num_players_input,
            *player_name_inputs,
            compare_choice_input,
            submit_button
        )
    ])

def perform_comparison_analysis(player_dict, comparison_choice):
    """
    Perform the comparison analysis based on confirmed player data and choice.
    Returns a structured result for batting and bowling analysis.
    """
    comparison_data = {
        "batting": [],
        "bowling": [],
        "not_found": [],  # Players with missing data
    }

    for player_name, (df_prefix, roles) in player_dict.items():
        formatted_name = df_prefix.replace(" ", "_")

        # Analyze Batting
        if "batting" in roles and comparison_choice in [1, 3]:
            try:
                batter_df_name = f"{formatted_name}_batting"
                batter_df = globals().get(batter_df_name)
                if batter_df is None or batter_df.empty:
                    comparison_data["not_found"].append(f"Batting data not found for {player_name}.")
                    continue

                batter_df_clean = clean_batter_data(batter_df)
                batter_df_clean["Impact"] = batter_df_clean.apply(calculate_batter_impact, axis=1)
                comparison_data["batting"].append({
                    "player_name": player_name,
                    "data": batter_df_clean
                })

            except Exception as e:
                comparison_data["not_found"].append(f"Error analyzing batting data for {player_name}: {e}")

        # Analyze Bowling
        if "bowling" in roles and comparison_choice in [2, 3]:
            try:
                bowler_df_name = f"{formatted_name}_bowling"
                bowler_df = globals().get(bowler_df_name)
                if bowler_df is None or bowler_df.empty:
                    comparison_data["not_found"].append(f"Bowling data not found for {player_name}.")
                    continue

                bowler_df_clean = clean_bowler_data(bowler_df)
                bowler_df_clean["Impact"] = bowler_df_clean.apply(calculate_bowler_impact, axis=1)
                comparison_data["bowling"].append({
                    "player_name": player_name,
                    "data": bowler_df_clean
                })

            except Exception as e:
                comparison_data["not_found"].append(f"Error analyzing bowling data for {player_name}: {e}")
    print(comparison_data)
    return comparison_data


# Helper function: Perform comparisons and return plots
def perform_comparisons_with_plots(batting_players, bowling_players, compare_choice):
    """
    Generate comparison plots for batting and bowling based on the selected comparison choice.

    Parameters:
        batting_players (list): List of dictionaries with keys "player_name" and "data" for batting data.
        bowling_players (list): List of dictionaries with keys "player_name" and "data" for bowling data.
        compare_choice (int): Comparison choice (1: Batting, 2: Bowling, 3: Both).

    Returns:
        list: List of Panel components containing plots.
    """
    plots = []

    # Convert lists of dictionaries to the required format for comparison functions
    batting_players_data = {player["player_name"]: player["data"] for player in batting_players}
    bowling_players_data = {player["player_name"]: player["data"] for player in bowling_players}

    if compare_choice in [1, 3]:  # Batting comparison
        print("Generating batting comparison...")
        batting_comparison_components = generate_batting_comparison(batting_players_data)
        plots.append(batting_comparison_components)

    if compare_choice in [2, 3]:  # Bowling comparison
        print("Generating bowling comparison...")
        bowling_comparison_components = generate_bowling_comparison(bowling_players_data)
        plots.append(bowling_comparison_components)

    print("Generated plots:", plots)
    return plots

def generate_batting_comparison(players_data):
    """
    Generate batting comparison for multiple players.
    Args:
        players_data: dict - {player_name: batter_df_clean}
    Returns:
        A Panel layout with horizontally aligned comparisons.
    """
    # Check if there are enough players for batting analysis
    if len(players_data) < 2:
        return pn.pane.Markdown("""
            <div style="text-align: center; color: #d9534f; font-size: 18px; font-weight: bold;">
                ❌ Not enough players for Batting Analysis (at least 2 players required).
            </div>
        """)

    comparison_panels = []

    # Heading for Batting Analysis
    heading = pn.pane.Markdown("<h1 style='text-align: center;'>Batting Performance Analysis</h1>", width=1200)

    # Player Info and Batting Stats
    for player_name, batter_df_clean in players_data.items():
        batting_stats = calculate_batting_stats(batter_df_clean)

        info_display = pn.pane.Markdown(f"""
            ### {player_name}
            **Most Common Position:** {batting_stats['Most Common Position']}  
            **Average Runs:** {batting_stats['Average Runs']}  
            **Average Strike Rate:** {batting_stats['Average Strike Rate']}  
        """)

        # Recent Form
        _, recent_form_plot = calculate_recent_form(batter_df_clean)

        # Combine into a single player's panel
        player_panel = pn.Column(info_display, recent_form_plot)
        comparison_panels.append(player_panel)

    # Shared Performance Comparisons
    senai_analysis = analyze_vs_senai(players_data)
    venue_analysis = analyze_vs_venue(players_data)

    # Combine all components with the heading
    return pn.Column(heading, pn.Row(*comparison_panels), pn.Row(senai_analysis, venue_analysis))


def generate_bowling_comparison(players_data):
    """
    Generate bowling comparison for multiple players.
    Args:
        players_data: dict - {player_name: bowler_df_clean}
    Returns:
        A Panel layout with horizontally aligned comparisons.
    """
    # Check if there are enough players for bowling analysis
    if len(players_data) < 2:
        return pn.pane.Markdown("""
            <div style="text-align: center; color: #d9534f; font-size: 18px; font-weight: bold;">
                ❌ Not enough players for Bowling Analysis (at least 2 players required).
            </div>
        """)

    comparison_panels = []

    # Heading for Bowling Analysis
    heading = pn.pane.Markdown("<h1 style='text-align: center;'>Bowling Performance Analysis</h1>", width=1200)

    # Player Info and Bowling Stats
    for player_name, bowler_df_clean in players_data.items():
        bowling_stats = calculate_bowling_stats(bowler_df_clean)

        info_display = pn.pane.Markdown(f"""
            ### {player_name}
            **Total Wickets:** {bowling_stats['Total Wickets']}  
            **Total Matches:** {bowling_stats['Total Matches']}  
            **Economy Rate:** {bowling_stats['Economy Rate']}  
            **Bowling Average:** {bowling_stats['Bowling Average']}  
        """)

        # Recent Form
        _, recent_form_plot = calculate_recent_form(bowler_df_clean)

        # Combine into a single player's panel
        player_panel = pn.Column(info_display, recent_form_plot)
        comparison_panels.append(player_panel)

    # Shared Performance Comparisons
    senai_analysis = analyze_vs_senai(players_data)
    venue_analysis = analyze_vs_venue(players_data)

    # Combine all components with the heading
    return pn.Column(heading, pn.Row(*comparison_panels), pn.Row(senai_analysis, venue_analysis))



def analyze_vs_venue(players_data):
    """
    Compare performance by venue for multiple players.
    Args:
        players_data: dict - {player_name: player_df_clean}
    Returns:
        A Panel Plotly pane with the venue comparison.
    """
    # Collect all venue data
    venues_by_player = {}
    for player_name, player_df_clean in players_data.items():
        venues_by_player[player_name] = player_df_clean.groupby('Ground').agg(
            Impact=('Impact', 'mean'),
            Matches=('Ground', 'count')  # Count number of matches per venue
        )

    # Identify common venues where all players have played at least once
    common_venues = set.intersection(
        *[set(venues.index) for venues in venues_by_player.values()]
    )

    if not common_venues:
        return pn.pane.Markdown("""
            <div style="text-align: center; color: #d9534f; font-size: 16px; font-weight: bold;">
                ❌ No common venues where all players have played.
            </div>
        """)

    # Limit to a maximum of 15 venues based on the number of matches
    venue_match_counts = pd.DataFrame({
        player_name: venues.loc[list(common_venues), 'Matches']
        for player_name, venues in venues_by_player.items()
    })

    # Sum match counts across players to determine top 15 venues
    venue_match_counts['Total_Matches'] = venue_match_counts.sum(axis=1)
    top_venues = venue_match_counts.nlargest(15, 'Total_Matches').index

    # Filter data to only include the top 15 venues
    filtered_data = {
        player_name: venues.loc[top_venues, 'Impact']
        for player_name, venues in venues_by_player.items()
    }

    # Combine data into a single DataFrame
    df = pd.DataFrame(filtered_data)

    # Create a Plotly bar chart for comparison
    fig = go.Figure()

    for player_name in players_data.keys():
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df[player_name],
                name=player_name,
                text=df[player_name].round(2),
                textposition='outside'
            )
        )

    # Update layout
    fig.update_layout(
        title="Venue Performance Comparison",
        xaxis_title="Venue",
        yaxis_title="Average Impact",
        yaxis=dict(range=[0, 100]),  # Restrict Y-axis to 0-100 range
        barmode='group',
        template='plotly_white',
        legend_title="Players",
        width=580,  # Set width to 580px
        height=360,  # Adjust height for a balanced aspect ratio
        xaxis=dict(tickangle=45),
    )

    # Return Panel component
    return pn.pane.Plotly(fig)

def analyze_vs_senai(player_data_dict):
    """
    Analyze performance vs SENAI countries and create an interactive Plotly bar chart.

    Parameters:
        player_data_dict (dict): Dictionary where keys are player names and values are DataFrames of player data.

    Returns:
        Panel object containing the interactive plot.
    """
    # Define SENAI countries
    senai_countries = ['v South Africa', 'v England', 'v New Zealand', 'v Australia', 'v India']

    # Prepare data for plotting
    fig = go.Figure()
    for player_name, player_df in player_data_dict.items():
        # Filter data for SENAI countries
        senai_df = player_df[player_df['Opposition'].isin(senai_countries)]

        # Handle case with no data
        if senai_df.empty:
            return pn.pane.Markdown(f"""
                <div style="text-align: center; color: #d9534f; font-size: 16px; font-weight: bold;">
                    ❌ No data available for SENAI countries for player: {player_name}.
                </div>
            """)

        # Calculate average impact per SENAI country
        impact_by_country = senai_df.groupby('Opposition')['Impact'].mean().sort_values()

        # Add trace for each player
        fig.add_trace(go.Bar(
            x=impact_by_country.index,
            y=impact_by_country.values,
            name=player_name,
            text=[f"{impact:.2f}" for impact in impact_by_country.values],  # Add text labels
            textposition='auto',
        ))

    # Customize the layout
    fig.update_layout(
        title=dict(
            text="Performance vs SENAI Countries",
            font=dict(size=20, color="#4CAF50"),
            x=0.5,
        ),
        xaxis=dict(
            title="Country",
            titlefont=dict(size=14, color="#37474F"),
            tickfont=dict(size=12, color="#37474F"),
            tickangle=45,  # Move this from the second xaxis definition
        ),
        yaxis=dict(
            title="Average Impact",
            titlefont=dict(size=14, color="#37474F"),
            tickfont=dict(size=12, color="#37474F"),
            range=[0, 100],  # Fix y-axis range
        ),
        legend=dict(
            title="Player",
            font=dict(size=8),
            orientation="h",
            x=1.2,  # Far right
            y=1.2,  # Top
            xanchor="right",
            yanchor="top"
        ),
        barmode="group",  # Group bars for comparison
        template="plotly_white",
        width=580,  # Set width to 580px
        height=360,  # Adjust height for a balanced aspect ratio
    )

    # Render in Panel
    return pn.pane.Plotly(fig)



def calculate_batting_stats(batter_df_clean):
    """Calculate batting stats: most common position, avg runs, strike rate."""
    most_common_position = get_most_common_position(batter_df_clean)
    avg_runs = batter_df_clean['Runs'].mean()
    avg_strike_rate = batter_df_clean['SR'].mean()
    return {
        'Most Common Position': most_common_position,
        'Average Runs': round(avg_runs, 2),
        'Average Strike Rate': round(avg_strike_rate, 2)
    }

def calculate_bowling_stats(bowler_df_clean):
    """Calculate bowling stats: total wickets, matches, economy rate, bowling average."""
    print(bowler_df_clean)
    total_wickets = bowler_df_clean['Wkts'].sum()
    total_matches = bowler_df_clean['Start Date'].nunique()
    economy_rate = bowler_df_clean['Econ'].mean()
    # Ensure 'Runs' is numeric
    bowler_df_clean['Runs'] = pd.to_numeric(bowler_df_clean['Runs'], errors='coerce')
    # Calculate total runs and total wickets
    total_runs = bowler_df_clean['Runs'].sum()
    # Calculate bowling average
    bowling_average = total_runs / total_wickets if total_wickets > 0 else float('nan')
    return {
        'Total Wickets': total_wickets,
        'Total Matches': total_matches,
        'Economy Rate': round(economy_rate, 2),
        'Bowling Average': round(bowling_average, 2)
    }

# Helper function: Gather player data from names
def gather_player_data_from_names(player_names):
    player_dict = {}
    print(player_names)
    for player_name in player_names:
        layout, player_data = multi_player_confirmation(player_name)  # Unpack the tuple
        print(player_data)
        if player_data:
            player_dict.update(player_data)  # Use only the dictionary part
            print(player_dict)
        else:
            main_display.append(pn.pane.Markdown(f"**Data not available for {player_name}. Please check the name.**"))
            return None
    return player_dict






# Helper function: Plot bar chart and return Panel-compatible plot
def plot_bar_chart(data, title, xlabel, ylabel, palette):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(data.keys()), y=list(data.values()), ax=ax, palette=palette)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return pn.pane.Matplotlib(fig)

    # Directory for saving dataframes
PLAYER_DATA_DIR = "./player_data"
os.makedirs(PLAYER_DATA_DIR, exist_ok=True)
def save_player_dataframes():
    """
    Fetch and save all player data dynamically to local files, including player_info_df.
    """# Get the current date in the required format (e.g., "1+Dec+2024")
    current_date = datetime.now().strftime("%d+%b+%Y")  # Example: "13+Dec+2024"

    # Update the URLs dynamically with the current date
    batting_url = f'https://stats.espncricinfo.com/ci/engine/stats/index.html?batting_positionmax2=7;batting_positionmin2=1;batting_positionval2=batting_position;class=3;filter=advanced;orderby=runs;qualmin1=5;qualval1=innings;size=200;spanmax1={current_date};spanmin1=1+Dec+2019;spanval1=span;team=1;team=2;team=25;team=3;team=4;team=40;team=5;team=6;team=7;team=8;template=results;type=batting'

    bowling_url = f'https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;filter=advanced;orderby=wickets;qualmin2=5;qualval2=innings_bowled;size=200;spanmax1={current_date};spanmin1=1+Dec+2019;spanval1=span;team=1;team=2;team=25;team=3;team=4;team=40;team=5;team=6;team=7;team=8;template=results;type=bowling'

    # Fetch player names
    batting_players = get_player_names(batting_url)
    bowling_players = get_player_names(bowling_url)

    # Combine both lists and remove duplicates
    all_players_list = list(set(batting_players + bowling_players))
    print("Fetched Players:", all_players_list)

    # Get player information
    players_info_df = get_players_info(all_players_list)

    # Save player_info_df locally
    player_info_file = os.path.join(PLAYER_DATA_DIR, "player_info.pkl")
    players_info_df.to_pickle(player_info_file)
    print(f"Saved player_info_df to {player_info_file}")

    # Mapping playing roles to data types
    playing_role_map = {
        "Batter": ["batting"],
        "Top order Batter": ["batting"],
        "Middle order Batter": ["batting"],
        "Opening Batter": ["batting"],
        "Wicketkeeper Batter": ["batting"],
        "Bowler": ["bowling"],
        "Spin Bowler": ["bowling"],
        "Pace Bowler": ["bowling"],
        "Batting Allrounder": ["batting", "bowling"],
        "Bowling Allrounder": ["batting", "bowling"],
        "Allrounder": ["batting", "bowling"]
    }

    # Create a dictionary of players and their roles
    all_players = {}
    for index, row in players_info_df.iterrows():
        player_name = row["Full Name"]
        playing_role = row["Playing Role"]
        player_type = playing_role_map.get(playing_role, None)
        if player_type:
            all_players[player_name] = (player_name, player_type)

    # Step to fetch player stats and save them
    not_found_players = []
    for player_name, (player_name, roles) in all_players.items():
        not_found_players.extend(get_and_save_player_stats(player_name, roles))  # Fetch and save player stats

    if not_found_players:
        print("\nPlayers for whom data was not found:")
        for player in not_found_players:
            print(player)

    # Iterate through all players and their roles to save data locally
    for player_name, (df_prefix, roles) in all_players.items():
        print(f"\nProcessing {player_name}...")

        # Replace spaces in player names with underscores for dataframe reference
        formatted_name = df_prefix.replace(" ", "_")

        # Save batting data if available
        if 'batting' in roles:
            batter_df_name = f"{formatted_name}_batting"
            try:
                # Dynamically get the dataframe
                batter_df = globals()[batter_df_name]

                # Save the dataframe
                file_name = f"{formatted_name}_local_batting.pkl"
                save_path = os.path.join(PLAYER_DATA_DIR, file_name)
                batter_df.to_pickle(save_path)  # Save as a pickle file
                print(f"Saved {batter_df_name} to {save_path}")

            except KeyError:
                print(f"Data for {batter_df_name} not found.")

        # Save bowling data if available
        if 'bowling' in roles:
            bowler_df_name = f"{formatted_name}_bowling"
            try:
                # Dynamically get the dataframe
                bowler_df = globals()[bowler_df_name]

                # Save the dataframe
                file_name = f"{formatted_name}_local_bowling.pkl"
                save_path = os.path.join(PLAYER_DATA_DIR, file_name)
                bowler_df.to_pickle(save_path)  # Save as a pickle file
                print(f"Saved {bowler_df_name} to {save_path}")

            except KeyError:
                print(f"Data for {bowler_df_name} not found.")

def fetch_local_player_data(directory=PLAYER_DATA_DIR):
    """
    Load all saved player dataframes dynamically, including player_info_df.

    Args:
        directory (str): The directory where player data is saved.

    Returns:
        dict: A dictionary of dataframes keyed by player-role names.
        pd.DataFrame: The player_info_df dataframe.
    """
    player_data = {}
    player_info_df = None

    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            file_path = os.path.join(directory, file)
            try:
                # Check for player_info_df
                if file == "player_info.pkl":
                    player_info_df = pd.read_pickle(file_path)
                    print(f"Loaded player_info_df from {file_path}")
                    continue

                # Extract player name and role from the filename
                base_name = os.path.splitext(file)[0]
                player_name, role = base_name.rsplit("_local_", 1)

                # Load the dataframe
                df = pd.read_pickle(file_path)

                # Store in the dictionary
                player_data[f"{player_name}_{role}"] = df
                print(f"Loaded data for {player_name} ({role}) from {file_path}")
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    print(type(player_data))
    return player_data, player_info_df


#
# # Example Usage
# if __name__ == "__main__":
#     # Step 1: Save player data to local files
#     save_player_dataframes()
#
#     # Step 2: Fetch saved player data for analysis
#     fetch_local_player_data()


def show_analysis_local():
    """
    Show predictions and analyses for all players using local saved datasets.
    """
    main_display.clear()

    # Step 1: Fetch local player data
    main_display.append(pn.pane.Markdown("# All Players Analysis (Local Data)"))
    main_display.append(pn.pane.Markdown("## Fetching local datasets..."))

    try:
        player_data,_ = fetch_local_player_data()
        if not player_data:
            main_display.append(pn.pane.Markdown("**Error: No local data found. Please check the directory.**"))
            return
    except Exception as e:
        main_display.append(pn.pane.Markdown(f"**Error fetching local data: {e}**"))
        return

    main_display.append(pn.pane.Markdown(f"Loaded data for {len(player_data)} players."))

    # Step 2: Analyze top players by position
    main_display.append(pn.pane.Markdown("## Analyzing Top Players by Position..."))

    def run_top_player_analysis():
        try:
            position_categories = {
                'overall': [],
                'opener': [],
                'position_3': [],
                'position_4': [],
                'finisher': []
            }

            for player_key, df in player_data.items():
                if 'batting' in player_key:
                    try:
                        df_clean = clean_batter_data(df)
                        df_clean = filter_recent_data(df_clean)
                        if df_clean.empty:
                            continue

                        df_clean['Impact'] = df_clean.apply(calculate_batter_impact, axis=1)
                        df_clean = df_clean.dropna(subset=['Impact'])

                        # Determine player's most common position
                        most_common_position = get_most_common_position(df_clean)
                        avg_impact = df_clean['Impact'].mean()

                        # Classify by position
                        player_name = player_key.split("_")[0]
                        position_categories['overall'].append((player_name, avg_impact))
                        if most_common_position in [1, 2]:
                            position_categories['opener'].append((player_name, avg_impact))
                        elif most_common_position == 3:
                            position_categories['position_3'].append((player_name, avg_impact))
                        elif most_common_position == 4:
                            position_categories['position_4'].append((player_name, avg_impact))
                        elif most_common_position in [5, 6]:
                            position_categories['finisher'].append((player_name, avg_impact))
                    except Exception as e:
                        main_display.append(pn.pane.Markdown(f"**Error analyzing batter {player_key}: {e}**"))
            # Plot the results
            main_display.append(pn.pane.Markdown("### Visualizing Top Players by Position"))
            for category, players in position_categories.items():
                if players:
                    top_players = sorted(players, key=lambda x: x[1], reverse=True)[:10]
                    player_names, impact_values = zip(*top_players)

                    # Create a plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(player_names, impact_values, color='skyblue')
                    ax.set_xlabel("Impact Rating")
                    ax.set_ylabel("Player Name")
                    ax.set_title(f"Top 10 Batters - {category.capitalize()}")
                    ax.invert_yaxis()
                    main_display.append(pn.pane.Matplotlib(fig, tight=True))
        except Exception as e:
            main_display.append(pn.pane.Markdown(f"**Error in top player analysis: {e}**"))

    run_top_player_analysis()

    # Step 3: Analyze players country-wise
    main_display.append(pn.pane.Markdown("## Analyzing Player Performance by Country..."))

    def run_countrywise_analysis():
        try:
            # Integrate the provided countrywise_analysis functionality
            top_countries = ["India", "England", "Australia", "New Zealand", "West Indies",
                             "Sri Lanka", "Bangladesh", "Pakistan", "South Africa", "Afghanistan"]

            # Initialize empty DataFrames for each country
            batters_dfs = {country: pd.DataFrame(columns=['Player', 'Average_Impact', 'Opponent Country', 'Batting Style'])
                           for country in top_countries}
            bowlers_dfs = {country: pd.DataFrame(columns=['Player', 'Average_Impact', 'Opponent Country', 'Bowling Style'])
                           for country in top_countries}

            for player_key, df in player_data.items():
                player_name = player_key.split("_")[0]
                role = player_key.split("_")[-1]

                if role == 'batting':
                    try:
                        df_clean = clean_batter_data(df)
                        if df_clean.empty:
                            continue

                        df_clean['Impact'] = df_clean.apply(calculate_batter_impact, axis=1)
                        df_clean = df_clean.dropna(subset=['Impact'])

                        for country in top_countries:
                            matches = df_clean[df_clean['Opposition'].str.contains(country, case=False, na=False)]
                            if len(matches) >= 3:
                                avg_impact = matches['Impact'].mean()
                                new_row = {"Player": player_name, "Average_Impact": avg_impact, "Opponent Country": country}
                                batters_dfs[country] = pd.concat([batters_dfs[country], pd.DataFrame([new_row])],
                                                                 ignore_index=True)
                    except Exception as e:
                        main_display.append(pn.pane.Markdown(f"**Error analyzing batter {player_key}: {e}**"))

                elif role == 'bowling':
                    try:
                        df_clean = clean_bowler_data(df)
                        if df_clean.empty:
                            continue

                        df_clean['Impact'] = df_clean.apply(calculate_bowler_impact, axis=1)
                        df_clean = df_clean.dropna(subset=['Impact'])

                        for country in top_countries:
                            matches = df_clean[df_clean['Opposition'].str.contains(country, case=False, na=False)]
                            if len(matches) >= 3:
                                avg_impact = matches['Impact'].mean()
                                new_row = {"Player": player_name, "Average_Impact": avg_impact, "Opponent Country": country}
                                bowlers_dfs[country] = pd.concat([bowlers_dfs[country], pd.DataFrame([new_row])],
                                                                 ignore_index=True)
                    except Exception as e:
                        main_display.append(pn.pane.Markdown(f"**Error analyzing bowler {player_key}: {e}**"))

            # Show results
            main_display.append(pn.pane.Markdown("### Visualizing Country-wise Performance"))
            for country, df in batters_dfs.items():
                main_display.append(pn.pane.Markdown(f"#### Batters against {country}"))
                main_display.append(pn.widgets.DataFrame(df, name=f"Batters vs {country}"))

            for country, df in bowlers_dfs.items():
                main_display.append(pn.pane.Markdown(f"#### Bowlers against {country}"))
                main_display.append(pn.widgets.DataFrame(df, name=f"Bowlers vs {country}"))

            # Additional Analysis: Player vs Countries
            def player_vs_countries(batters_dfs, bowlers_dfs):
                # Initialize empty dictionaries to store best performers for each country
                best_batters = {}
                best_bowlers = {}

                # Loop through each opponent in batters_dfs
                for opponent, batter_df in batters_dfs.items():
                    # Initialize an empty DataFrame to store best batter per nationality for each opponent
                    best_batter_df = pd.DataFrame(
                        columns=['Player', 'Average_Impact', 'Opponent Country', 'Batting Style', 'Nationality'])

                    # Group by Nationality and find the batter with the highest Average_Impact
                    for nationality, group in batter_df.groupby('Nationality'):
                        best_batter = group.loc[group['Average_Impact'].idxmax()]
                        best_batter_df = pd.concat([best_batter_df, pd.DataFrame([best_batter])], ignore_index=True)

                    # Save the results for this opponent
                    best_batters[opponent] = best_batter_df

                # Repeat the same for bowlers_dfs to get best bowler per nationality
                for opponent, bowler_df in bowlers_dfs.items():
                    # Initialize an empty DataFrame to store best bowler per nationality for each opponent
                    best_bowler_df = pd.DataFrame(
                        columns=['Player', 'Average_Impact', 'Opponent Country', 'Bowling Style', 'Nationality'])

                    # Group by Nationality and find the bowler with the highest Average_Impact
                    for nationality, group in bowler_df.groupby('Nationality'):
                        best_bowler = group.loc[group['Average_Impact'].idxmax()]
                        best_bowler_df = pd.concat([best_bowler_df, pd.DataFrame([best_bowler])], ignore_index=True)

                    # Save the results for this opponent
                    best_bowlers[opponent] = best_bowler_df

                # Print example output for each opponent's best batter and bowler per nationality
                for opponent in best_batters:
                    print(f"Best Batters vs {opponent}:")
                    print(best_batters[opponent])
                    print("\n")

                for opponent in best_bowlers:
                    print(f"Best Bowlers vs {opponent}:")
                    print(best_bowlers[opponent])
                    print("\n")

            player_vs_countries(batters_dfs, bowlers_dfs)
        except Exception as e:
            main_display.append(pn.pane.Markdown(f"**Error in country-wise analysis: {e}**"))

    run_countrywise_analysis()

    return main_display



def show_prediction_local():
    try:
        # Fetching local player data
        player_data,player_info_df = fetch_local_player_data()
        if not player_data:
            return pn.pane.Markdown("**Error: No local data found. Please check the directory.**")
    except Exception as e:
        return pn.pane.Markdown(f"**Error fetching local data: {e}**")
    print(type(player_data))
    print(player_data)
    # Filter out players in P_Test_list
    all_iplayers = {key: value for key, value in player_data.items() if key not in P_Test_list}
    print(f"Players to process: {list(all_iplayers.keys())}")

    # Initialize DataFrame for predictions
    predictions_df = pd.DataFrame(columns=[
        'Player Name', 'Type', 'Position', 'Expected Impact',
        'Prediction 1', 'Prediction 2', 'Prediction 3'
    ])

    # Iterate over players in all_iplayers
    for player_name, data_frame in all_iplayers.items():
        try:
            print(f"Processing player: {player_name}")
            if "batting" in player_name.lower():
                # Batting predictions
                try:
                    batter_df_clean = clean_batter_data(data_frame)
                    batter_df_clean = filter_recent_data(batter_df_clean)
                    print(f"Batter DF Clean for {player_name}:{batter_df_clean.head()}")

                    if not batter_df_clean.empty:
                        batter_df_clean['Impact'] = batter_df_clean.apply(calculate_batter_impact, axis=1)
                        batter_df_clean = batter_df_clean.dropna(subset=['Impact'])
                        batter_df_clean = add_ema_impact(batter_df_clean, span=10)
                        train, test = split_time_series(batter_df_clean)
                        most_common_position = get_most_common_position(bowler_df_clean)

                        if len(test) > 0:
                            model = train_arima(train)
                            predictions = predict_arima(model, steps=3)
                            predictions = np.abs(predictions)

                            # Calculate averages and predictions
                            avg_impact = np.mean(predictions)
                            prediction_details = []
                            for idx, predicted_value in enumerate(predictions):
                                runs, sr = batting_performance(predicted_value, most_common_position)
                                prediction_details.append(f"Runs: {runs}, SR: {sr}")

                            # Add to DataFrame
                            predictions_df.loc[len(predictions_df)] = {
                                'Player Name': player_name,
                                'Type': 'Batting',
                                'Position': most_common_position,  # Add position logic if needed
                                'Expected Impact': avg_impact,
                                'Prediction 1': prediction_details[0],
                                'Prediction 2': prediction_details[1],
                                'Prediction 3': prediction_details[2]
                            }

                except Exception as e:
                    print(f"### Error processing batting data for {player_name}: {e}")

            elif "bowling" in player_name.lower():
                # Bowling predictions
                try:
                    bowler_df_clean = clean_bowler_data(data_frame)
                    bowler_df_clean = filter_recent_data(bowler_df_clean)
                    print(f"Bowler DF Clean for {player_name}:{bowler_df_clean.head()}")

                    if not bowler_df_clean.empty:
                        bowler_df_clean['Impact'] = bowler_df_clean.apply(calculate_bowler_impact, axis=1)
                        bowler_df_clean = bowler_df_clean.dropna(subset=['Impact'])
                        bowler_df_clean = add_ema_impact(bowler_df_clean, span=10)
                        train, test = split_time_series(bowler_df_clean)

                        if len(test) > 0:
                            model = train_arima(train)
                            predictions = predict_arima(model, steps=3)
                            predictions = np.abs(predictions)

                            # Calculate averages and predictions
                            avg_impact = np.mean(predictions)
                            prediction_details = []
                            for idx, predicted_value in enumerate(predictions):
                                performance_string = bowling_performance(predicted_value)
                                wickets, runs_conceded, overs = parse_bowling_performance(performance_string)
                                eco_rate = runs_conceded/overs
                                prediction_details.append(f"Wickets: {wickets}, Eco: {eco_rate}")

                            # Add to DataFrame
                            predictions_df.loc[len(predictions_df)] = {
                                'Player Name': player_name,
                                'Type': 'Bowling',
                                'Position': 'N/A',
                                'Expected Impact': avg_impact,
                                'Prediction 1': prediction_details[0],
                                'Prediction 2': prediction_details[1],
                                'Prediction 3': prediction_details[2]
                            }

                except Exception as e:
                    print(f"### Error processing bowling data for {player_name}: {e}")

        except Exception as e:
            print(f"### Error processing {player_name}: {e}")

    # Debugging: Print DataFrame to ensure it has data
    print("Final Predictions DataFrame:")
    print(predictions_df)

    # Check if predictions_df is empty
    if predictions_df.empty:
        return pn.pane.Markdown("**No predictions available. Please check the input data.**")

    # Display the DataFrame on the Panel Dashboard using Tabulator
    predictions_tabulator = pn.widgets.Tabulator(
        predictions_df, show_index=False, theme="modern", pagination="local", page_size=10
    )

    return pn.Column(
        pn.pane.Markdown("# Consolidated Player Predictions"),
        predictions_tabulator
    )



# Visualization
def create_bar_chart(data, title):
    """Create a bar chart for top players."""
    data = sorted(data, key=lambda x: x[1], reverse=True)[:10]
    if not data:
        return pn.pane.Markdown(f"**No data available for {title}**")

    player_names, impact_values = zip(*data)

    fig = go.Figure(
        data=[go.Bar(
            x=impact_values,
            y=player_names,
            orientation='h',
            marker=dict(color='skyblue')
        )]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Impact Rating",
        yaxis_title="Player Name",
        yaxis=dict(autorange='reversed'),
        template="plotly_white"
    )
    return pn.pane.Plotly(fig, config={'responsive': True})


# Analysis Functions
def countrywise_comparison(player_data, player_info_df):
    country_impact = {}

    for player_key, df in player_data.items():
        if "batting" in player_key:
            try:
                df_clean = clean_batter_data(df)
                df_clean = filter_recent_data(df_clean)
                if df_clean.empty:
                    continue

                df_clean["Impact"] = df_clean.apply(calculate_batter_impact, axis=1)
                player_name = clean_player_key(player_key)

                if player_name not in player_info_df["Full Name"].values:
                    print(f"Warning: Player '{player_name}' not found in player_info_df.")
                    continue

                player_country = player_info_df.loc[player_info_df["Full Name"] == player_name, "Country"].values[0]

                if player_country not in country_impact:
                    country_impact[player_country] = []
                country_impact[player_country].append(df_clean["Impact"].mean())
            except Exception as e:
                print(f"Error processing player '{player_key}': {e}")

    avg_country_impact = {k: np.mean(v) for k, v in country_impact.items()}

    countries = list(avg_country_impact.keys())
    values = list(avg_country_impact.values())

    # Plotly Bar Chart
    fig = px.bar(
        x=countries,
        y=values,
        labels={"x": "Country", "y": "Average Impact"},
        title="Average Player Impact by Country",
    )

    fig.update_layout(xaxis_title="Country", yaxis_title="Average Impact")
    return pn.pane.Plotly(fig)


def extract_player_key(player_key):
    """Clean the player key to extract the proper player name (excluding '_batting' or '_bowling')."""
    return player_key.replace('_batting', '').replace('_bowling', '')


def all_rounder_impact_plot(player_data):
    all_rounder_stats = []

    # Iterate over player data to identify all-rounders and calculate their impacts
    for player_key in player_data:
        try:
            # Clean player key to get the player name (by removing _batting/_bowling)
            player_name = extract_player_key(player_key)

            # Check if we have both batting and bowling data for this player
            if f"{player_name}_batting" in player_data and f"{player_name}_bowling" in player_data:
                # Get batting data
                batting_df = player_data.get(f"{player_name}_batting")
                df_batting_clean = clean_batter_data(batting_df)
                if df_batting_clean.empty:
                    continue
                df_batting_clean = filter_recent_data(df_batting_clean)

                # Get bowling data
                bowling_df = player_data.get(f"{player_name}_bowling")
                df_bowling_clean = clean_bowler_data(bowling_df)
                if df_bowling_clean.empty:
                    continue
                df_bowling_clean = filter_recent_data(df_bowling_clean)

                # Calculate batting and bowling impact
                df_batting_clean["Impact"] = df_batting_clean.apply(calculate_batter_impact, axis=1)
                df_bowling_clean["Impact"] = df_bowling_clean.apply(calculate_bowler_impact, axis=1)

                # Average impact values
                batting_impact = df_batting_clean["Impact"].mean() if not df_batting_clean.empty else 0
                bowling_impact = df_bowling_clean["Impact"].mean() if not df_bowling_clean.empty else 0

                all_rounder_stats.append([player_name, batting_impact, bowling_impact])
        except Exception as e:
            print(f"Error processing player '{player_key}': {e}")

    # Create a DataFrame for the all-rounders
    all_rounder_df = pd.DataFrame(all_rounder_stats, columns=["Player", "Batting Impact", "Bowling Impact"])

    # Scatterplot for Best All-Rounders: Batting Impact vs Bowling Impact
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=all_rounder_df["Bowling Impact"],
            y=all_rounder_df["Batting Impact"],
            mode="markers+text",
            text=all_rounder_df["Player"],
            textposition="top center",
            marker=dict(size=12, color="orange", opacity=0.7),
        )
    )

    fig.update_layout(
        title="Best All-Rounders: Batting Impact vs Bowling Impact",
        xaxis_title="Bowling Impact",
        yaxis_title="Batting Impact",
        template="plotly_white",
        height=600,
        width=800,
    )

    # Return Plotly graph as a Panel pane
    return pn.pane.Plotly(fig, width=900)

def top_batters_and_heatmap(player_data):
    main_countries = [
        "v India", "v Australia", "v England", "v South Africa", "v New Zealand",
        "v Pakistan", "v West Indies", "v Sri Lanka", "v Bangladesh", "v Afghanistan"
    ]

    # Step 1: Calculate Overall Impact for Batters
    batter_impact = []

    for player_key, df in player_data.items():
        if "batting" in player_key:
            try:
                df_clean = clean_batter_data(df)
                df_clean = filter_recent_data(df_clean)
                if df_clean.empty:
                    continue

                df_clean["Impact"] = df_clean.apply(calculate_batter_impact, axis=1)
                player_name = clean_player_key(player_key)

                # Append overall impact
                if not df_clean["Impact"].empty:
                    batter_impact.append((player_name, df_clean["Impact"].mean()))
            except Exception as e:
                print(f"Error processing player '{player_key}': {e}")

    # Sort and select top 15 batters
    top_batters = sorted(batter_impact, key=lambda x: x[1], reverse=True)[:15]
    top_batter_names = [b[0] for b in top_batters]

    # Step 2: Construct Heatmap Data for Top Batters vs Main Countries
    heatmap_data = {player: {country: 0 for country in main_countries} for player in top_batter_names}

    for player_key, df in player_data.items():
        if "batting" in player_key:
            try:
                df_clean = clean_batter_data(df)
                df_clean = filter_recent_data(df_clean)
                if df_clean.empty:
                    continue

                df_clean["Impact"] = df_clean.apply(calculate_batter_impact, axis=1)
                player_name = clean_player_key(player_key)

                if player_name in top_batter_names:
                    for opponent in df_clean["Opposition"].unique():
                        if opponent in main_countries:
                            impacts = df_clean[df_clean["Opposition"] == opponent]["Impact"]
                            if not impacts.empty:
                                heatmap_data[player_name][opponent] = impacts.mean()
            except Exception as e:
                print(f"Error processing player '{player_key}': {e}")

    # Convert to DataFrame for Heatmap
    print(heatmap_data)
    print(main_countries)
    heatmap_df = pd.DataFrame(heatmap_data).T[main_countries]

    # # Step 3: Create Bar Chart for Top 15 Batters
    # batter_df = pd.DataFrame(top_batters, columns=["Player", "Impact"])
    # bar_chart = go.Figure(
    #     data=go.Bar(x=batter_df["Player"], y=batter_df["Impact"], marker_color="blue")
    # )
    # bar_chart.update_layout(
    #     title="Top 15 Batters Overall Impact",
    #     xaxis_title="Player",
    #     yaxis_title="Impact",
    # )

    # Step 4: Create Heatmap for Top Batters vs Main Countries
    heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale="viridis",
        )
    )
    heatmap.update_layout(
        title="Top 15 Batters Impact vs Countries",
        xaxis_title="Opposition",
        yaxis_title="Players",
    )

    # Combine Outputs in a Panel Layout
    return pn.Column(
        pn.pane.Plotly(heatmap),
        sizing_mode="stretch_width",
    )


import pandas as pd
import plotly.graph_objects as go
import panel as pn

import pandas as pd
import plotly.graph_objects as go


def classify_bowler(player_info_df):
    """Classify bowlers into Fast and Spin categories based on their bowling style."""
    fast_bowler_keywords = ['fast', 'medium']
    spin_bowler_keywords = [
        'orthodox', 'offspinner', 'legspinner', 'wristspinner',
        'wrist spin', 'leg spin', 'off spinner', 'offbreak', 'off break'
    ]

    # Convert 'Bowling Style' column to lowercase for comparison
    player_info_df['Bowling Style'] = player_info_df['Bowling Style'].str.lower()

    # Filter players based on their lowercase bowling style column
    fast_bowlers = player_info_df[
        player_info_df['Bowling Style'].str.contains('|'.join(fast_bowler_keywords), na=False)]
    spin_bowlers = player_info_df[
        player_info_df['Bowling Style'].str.contains('|'.join(spin_bowler_keywords), na=False)]

    return fast_bowlers, spin_bowlers


def process_bowler(player_key, player_name, bowler_type, player_data):
    """Process each bowler and calculate their bowling impact."""
    try:
        player_name_key = player_name.replace(" ", "_")
        bowling_df = player_data.get(f"{player_name_key}_bowling")
        df_bowling_clean = clean_bowler_data(bowling_df)  # You should have this function defined
        if df_bowling_clean.empty:
            return None

        # Calculate bowling impact
        df_bowling_clean["Impact"] = df_bowling_clean.apply(calculate_bowler_impact,
                                                            axis=1)  # Also define this function
        bowling_impact = df_bowling_clean["Impact"].mean() if not df_bowling_clean.empty else 0

        return [player_name, bowler_type, bowling_impact]
    except Exception as e:
        return None


def top_bowlers_plot(player_data, player_info_df):
    # Classify bowlers into Fast and Spin categories
    fast_bowlers, spin_bowlers = classify_bowler(player_info_df)

    # Process Fast bowlers
    fast_bowler_stats = []
    for player_name in fast_bowlers['Full Name']:
        result = process_bowler(player_name, player_name, "Fast Bowler", player_data)
        if result:
            fast_bowler_stats.append(result)

    # Process Spin bowlers
    spin_bowler_stats = []
    for player_name in spin_bowlers['Full Name']:
        result = process_bowler(player_name, player_name, "Spin Bowler", player_data)
        if result:
            spin_bowler_stats.append(result)

    # Combine Fast and Spin stats into a DataFrame
    all_bowler_stats = fast_bowler_stats + spin_bowler_stats
    bowler_stats_df = pd.DataFrame(all_bowler_stats, columns=["Player", "Bowler Type", "Bowling Impact"])

    # Sort by Bowling Impact and take top 10 from each category
    top_fast_bowlers = bowler_stats_df[bowler_stats_df["Bowler Type"] == "Fast Bowler"].nlargest(10, 'Bowling Impact')
    top_spin_bowlers = bowler_stats_df[bowler_stats_df["Bowler Type"] == "Spin Bowler"].nlargest(10, 'Bowling Impact')

    # Combine top 10 from each category
    top_bowlers = pd.concat([top_fast_bowlers, top_spin_bowlers])

    # Plotting the top 10 bowlers (Fast vs. Spin)
    fig = go.Figure()

    # Add Fast bowlers trace
    fig.add_trace(
        go.Bar(
            x=top_fast_bowlers["Player"],
            y=top_fast_bowlers["Bowling Impact"],
            name="Fast Bowlers",
            text=top_fast_bowlers["Player"],
            textposition="auto",
            marker=dict(color='blue', opacity=0.7),
        )
    )

    # Add Spin bowlers trace
    fig.add_trace(
        go.Bar(
            x=top_spin_bowlers["Player"],
            y=top_spin_bowlers["Bowling Impact"],
            name="Spin Bowlers",
            text=top_spin_bowlers["Player"],
            textposition="auto",
            marker=dict(color='green', opacity=0.7),
        )
    )

    # Update layout
    fig.update_layout(
        title="Top 10 Bowlers: Fast vs. Spin",
        xaxis_title="Player",
        yaxis_title="Bowling Impact",
        barmode="group",
        height=600,
        width=900,
        template="plotly_white"
    )

    return fig


def clean_player_key(player_key):
    """Clean player key to extract proper name."""
    cleaned_name = player_key.replace('_', ' ')
    cleaned_name = cleaned_name.replace(' batting', '').replace(' bowling', '')
    return cleaned_name

# Main execution
def run_all_analyses():
    """Run all analysis functions and return a combined layout."""
    player_data, player_info_df = fetch_local_player_data()
    layout = pn.Column(
        pn.pane.Markdown("# Comprehensive Player Analysis Dashboard"),
        pn.Tabs(
    ("All Rounders Analysis", all_rounder_impact_plot(player_data)),
            ("Top Batters Analysis", top_batters_and_heatmap(player_data)),
            ("Top Bowlers Analysis", top_bowlers_plot(player_data,player_info_df)),
            ("Country-wise Comparison", countrywise_comparison(player_data, player_info_df))
        )
    )
    return layout

def show_analysis_all():
    def run_analysis(event=None):
        try:
            # Step 1: Fetch data for all players
            global player_data, player_info_df

            # Initialize the player_data dictionary if not already defined
            if "player_data" not in globals():
                player_data = {}
            else:
                print("Player data already exists. Skipping redundant steps.")
                return

            # Get the current date in the required format (e.g., "1+Dec+2024")
            current_date = datetime.now().strftime("%d+%b+%Y")  # Example: "13+Dec+2024"

            # Update the URLs dynamically with the current date
            batting_url = f'https://stats.espncricinfo.com/ci/engine/stats/index.html?batting_positionmax2=7;batting_positionmin2=1;batting_positionval2=batting_position;class=3;filter=advanced;orderby=runs;qualmin1=5;qualval1=innings;size=200;spanmax1={current_date};spanmin1=1+Dec+2019;spanval1=span;team=1;team=2;team=25;team=3;team=4;team=40;team=5;team=6;team=7;team=8;template=results;type=batting'
            bowling_url = f'https://stats.espncricinfo.com/ci/engine/stats/index.html?class=3;filter=advanced;orderby=wickets;qualmin2=5;qualval2=innings_bowled;size=200;spanmax1={current_date};spanmin1=1+Dec+2019;spanval1=span;team=1;team=2;team=25;team=3;team=4;team=40;team=5;team=6;team=7;team=8;template=results;type=bowling'

            # Step 2: Get player names for batting and bowling roles
            batting_players = get_player_names(batting_url)
            bowling_players = get_player_names(bowling_url)

            # Combine and deduplicate
            all_players_list = list(set(batting_players + bowling_players))
            print(f"Total Players Fetched: {len(all_players_list)}")

            # Step 3: Fetch player metadata (player_info_df)
            player_info_df = get_players_info(all_players_list)

            # Step 4: Map playing roles to batting and bowling
            playing_role_map = {
                "Batter": ["batting"],
                "Top order Batter": ["batting"],
                "Middle order Batter": ["batting"],
                "Opening Batter": ["batting"],
                "Wicketkeeper Batter": ["batting"],
                "Bowler": ["bowling"],
                "Spin Bowler": ["bowling"],
                "Pace Bowler": ["bowling"],
                "Batting Allrounder": ["batting", "bowling"],
                "Bowling Allrounder": ["batting", "bowling"],
                "Allrounder": ["batting", "bowling"]
            }

            # Step 5: Process each player and fetch stats
            not_found_players = []
            for _, row in player_info_df.iterrows():
                player_name = row["Full Name"]
                roles = playing_role_map.get(row["Playing Role"], [])

                # Fetch data for the player's roles
                for role in roles:
                    formatted_name = player_name.replace(" ", "_")
                    df_name = f"{formatted_name}_{role}"

                    try:
                        # Fetch data for this role
                        _ = get_and_save_player_stats(player_name, [role])  # Save stats globally

                        # Access the saved dataframe using the global variable
                        if df_name in globals():
                            df = globals()[df_name]
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                player_data[df_name] = df
                            else:
                                print(f"No data available in dataframe {df_name}.")
                                not_found_players.append(player_name)
                        else:
                            print(f"Global dataframe {df_name} not found.")
                            not_found_players.append(player_name)

                    except Exception as e:
                        print(f"Error fetching data for {player_name} ({role}): {e}")
                        not_found_players.append(player_name)

            # Log players for whom data was not found
            if not_found_players:
                print(f"Data not found for the following players: {', '.join(not_found_players)}")

            print(f"Total dataframes in player_data: {len(player_data)}")

        except Exception as e:
            print(f"An error occurred during analysis: {e}")

    # Run the analysis and prepare plots
    run_analysis()

    # Step 6: Call the required plotting functions
    layout = pn.Column(
        pn.pane.Markdown("# Comprehensive Player Analysis Dashboard"),
        pn.Tabs(
            ("All Rounders Analysis", all_rounder_impact_plot(player_data)),
            ("Top Batters Analysis", top_batters_and_heatmap(player_data)),
            ("Top Bowlers Analysis", top_bowlers_plot(player_data, player_info_df)),
            ("Country-wise Comparison", countrywise_comparison(player_data, player_info_df))
        )
    )
    return layout





if __name__ == "__main__":
    pn.extension(theme='dark')  # Apply a dark theme

    # Custom CSS for buttons
    button_style = """
    button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 14px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
        overflow-wrap: break-word; /* Wraps text within the button */
        white-space: normal; /* Allows multiple lines of text */
    }
    """

    pn.extension(css=button_style)

    # Sidebar style
    sidebar_style = """
    .sidebar {
        background-color: #222;
    }
    """
    pn.extension(css=sidebar_style)
    pn.extension('plotly')

    # Sidebar and main display area setup
    sidebar = pn.Column(
        pn.pane.Markdown("## Player Impact Analysis Menu"),
        width=300  # Increased width for better text fitting
    )
    main_display = pn.Column(
        pn.pane.Markdown("### Select an option from the menu"),
        margin=(0, 50)  # Add some left margin to avoid overlap
    )
    pn.extension('tabulator')

    # Function to update main_display based on button selection
    def update_display(callback):
        main_display.clear()
        main_display.append(callback())

    # Adding buttons to sidebar with labels formatted for better readability
    buttons = [
        ("Analyze a Specific Player", show_analyze_player),
        ("Compare Multiple Players", show_compare_players),
        ("Predict Specific Player's Performance", show_predict_player),
        ("All Players Predictions\n(Data Updated till 1 Dec 24)", show_prediction_local),
        ("All Players Analysis\n(Data Updated till 1 Dec 24)", run_all_analyses),
        ("All Players Predictions\n(RealTime Data - Takes Longer to run)", show_predictions_all),
        ("All Players Analysis\n(RealTime Data - Takes Longer to run)", show_analysis_all)
    ]

    for label, callback in buttons:
        button = pn.widgets.Button(name=label, width=280)  # Wider buttons for longer text
        button.on_click(lambda event, cb=callback: update_display(cb))
        sidebar.append(button)

    # Add Exit button to reset or exit the app
    exit_button = pn.widgets.Button(name="Exit", width=280)  # Consistent width
    exit_button.on_click(lambda event: pn.state.clear())
    sidebar.append(exit_button)

    # Combine sidebar and main display into the main dashboard layout
    dashboard = pn.Row(sidebar, main_display)

    # Serve the dashboard
    pn.serve(dashboard, start=True, show=True, port=5006)




