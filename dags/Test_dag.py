
import requests
import json
import os
import pandas as pd
from airflow import DAG ,settings
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from airflow.operators.python import PythonOperator
from sklearn.model_selection import cross_val_score
from joblib import dump



default_args = {
    'owner': 'airflow',
    'start_date': datetime(2021, 1, 1),
    'schedule_interval': '@every 1 minutes',
}

dag = DAG('weather_data_processing', default_args=default_args, catchup=False)

# Define a variable to store the list of cities
cities = ['paris', 'london', 'washington','berlin','siena','rome','cairo','zurich','brussels','ottawa','athens','tokyo','moscow','riyadh','seoul','madrid','stockholm','ankara','canberra']

# Task 1: Retrieve weather data from OpenWeatherMap
def retrieve_weather_data():
    for city in cities:
        # Construct the URL to query OpenWeatherMap
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=ebbbdf1d21237bdca7987a73379e21e6"

        # Send the GET request and extract the JSON data
        response = requests.get(url)
        data = response.json()

        # Format the date and time for the filename
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Generate the filename for the JSON file with the city name
        file_name = f"{current_date}_{city.replace(' ', '_')}.json"

        # Write the JSON data to the file
        with open(f"/app/raw_files/{file_name}", 'w') as json_file:
            json.dump(data, json_file)

retrieve_weather_data_task = PythonOperator(
    task_id='retrieve_weather_data',
    python_callable=retrieve_weather_data,
    dag=dag
)



def transform_data_last_20():
    # Get data from raw files
    last_20_files = sorted(os.listdir("/app/raw_files"), reverse=True)[:20]

    # Transform data for the last 20 files
    data_frames = []
    for file in last_20_files:
        file_path = f"/app/raw_files/{file}"

        # Check if the file is not empty
        if os.path.getsize(file_path) > 0:
            with open(file_path, "r") as f:
                try:
                    data_temp = json.load(f)

                    temperature = data_temp["main"]["temp"]
                    city = data_temp["name"]
                    pressure = data_temp["main"]["pressure"]
                    date = file.split("_")[0]

                    data_frames.append({
                        "temperature": temperature,
                        "city": city,
                        "pressure": pressure,
                        "date": date,
                    })
                except json.decoder.JSONDecodeError as e:
                    print(f"Error loading JSON from file {file_path}: {e}")

    df = pd.DataFrame(data_frames)

    # Save data to data.csv file
    df.to_csv("/app/clean_data/data.csv", index=False)


def transform_data_all():
    # Get data from all raw files
    full_data_files = list(os.listdir("/app/raw_files"))

    # Transform data for all files
    data_frames = []
    for file in full_data_files:
        file_path = f"/app/raw_files/{file}"

        # Check if the file is not empty
        if os.path.getsize(file_path) > 0:
            with open(file_path, "r") as f:
                try:
                    data_temp = json.load(f)

                    temperature = data_temp["main"]["temp"]
                    city = data_temp["name"]
                    pressure = data_temp["main"]["pressure"]
                    date = file.split("_")[0]

                    data_frames.append({
                        "temperature": temperature,
                        "city": city,
                        "pressure": pressure,
                        "date": date,
                    })
                except json.decoder.JSONDecodeError as e:
                    print(f"Error loading JSON from file {file_path}: {e}")

    df = pd.DataFrame(data_frames)

    # Save data to fulldata.csv file
    df.to_csv("/app/clean_data/fulldata.csv", index=False)


# Define tasks in the DAG
transform_data_last_20_task = PythonOperator(
    task_id='transform_data_last_20',
    python_callable=transform_data_last_20,
    dag=dag,
)

transform_data_all_task = PythonOperator(
    task_id='transform_data_all',
    python_callable=transform_data_all,
    dag=dag,
)




def compute_model_score(model, X, y):
    # computing cross val
    cross_validation = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring='neg_mean_squared_error')

    model_score = cross_validation.mean()

    return model_score




def train_and_save_model(model, X, y, path_to_model='./app/model.pckl'):
    # training the model
    model.fit(X, y)
    # saving model
    print(str(model), 'saved at ', path_to_model)
    dump(model, path_to_model)


def prepare_data(path_to_data='/app/clean_data/fulldata.csv'):
    # reading data
    df = pd.read_csv(path_to_data)
    # ordering data according to city and date
    df = df.sort_values(['city', 'date'], ascending=True)

    dfs = []

    for c in df['city'].unique():
        df_temp = df[df['city'] == c]

        # creating target
        df_temp.loc[:, 'target'] = df_temp['temperature'].shift(1)

        # creating features
        for i in range(1, 10):
            df_temp.loc[:, 'temp_m-{}'.format(i)
                        ] = df_temp['temperature'].shift(-i)

        # deleting null values
        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    # concatenating datasets
    df_final = pd.concat(
        dfs,
        axis=0,
        ignore_index=False
    )

    # deleting date variable
    df_final = df_final.drop(['date'], axis=1)

    # creating dummies for city variable
    df_final = pd.get_dummies(df_final)

    features = df_final.drop(['target'], axis=1)
    target = df_final['target']

    return features, target
    
    
    
    
# Task 4': Train Linear Regression Model
def train_linear_regression_model():
    X, y = prepare_data('/app/clean_data/fulldata.csv')
    model = LinearRegression()
    train_and_save_model(model, X, y, '/app/clean_data/linear_regression_model.pickle')

train_linear_regression_model_task = PythonOperator(
    task_id='train_linear_regression_model',
    python_callable=train_linear_regression_model,
    dag=dag,
)
    
 
 
# Task 4'': Train Decision Tree Model
def train_decision_tree_model():
    X, y = prepare_data('/app/clean_data/fulldata.csv')
    model = DecisionTreeRegressor()
    train_and_save_model(model, X, y, '/app/clean_data/decision_tree_model.pickle')

train_decision_tree_model_task = PythonOperator(
    task_id='train_decision_tree_model',
    python_callable=train_decision_tree_model,
    dag=dag,
)

# Task 4''': Train Random Forest Model
def train_random_forest_model():
    X, y = prepare_data('/app/clean_data/fulldata.csv')
    model = RandomForestRegressor()
    train_and_save_model(model, X, y, '/app/clean_data/random_forest_model.pickle')

train_random_forest_model_task = PythonOperator(
    task_id='train_random_forest_model',
    python_callable=train_random_forest_model,
    dag=dag,
)



# Task 5: Select Best Model
def select_best_model():
    X, y = prepare_data('/app/clean_data/fulldata.csv')
    score_lr = compute_model_score(LinearRegression(), X, y)
    score_dt = compute_model_score(DecisionTreeRegressor(), X, y)
    score_rf = compute_model_score(RandomForestRegressor(), X, y)

    best_model = LinearRegression()  # Default to Linear Regression

    if score_dt < score_lr and score_dt < score_rf:
        best_model = DecisionTreeRegressor()
    elif score_rf < score_lr and score_rf < score_dt:
        best_model = RandomForestRegressor()

    # Save the best model
    train_and_save_model(best_model, X, y, '/app/clean_data/best_model.pickle')

select_best_model_task = PythonOperator(
    task_id='select_best_model',
    python_callable=select_best_model,
    dag=dag,
)
    
    
# Set the DAG dependencies
retrieve_weather_data_task >> transform_data_last_20_task
retrieve_weather_data_task >> transform_data_all_task >> [
    train_linear_regression_model_task,
    train_decision_tree_model_task,
    train_random_forest_model_task
] >> select_best_model_task



