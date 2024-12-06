from flask import request, jsonify, make_response
import pandas as pd
import json
import os
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.models import load_model
import time
import os
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

useHardCodedData = os.getenv('USE_HARDCODED_DATA', default=True)
current_model = None

def convert_gpa_to_numeric(gpa_value):
    if isinstance(gpa_value, str):
        # Remove any non-numeric characters (like commas or symbols) except for dots
        cleaned_gpa = re.sub(r'[^\d.]', '', gpa_value)

        try:
            # Attempt to convert cleaned GPA to float
            return float(cleaned_gpa)
        except ValueError:
            # If conversion fails, return NaN
            return None
    return gpa_value

# Function to convert string date format "MM/YYYY" to datetime
def convert_month_year_to_date(date_str):
    try:
        return datetime.strptime(str(date_str), "%m/%Y")
    except (ValueError, TypeError):
        return None

# Function to convert answer to its corresponding weightage
def convert_to_weightage(answer, question):
    options = question.get('options', [])
    for option in options:
        if answer == option['optionValue']:
            return option['optionWeightage']
    return None

def scale_with_pickle_file(action, data, questionSlugAndType, model_name, model_dir):

    # Convert the dictionary to a Pandas DataFrame
    if str(action).lower() == 'predict':
        df = pd.DataFrame([data])

    elif str(action).lower() == 'train':
        df = pd.DataFrame(data)

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    if str(action).lower() == 'train':

        # Fit and transform the data using questionSlugAndType
        with open(os.path.join(model_dir, 'questionSlugAndType.json'), 'r') as json_file:
            total_questions = json.load(json_file)

        # Add any missing columns in prediction data with default value 0
        for key in total_questions.keys():
            if key not in df.columns:
                df[key] = 0

        # Drop the 'startdate' and 'enddate' columns since they are not needed for scaling
        df_dropped = df.drop(columns=['startdate', 'enddate'])

        print(f"Data columns used for train scaling................\n: {df_dropped.columns}")

        scaled_data = scaler.fit_transform(df_dropped)

        # saving the columns after fitting for prediction fit use.
        with open(os.path.join(model_dir, '_fitted_data_questions.json'), 'w') as json_file:
            json.dump(list(df_dropped.keys()), json_file)


        # Save the fitted scaler
        scaler_file = os.path.join(model_dir, model_name + '_scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)

    elif str(action).lower() == 'predict':
        # Load the fitted scaler
        scaler_file = os.path.join(model_dir, model_name + '_scaler.pkl')

        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)


        # Convert startdate and enddate from "MM/YYYY" format to datetime
        df['startdate'] = df['startdate'].apply(convert_month_year_to_date)
        df['enddate'] = df['enddate'].apply(convert_month_year_to_date)
        # Calculate duration (in years) from startdate and enddate
        df['duration'] = (df['enddate'] - df['startdate']).dt.days / 365.25  # Approximate to years

        # Fit and transform the data using questionSlugAndType
        with open(os.path.join(model_dir, '_fitted_data_questions.json'), 'r') as json_file:
            total_questions = json.load(json_file)

        # Add any missing columns in prediction data with default value 0
        for key in total_questions:
            if key not in df.columns:
                df[key] = 0

        # Drop the 'startdate' and 'enddate' columns since they are not needed for scaling
        df_dropped = df.drop(columns=['startdate', 'enddate'])

        # Reorder df_dropped to maintain the same order as total_questions
        df_dropped = df_dropped[total_questions]

        print("Data Columns used for prediction scaling...............\n", df_dropped.columns)

        scaled_data = scaler.transform(df_dropped)

        data_order = list(questionSlugAndType)

        all_data_columns = set(df_dropped.columns)
        missing_cols = set(data_order) - all_data_columns

        # Assigning null values to the empty columns.
        for c in missing_cols:
            df_dropped[c] = 0
        df_dropped = df_dropped.drop(columns = "duration")
        print(f"After adding the missed columns {df_dropped.columns}")

    def success_rate():
      # Formula: 50% (gpa/4) + 50% (4/duration)
      scaled_df['success_rate'] = (0.5 * (scaled_df['gpa'] / 4) + 0.5 * (4 / scaled_df['duration'])) * 10

      # Ensure the success rate is capped at 100
      scaled_df['success_rate'] = scaled_df['success_rate'].clip(upper=100)

      print(f"After adding the Success Rate columns {scaled_df}")


    if str(action).lower() == 'train':
        scaled_df = pd.DataFrame(scaled_data, columns=df_dropped.columns)

        # print(f"Scaled data dataframe {scaled_df}")
        data_order = list(questionSlugAndType.keys())

        all_data_columns = set(df_dropped.columns)
        missing_cols = set(data_order) - all_data_columns

        # Assigning null values to the empty columns.
        for c in missing_cols:
            scaled_df[c] = 0

        print(f"Final Data before sending to train the model........\n {scaled_df.columns}")

        success_rate()
        x = scaled_df[list(data_order)]
        y = scaled_df["success_rate"]
        X_scaled = scaler.fit_transform(x)

        print(f"Columnd used for training the neural model....\n {x.columns} {y}")

        return X_scaled, y

    elif str(action).lower() == 'predict':
        return df_dropped

# Step 1: Load JSON data and preprocess GPA and Date Duration followed with converting to numeric.
def load_and_preprocess_data(input_data, model_dir):
    start_time = time.time()
    action = "Train"
    # Load the JSON file
    # if useHardCodedData.lower() == "true":
    #   file_path = 'models/model_test_new/training_seeded_data.json'
    #   # Reading the JSON file
    #   with open(file_path, 'r') as file:
    #     data = json.load(file)
    # else:
    data = input_data
    # Extracting the question details with their respective weightages
    questions = data['data']['questionSlugAndType'][0]['questions']

    model_name = data['data']['modelName']

    # Save questionSlugAndType to a JSON file for future use.
    with open(os.path.join(model_dir, 'questionSlugAndType.json'), 'w') as json_file:
        json.dump(questions, json_file)

    # Extracting the answers and converting them to weightages
    # Extracting the answers and converting them to weightages
    all_answers_with_weightages = []
    for response in data['data']['answers']:
        weighted_answers = {}
        for key, answer in response['answers'].items():
            if key != "startdate" and key != "enddate":
                question = questions.get(key)
                if question:
                    weightage = convert_to_weightage(answer, question)
                    weighted_answers[key] = weightage if weightage is not None else answer
                all_answers_with_weightages.append(weighted_answers)
            else:
                weighted_answers[key] = answer
                all_answers_with_weightages.append(weighted_answers)

    # print(f"All answers has been updated to its corresponding weitages: {all_answers_with_weightages}")

    # print(f"Type {type(all_answers_with_weightages)}")

    # Convert JSON data to DataFrame
    df = pd.json_normalize(all_answers_with_weightages)
    # Ensure GPA is numeric
    df['gpa'] = df['gpa'].apply(convert_gpa_to_numeric)
    # Convert startdate and enddate from "MM/YYYY" format to datetime
    df['startdate'] = df['startdate'].apply(convert_month_year_to_date)
    df['enddate'] = df['enddate'].apply(convert_month_year_to_date)
    # Calculate duration (in years) from startdate and enddate
    df['duration'] = (df['enddate'] - df['startdate']).dt.days / 365.25  # Approximate to years

    # Drop rows where GPA or Duration is missing or invalid
    df.dropna(subset=['gpa', 'duration'], inplace=True)

    # print(f"The final data set with course completion duration {df}")

    # Convert DataFrame to a list of dictionaries
    list_of_weighted_answers = df.to_dict(orient='records')

    # Now you can use list_of_weighted_answers as needed
    print(f"DataFrame converted back to list: {list_of_weighted_answers}")

    x,y=scale_with_pickle_file(action, list_of_weighted_answers, questions, model_name, model_dir)

    print(f"x value {x}")
    print(f"y value {y}")

    # Normalize the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(x)

    alphas = [0.01, 0.1, 1, 1.5, 4, 5, 10]
    # return train_and_evaluate_models(X_scaled, y, model_dir, model_name, start_time, alphas)
    return train_regression_model(x, y, model_dir, model_name, start_time)

# Step 3: Prepare the data for training (scaling and splitting)
def prepare_training_data(df, model_dir):
    #this will convert all non numeric to numeric and add duration column
    # print(load_and_preprocess_data(df, model_dir))
    return load_and_preprocess_data(df, model_dir)

def plot_analysis(test_y, prediction, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Residual plot
    residuals = abs(test_y - prediction)
    plt.figure(figsize=(8, 6))
    plt.scatter(prediction, residuals, edgecolor='k')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    residual_plot_path = os.path.join(output_dir, 'residual_plot.jpeg')
    plt.savefig(residual_plot_path, format='jpeg')

from sklearn.model_selection import learning_curve
import numpy as np

def plot_learning_curve(estimator, X, y, output_dir, scoring='neg_mean_squared_error', cv=5):
    """
    Plots a learning curve for a given estimator.
    
    Parameters:
    - estimator: The machine learning model (e.g., LinearRegression()).
    - X: Feature matrix.
    - y: Target variable.
    - scoring: Metric for evaluating performance (default is neg_mean_squared_error).
    - cv: Number of cross-validation folds (default is 5).
    """
    # Get learning curve data
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # Calculate mean and standard deviation of scores
    train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, -train_scores_mean, label="Training Error", color="r")
    plt.plot(train_sizes, -test_scores_mean, label="Validation Error", color="g")
    
    # # Fill between for standard deviation
    # plt.fill_between(train_sizes, 
    #                  -train_scores_mean - train_scores_std, 
    #                  -train_scores_mean + train_scores_std, 
    #                  alpha=0.2, color="r")
    # plt.fill_between(train_sizes, 
    #                  -test_scores_mean - test_scores_std, 
    #                  -test_scores_mean + test_scores_std, 
    #                  alpha=0.2, color="g")
    
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="best")
    plot_path = os.path.join(output_dir, 'learning_curve.jpeg')
    plt.savefig(plot_path, format='jpeg')
    plt.close()

    print(f"Learning curve saved to {plot_path}")


# Step 4: Train a simple neural network model using TensorFlow/Keras
def train_regression_model(x, y, model_dir, model_name, start_time):
    #Regression model
    print(f"Training regression model started..........\n")
    # Splitting data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Build the regression model
    # model = LinearRegression()
    model = Ridge(alpha=1.0) 

    # Train the model
    model.fit(X_train, Y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)
    r2_sc = r2_score(Y_test, predictions)

    # Saving the model
    model_path = os.path.join(model_dir, f'{model_name}.h5')
    import joblib
    joblib.dump(model, model_path)

    # End timing
    end_time = time.time()


    # Calculate the elapsed time
    total_time = end_time - start_time
    # NEED FOLLOWING METRICS
    output = {
    'absolute': mae.tolist(),  # Convert NumPy array to a Python list
    'squared': [float(mse)],
    'r2': [float(r2_sc)] ,
    'trainTime': [total_time],
    'filePath' :[model_path]
    }

    # Plot the analysis using the function
    plot_analysis(Y_test, predictions, model_dir)

    plot_learning_curve(model, x, y,model_dir, scoring='neg_mean_squared_error', cv=5)

    # Check environment variable
    model_env = os.getenv('MODELENV', 'local')

    if model_env == 'production' or model_env == 'development':
        # Return as a request for production
        print(f"Output: {output}")
        return make_response(jsonify(output), 200)
    else:
        # Return JSON output for development
        return json.dumps(output)
    
def plot_metrics_by_alpha(results_path, save_path):
    # Load the results from the JSON file
    with open(results_path, 'r') as file:
        results = json.load(file)
    
    # Prepare lists to store the alpha values and corresponding metrics
    alphas = []
    maes = []
    mses = []
    r2_scores = []
    
    # Extract metrics from results
    for result in results:
        alphas.append(result['alpha'])
        maes.append(result['absolute_error'])
        mses.append(result['mean_squared_error'])
        r2_scores.append(result['r2_score'])
    
    # Creating plots
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot Mean Absolute Error
    ax[0].plot(alphas, maes, marker='o', linestyle='-', color='blue')
    ax[0].set_title('Mean Absolute Error by Alpha')
    ax[0].set_xlabel('Alpha')
    ax[0].set_ylabel('Mean Absolute Error')
    ax[0].set_xscale('log')
    ax[0].grid(True)
    
    # Plot Mean Squared Error
    ax[1].plot(alphas, mses, marker='o', linestyle='-', color='red')
    ax[1].set_title('Mean Squared Error by Alpha')
    ax[1].set_xlabel('Alpha')
    ax[1].set_ylabel('Mean Squared Error')
    ax[1].set_xscale('log')
    ax[1].grid(True)
    
    # Plot R² Score
    ax[2].plot(alphas, r2_scores, marker='o', linestyle='-', color='green')
    ax[2].set_title('R² Score by Alpha')
    ax[2].set_xlabel('Alpha')
    ax[2].set_ylabel('R² Score')
    ax[2].set_xscale('log')
    ax[2].grid(True)
    
    plot_path = os.path.join(save_path, 'alpha_analysis.jpeg')
    plt.savefig(plot_path, format='jpeg')
    plt.close()

def train_and_evaluate_models(x, y, model_dir, model_name, start_time, alphas):
    best_model = None
    best_r2 = -float('inf')
    results = []

    for alpha in alphas:
        # Train the model with the current alpha
        model = Ridge(alpha=alpha)
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(Y_test, predictions)
        mse = mean_squared_error(Y_test, predictions)
        r2_sc = r2_score(Y_test, predictions)

        # Store results
        result = {
            'alpha': alpha,
            'absolute_error': mae,
            'mean_squared_error': mse,
            'r2_score': r2_sc
        }
        results.append(result)

        # Update best model if current model is better
        if r2_sc > best_r2:
            best_r2 = r2_sc
            best_model = model

    end_time = time.time()
    total_time = end_time - start_time

    # Optionally, save all results to a JSON file
    results_path = os.path.join(model_dir, 'all_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"best r2: {best_r2}")
    #ploting graph for each alpha
    plot_metrics_by_alpha(results_path, model_dir)

    # Saving the model
    model_path = os.path.join(model_dir, f'{model_name}.h5')
    import joblib
    joblib.dump(best_model, model_path)

    output = {
    'r2': [float(best_r2)] ,
    'trainTime': [total_time],
    'filePath' :[model_path]
    }

    # Check environment variable
    model_env = os.getenv('MODELENV', 'local')

    if model_env == 'production' or model_env == 'development':
        # Return as a request for production
        print(f"Output: {output}")
        return make_response(jsonify(output), 200)
    else:
        # Return JSON output for development
        return json.dumps(output)


def predict_regression_model():
    action='predict'

    #starting the timer
    start_time = time.time()

    if useHardCodedData.lower() == 'true':
       file_path = 'models/input_test_files/predict_seeded_dat_new.json'
       # Reading the JSON file
       with open(file_path, 'r') as file:
        data = json.load(file)
    else:
        data = request.json

    print(f"Data: {data}")
    # Extract the model path
    model_path = data['modelDetails']['filePath']
    model_name = data['modelDetails']['name']
    model_dir = os.path.join('models', model_name)

    # Extract the allAnswers dictionary
    all_answers = data['allAnswers']

    # Iterate over all the answers
    for answer in all_answers:
        for key, value in answer.items():
            if key != "startdate" and key != "enddate":
                if isinstance(value, dict):
                    # Replace the value with the weight if it's a dictionary
                    if 'weight' in value:
                        # Replace with the weight if 'weight' is available
                        answer[key] = value['weight']
                    elif 'value' in value:
                        # Replace with the value if 'value' is available but no weight
                        answer[key] = value['value']
                elif isinstance(value, list):
                    # Sum the weights if it's a list and replace the entry with the sum
                    answer[key] = sum(item['weight'] for item in value)
            else:
                answer[key] = value['value']

    # Save the updated data to a JSON variable
    updated_json_data = json.dumps(data, indent=4)

    print(f"updated_json_data: {updated_json_data}")

    data_dict = json.loads(updated_json_data)

    # Extract the `allAnswers` data
    answers = data_dict['allAnswers']

    # Flatten the `allAnswers` list of dictionaries into a single dictionary
    flattened_answers = {}
    for answer in answers:
        flattened_answers.update(answer)

    print(flattened_answers)
    # Convert the flattened dictionary into a DataFrame
    df = pd.DataFrame([flattened_answers])
    print(df)
    # Extract only the questions (keys)
    data_dict = json.loads(updated_json_data)
    questions = []
    for answer in data_dict['allAnswers']:
        questions.extend(answer.keys())

    #scaling the answers between 0 and 1
    scaled_data = scale_with_pickle_file(action, flattened_answers, questions, model_name, model_dir)

    # Load the model
    model = load_model(model_path)

    # Print the DataFrame
    print(f"scaled data: {scaled_data}")

    # Predict the result
    predictions = model.predict(scaled_data)

    # End timing
    end_time = time.time()

    # Calculate the elapsed time
    total_time = end_time - start_time

    # Round and calculate the success rate
    predicted_successRate = round(predictions[0][0] / 10, 2)

    output = {
    'successRate': predicted_successRate,
    'timeTook': total_time
    }

    print(f"Output: {output}")
    # Check environment variable
    model_env = os.getenv('MODELENV', 'development')

    if model_env == 'production' or model_env == 'development':
        # Return as a request for production
        return make_response(jsonify(output), 200)
    else:
        return json.dumps(output)



if __name__ == "__main__":
    file_path = "models/regression_model_test/SeededTrainingData_300.json"
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load the JSON data

    model_name = data['data']['modelName']
    model_dir = os.path.join('models', model_name)
    print(prepare_training_data(data, model_dir))
    # print(predict_regression_model())
