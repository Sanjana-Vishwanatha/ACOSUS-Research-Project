import random
from flask import Flask, request, jsonify, Response, make_response
from flask_cors import CORS
import pandas as pd
import json
import numpy as np
import os
from flask import Flask, request, jsonify
import pandas as pandas
from pickle import dump,load
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.layers import Dense,Activation,Dropout
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from pymongo import MongoClient
import numpy as np
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

import time
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

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
    except ValueError:
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
        # # Ensure that the input DataFrame matches the trained model's columns
        # # 1. Add missing columns with a default value of 0
        # for col in train_columns:
        #     if col not in df.columns:
        #         df[col] = 0

        # # 2. Drop any extra columns that are not in the trained model's columns
        # df = df[train_columns]

        # # 3. Reorder the columns to match the order from the training set
        # df = df[train_columns]

        # print("Columns after filtering and ordering:", df.columns)

    def success_rate():
      # Formula: 50% (gpa/4) + 50% (4/duration)
      scaled_df['success_rate'] = (0.5 * (scaled_df['gpa'] / 4) + 0.5 * (4 / scaled_df['duration'])) * 10

      # Ensure the success rate is capped at 100
      scaled_df['success_rate'] = scaled_df['success_rate'].clip(upper=100)

    #   print(f"After adding the Success Rate columns {scaled_df}")


    if str(action).lower() == 'train':
        scaled_df = pd.DataFrame(scaled_data, columns=df_dropped.columns)

        # print(f"Scaled data dataframe {scaled_df}")
        data_order = list(questionSlugAndType.keys())

        all_data_columns = set(df_dropped.columns)
        missing_cols = set(data_order) - all_data_columns

        # Assigning null values to the empty columns.
        for c in missing_cols:
            scaled_df[c] = 0

        # print(f"Final Data before sending to train the model........\n {scaled_df.columns}")

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
    # if useHardCodedData.lower() == 'true':
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

    return hyper_param_tuning(X_scaled, y, model_dir, model_name, start_time)


# Step 3: Prepare the data for training (scaling and splitting)
def prepare_training_data(df, model_dir):
    #this will convert all non numeric to numeric and add duration column
    # print( load_and_preprocess_data(df, model_dir))
    return load_and_preprocess_data(df, model_dir)

def plot_analysis(test_y, prediction, output_dir):
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # Flatten prediction if necessary
    if len(prediction.shape) > 1:
        prediction = prediction.flatten()

    # Residual plot
    residuals = test_y - prediction
    plt.figure(figsize=(10, 6))
    plt.scatter(prediction, residuals, edgecolor='k')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    residual_plot_path = os.path.join(output_dir, 'residual_plot_nn.jpeg')
    plt.savefig(residual_plot_path, format='jpeg')

# Step 4: Train a simple neural network model using TensorFlow/Keras
def train_neural_model(x, y, model_dir, model_name, start_time):

    #Neural model training starts
    traindata_x,testdata_x,traindata_y,testdata_y=train_test_split(x,y,test_size=0.2, random_state=42)
    model=Sequential([Dense(150, input_shape=(x.shape[1],),activation='relu'),
                 Dropout(0.1),
                 Dense(150,activation='relu'),
                 Dropout(0.1),
                 Dense(150,activation='relu'),
                 Dropout(0.1),
                 Dense(150,activation='relu'),
                 Dropout(0.1),
                 Dense(150,activation='relu'),
                 Dropout(0.1),
                 Dense(150,activation='relu'),
                 Dropout(0.1),
                 Dense(1,activation='relu')])
    model.compile(optimizer='adam',loss='mean_squared_error' ,metrics=['mean_squared_error'])
    model.summary()
    model_fit=model.fit(x=traindata_x,y=traindata_y,epochs=200,batch_size=10, validation_split=0.2)
    model_validate=model.predict(testdata_x)
    mae=mean_absolute_error(testdata_y,model_validate)

    #edited........
    mse_metric = MeanSquaredError()
    mse_metric.update_state(testdata_y, model_validate)
    mse = mse_metric.result().numpy()
    r2_sc=r2_score(testdata_y,model_validate)

    # Saving the model
    model_path = os.path.join(model_dir, f'{model_name}.h5')
    model.save(model_path)

    # Get the absolute file path
    file_path = os.path.abspath(f'{model_name}.h5')

    # End timing
    end_time = time.time()

    # Calculate the elapsed time
    total_time = end_time - start_time
    output = {
    'absolute': mae.tolist(),  # Convert NumPy array to a Python list
    'squared': [float(mse)],
    'r2': [float(r2_sc)] ,
    'trainTime': [total_time],
    'filePath' :[model_path]
    }

    # Check environment variable
    model_env = os.getenv('MODELENV', 'local')

    if model_env == 'production' or model_env == 'development':
        # Return as a request for production
        return make_response(jsonify(output), 200)
    else:
        # Return JSON output for development
        return json.dumps(output)

def train_with_hyperparameters(x, y, model_dir, model_name, start_time, best_hyperparam):
    print("Training with hyper parameters.....!")
    traindata_x,testdata_x,traindata_y,testdata_y=train_test_split(x,y,test_size=0.2, random_state=42)
    neurons = best_hyperparam["neurons"]
    dropout_rate = best_hyperparam["dropout_rate"]

    model = Sequential([
                        Dense(neurons, input_shape=(x.shape[1],), activation='relu'),
                        Dropout(dropout_rate),
                        Dense(neurons, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(neurons, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(neurons, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(neurons, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(1, activation='relu')
                    ])
    model.compile(optimizer='adam',loss='mean_squared_error' ,metrics=['mean_squared_error'])
    model.summary()
    model_fit=model.fit(x=traindata_x,y=traindata_y,epochs=best_hyperparam["epochs"],batch_size=best_hyperparam["batch_size"], validation_split=0.2)
    model_validate=model.predict(testdata_x)
    
    #ploting residual graph
    plot_analysis(testdata_y, model_validate, model_dir)

    #edited........
    mae=mean_absolute_error(testdata_y,model_validate)
    mse_metric = MeanSquaredError()
    mse_metric.update_state(testdata_y, model_validate)
    mse = mse_metric.result().numpy()
    r2_sc=r2_score(testdata_y,model_validate)

    # Saving the model
    model_path = os.path.join(model_dir, f'{model_name}.h5')
    model.save(model_path)

    # Get the absolute file path
    file_path = os.path.abspath(f'{model_name}.h5')

    # End timing
    end_time = time.time()

    # Calculate the elapsed time
    total_time = end_time - start_time
    output = {
    'absolute': mae.tolist(),  # Convert NumPy array to a Python list
    'squared': [float(mse)],
    'r2': [float(r2_sc)] ,
    'trainTime': [total_time],
    'filePath' :[model_path]
    }

    # Check environment variable
    model_env = os.getenv('MODELENV', 'local')

    if model_env == 'production' or model_env == 'development':
        # Return as a request for production
        return make_response(jsonify(output), 200)
    else:
        # Return JSON output for development
        return json.dumps(output)


def hyper_param_tuning(x, y,model_dir, model_name, start_time):
    print("\nTraining begins......")
    # Neural model training starts
    traindata_x, testdata_x, traindata_y, testdata_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # Define ranges for hyperparameters
    neurons_options = [50, 100]
    dropout_rates = [0.1, 0.2]
    batch_sizes = [10, 20]
    epochs_options = [100]

    execution_count = 0

    # Create a directory to save plots if it doesn't exist
    save_dir = 'training_plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_mse = float('inf')  # Start with a very high MSE
    best_mae = float('inf')  # Start with a very high MAE
    best_r2 = -float('inf')  # Start with a very low R²
    best_hyperparameters = {}

    # Loop through each combination of hyperparameters
    for neurons in neurons_options:
        for dropout_rate in dropout_rates:
            for batch_size in batch_sizes:
                for epochs in epochs_options:
                    execution_count += 1 
                    start_time = time.time()

                    print(f"\nExecution #{execution_count}")
                    print(f"Training with Hyperparameters: Neurons={neurons}, Dropout={dropout_rate}, Batch_size={batch_size}, Epochs={epochs}")
                    warnings.filterwarnings("ignore")

                    model = Sequential([
                        Dense(neurons, input_shape=(x.shape[1],), activation='relu'),
                        Dropout(dropout_rate),
                        Dense(neurons, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(neurons, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(neurons, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(neurons, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(1, activation='relu')
                    ])
                    
                    # Define R² as a custom metric
                    def r2_score(y_true, y_pred):
                        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
                        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
                        return 1 - (ss_res / (ss_tot + tf.keras.backend.epsilon()))

                    # Compile and train the model
                    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error', r2_score])

                    history = model.fit(traindata_x, traindata_y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

                    # Calculate time taken for this iteration
                    end_time = time.time()
                    iteration_time = end_time - start_time
                    print(f"Time taken: {iteration_time:.2f} seconds")
                    
                    # Access metrics for each epoch
                    training_mse = history.history['mean_squared_error']
                    training_mae = history.history['mean_absolute_error']
                    training_r2 = history.history['r2_score']
                    validation_mse = history.history['val_mean_squared_error']
                    validation_mae = history.history['val_mean_absolute_error']
                    validation_r2 = history.history['val_r2_score']

                    # Track the best hyperparameter set based on validation MSE and R²
                    final_validation_mse = validation_mse[-1]
                    final_validation_mae = validation_mae[-1]
                    final_validation_r2 = validation_r2[-1]

                    if final_validation_mse < best_mse:
                        best_mse = final_validation_mse
                        best_hyperparameters = {
                            'neurons': neurons,
                            'dropout_rate': dropout_rate,
                            'batch_size': batch_size,
                            'epochs': epochs
                        }
                        print(f"New best model based on MSE: {best_hyperparameters}, Validation MSE: {best_mse:.4f}")

                    if final_validation_mae < best_mae:
                        best_mae = final_validation_mae
                        best_hyperparameters = {
                            'neurons': neurons,
                            'dropout_rate': dropout_rate,
                            'batch_size': batch_size,
                            'epochs': epochs
                        }
                        print(f"New best model based on MAE: {best_hyperparameters}, Validation MAE: {best_mae:.4f}")

                    if final_validation_r2 > best_r2:
                        best_r2 = final_validation_r2
                        best_hyperparameters = {
                            'neurons': neurons,
                            'dropout_rate': dropout_rate,
                            'batch_size': batch_size,
                            'epochs': epochs
                        }
                        print(f"New best model based on R²: {best_hyperparameters}, Validation R²: {best_r2:.4f}")

                    # Print metrics for each epoch
                    # for epoch in range(len(training_mse)):
                    #     print(f"Epoch {epoch + 1}:")
                    #     print(f"  Training MSE: {training_mse[epoch]:.4f}, MAE: {training_mae[epoch]:.4f}, R²: {training_r2[epoch]:.4f}")
                    #     print(f"  Validation MSE: {validation_mse[epoch]:.4f}, MAE: {validation_mae[epoch]:.4f}, R²: {validation_r2[epoch]:.4f}")

                    # Plot metrics for each epoch
                    epochs_range = range(1, len(training_mse) + 1)
                    plt.figure(figsize=(18, 6))

                    # Plot MSE
                    plt.subplot(1, 3, 1)
                    plt.plot(epochs_range, training_mse, label='Training MSE', marker='o')
                    plt.plot(epochs_range, validation_mse, label='Validation MSE', marker='o')
                    plt.xlabel('Epochs')
                    plt.ylabel('MSE')
                    plt.title('Mean Squared Error Across Epochs')
                    plt.legend()
                    plt.grid(True)

                    # Plot MAE
                    plt.subplot(1, 3, 2)
                    plt.plot(epochs_range, training_mae, label='Training MAE', marker='o')
                    plt.plot(epochs_range, validation_mae, label='Validation MAE', marker='o')
                    plt.xlabel('Epochs')
                    plt.ylabel('MAE')
                    plt.title('Mean Absolute Error Across Epochs')
                    plt.legend()
                    plt.grid(True)

                    # Plot R²
                    plt.subplot(1, 3, 3)
                    plt.plot(epochs_range, training_r2, label='Training R²', marker='o')
                    plt.plot(epochs_range, validation_r2, label='Validation R²', marker='o')
                    plt.xlabel('Epochs')
                    plt.ylabel('R² Score')
                    plt.title('R² Score Across Epochs')
                    plt.legend()
                    plt.grid(True)

                    plt.tight_layout()

                    # Save the plot to the directory
                    plot_filename = os.path.join(save_dir, f'plot_execution_{execution_count}.png')
                    plt.savefig(plot_filename)

                    # Show the plot
                    plt.show()

    # After all executions, print the best hyperparameters found
    print(f"\nBest Hyperparameters based on validation MSE: {best_hyperparameters}")
    print(f"Best Validation MSE: {best_mse:.4f}")
    print(f"\nBest Hyperparameters based on validation MAE: {best_hyperparameters}")
    print(f"Best Validation MAE: {best_mae:.4f}")
    print(f"\nBest Hyperparameters based on validation R²: {best_hyperparameters}")
    print(f"Best Validation R²: {best_r2:.4f}")

    train_with_hyperparameters(x, y, model_dir, model_name, start_time, best_hyperparameters)


def predict_neural_model():
    action='predict'

    #starting the timer
    start_time = time.time()

    if useHardCodedData.lower() == 'true':
       file_path = 'models/input_test_files/predict_seeded_data_new_nn.json'
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
    file_path = "models/NeuralNetwork_model_test/SeededTrainingData.json"
    with open(file_path, 'r') as file:
        data = json.load(file)  # Load the JSON data

    model_name = data['data']['modelName']
    model_dir = os.path.join('models', model_name)
    print(prepare_training_data(data, model_dir))
    # print(predict_neural_model())
