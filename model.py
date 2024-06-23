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
# from keras.models import Model
# from keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from pymongo import MongoClient
import numpy as np
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import ReLU
from tensorflow.keras.metrics import MeanSquaredError

import time
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)
# CORS(app)


current_model = None

def convert_to_numeric(df):
    # Initialize LabelEncoder
    le = LabelEncoder()
    
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column].astype(str))
        elif df[column].dtype == 'bool':
            df[column] = df[column].astype(int)
        elif df[column].dtype == 'string':
            df[column] = le.fit_transform(df[column])

        # Convert NaN to 0
        df[column] = df[column].fillna(0)
    
    return df
#Function will scale the input values highest will be 1 and lowest will be 0.

def scale(action, data, questionSlugAndType):
    # Initialize the StandardScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    scaled_data = scaler.fit_transform(data)

    # Convert the scaled data back to a DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    print(f"Before adding the missed columns{scaled_df}")
    global transformed_data


    def success_rate():
        #assigning success rate using the priority given.
        priority_scores = {key: value['priorityScore'] for key, value in questionSlugAndType.items()}
        total_priority_score = sum(priority_scores.values())

        # Normalize the priority scores to get the weights
        weights = {key: score / total_priority_score for key, score in priority_scores.items()}

        # Calculate the success rate usning the priority given and doing 
        # sum for each student and assigning respectively.
        scaled_df['success_rate'] = 0  # Initialize the column with zeros
        for col, weight in weights.items():
            scaled_df['success_rate'] += scaled_df[col] * weight
            # print(f"After adding weight for {col}: {scaled_df['success_rate']}")
        print(f"After adding the Success Rate columns{scaled_df}")

    # Ensure the order of column in the test set is in the same order than in train set'''
    if str(action).lower()=='predict':
        # Load questionSlugAndType from JSON file
        with open('question_slug_and_type.json', 'r') as json_file:
            total_questions = json.load(json_file)

        missing_cols = set( total_questions) - set( data )

        #assigning null values to the empty columns.
        for c in missing_cols:
            scaled_df[c] = 0
            
        # scaled_df=scaled_df[list(data_order)]
        prediction_set=scaler.fit_transform(scaled_df)
        return(prediction_set)
    
    elif str(action).lower()=='train':

        data_order = questionSlugAndType

        missing_cols = set( data_order) - set( data )

        #assigning null values to the empty columns.
        for c in missing_cols:
            scaled_df[c] = 0

        print(f"After adding the missed columns{scaled_df}")

        # Save questionSlugAndType to a JSON file for prediction use
        with open('question_slug_and_type.json', 'w') as json_file:
            json.dump(questionSlugAndType, json_file)
        
        success_rate()
        x=scaled_df[list(data_order)]
        y=scaled_df["success_rate"]
        X_scaled = scaler.fit_transform(x)
        return(X_scaled,y)

# @app.route('/trainNeural', methods=['POST','GET'])
def trainNeural():
    action='train'

    # Example input data
    test = {
  "statusCode": 201,
  "message": "Model trained",
  "data": {
    "answers": [
      {
        "studentId": "666d4154932a2f10fa42d69b",
        "email": "flavie_kunde@nmail.edu",
        "quizId": "666d4157932a2f10fa42d716",
        "answers": {
          "alius-excepturi-suspendo": "false",
          "valeo-subseco-audentia": "true"
        }
      },
      {
        "studentId": "666d4154932a2f10fa42d6a1",
        "email": "mathilde.braun-schroeder@nmail.edu",
        "quizId": "666d4157932a2f10fa42d716",
        "answers": {
          "ambitus-pauci-synagoga": "true",
          "deludo-cresco-ipsam": "et"
        }
      },
      {
        "studentId": "666d4154932a2f10fa42d693",
        "email": "doris.kuhlman8@nmail.edu",
        "quizId": "666d4157932a2f10fa42d716",
        "answers": {
          "valeo-subseco-audentia": "canto"
        }
      },
      {
        "studentId": "666d4154932a2f10fa42d69d",
        "email": "caleb_zieme@nmail.edu",
        "quizId": "666d4157932a2f10fa42d716",
        "answers": {
          "alius-excepturi-suspendo": 1812502614638592,
          "voluptatum-tendo-magnam": 2964357889654784,
          "creta-damno-thalassinus": 3452212776796160
        }
      },
      {
        "studentId": "666d4153932a2f10fa42d691",
        "email": "ali.sauer29@nmail.edu",
        "quizId": "666d4157932a2f10fa42d716",
        "answers": {
          "ambitus-pauci-synagoga": "false"
        }
      },
      {
        "studentId": "666d4154932a2f10fa42d699",
        "email": "myrl_connelly@nmail.edu",
        "quizId": "666d4157932a2f10fa42d716",
        "answers": {
          "eos-ad-canis": 7881001527148544,
          "alius-excepturi-suspendo": 1178091521048576
        }
      },
      {
        "studentId": "666d4154932a2f10fa42d69f",
        "email": "helene.cummerata3@nmail.edu",
        "quizId": "666d4157932a2f10fa42d716",
        "answers": {
          "deludo-cresco-ipsam": "annus",
          "creta-damno-thalassinus": 1856917341208576,
          "voluptatum-tendo-magnam": 1669142425894912
        }
      },
      {
        "studentId": "666d4154932a2f10fa42d695",
        "email": "carmel.baumbach62@nmail.edu",
        "quizId": "666d4157932a2f10fa42d716",
        "answers": {
          "peior-debilito-cibo": "clarus"
        }
      },
      {
        "studentId": "666d4153932a2f10fa42d68c",
        "email": "dallas.schmitt@nmail.edu",
        "quizId": "666d4157932a2f10fa42d716",
        "answers": {
          "deludo-cresco-ipsam": "false",
          "sunt-commodo-aperio": 3457725730127872,
          "alius-excepturi-suspendo": 1866032362815488,
          "voluptatum-tendo-magnam": 4555981688143872,
          "valeo-subseco-audentia": "true"
        }
      }
    ],
    "modelName": "nn_2024-06-15_abstergo-dolor-molestias_F7IZuu",
    "algorithmType": "nn",
    "questionSlugAndType": [
      {
        "quizId": "666d4157932a2f10fa42d716",
        "questions": {
          "peior-debilito-cibo": {
            "type": "boolean",
            "priorityScore": 1
          },
          "valeo-subseco-audentia": {
            "type": "boolean",
            "priorityScore": 1
          },
          "deludo-cresco-ipsam": {
            "type": "boolean",
            "priorityScore": 10
          },
          "eos-ad-canis": {
            "type": "number",
            "priorityScore": 7
          },
          "ambitus-pauci-synagoga": {
            "type": "boolean",
            "priorityScore": 5
          },
          "commodo-urbanus-talus": {
            "type": "string",
            "priorityScore": 2
          },
          "creta-damno-thalassinus": {
            "type": "string",
            "priorityScore": 9
          },
          "aspicio-vomica-quaerat": {
            "type": "number",
            "priorityScore": 7
          },
          "voluptatum-tendo-magnam": {
            "type": "boolean",
            "priorityScore": 10
          },
          "sunt-commodo-aperio": {
            "type": "string",
            "priorityScore": 8
          },
          "alius-excepturi-suspendo": {
            "type": "number",
            "priorityScore": 7
          }
        }
      }
    ]
  }
}
    

    # Extract all answer values
    input = test['data']['answers']

    #Starting the timer
    start_time = time.time()

    # Creating a list to hold the transformed data
    global transformed_data
    transformed_data = []
    for student in input:
        student_data = {
            "studentId": student['studentId'],
            "email": student['email'],
            "quizId": student['quizId']
        }
        # Adding answers to the student data
        student_data.update(student['answers'])
        transformed_data.append(student_data)

    # Converting the list of dictionaries to a DataFrame
    data_df = pd.DataFrame(transformed_data)

    # Dropping the columns studentId, email, and quizId from the DataFrame
    answers_df = data_df.drop(columns=['studentId', 'email', 'quizId'])

    #getting questions keywords 
    questionsSlagAndType = test['data']['questionSlugAndType'][0]['questions']

    #getting the model name
    model_name = test['data']['modelName']

    print(answers_df)
    print(questionsSlagAndType)

    numeric_data = convert_to_numeric(answers_df)
    print(numeric_data)
    x,y=scale(action, numeric_data, questionsSlagAndType)

    print(x)
    print(y)

    #Neural model training starts
        
    traindata_x,testdata_x,traindata_y,testdata_y=train_test_split(x,y,test_size=0.2, random_state=42)
    print(traindata_y,testdata_y)
    model=Sequential([Dense(50, input_shape=(x.shape[1],),activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(1,activation='relu')])
    model.compile(optimizer='adam',loss='mean_squared_error' ,metrics=['mean_squared_error'])
    model.summary()
    model_fit=model.fit(x=traindata_x,y=traindata_y,epochs=50,batch_size=10, validation_split=0.2)
    model_validate=model.predict(testdata_x)
    mae=mean_absolute_error(testdata_y,model_validate)

    #edited........
    mse_metric = MeanSquaredError()
    mse_metric.update_state(testdata_y, model_validate)
    mse = mse_metric.result().numpy() 
    r2_sc=r2_score(testdata_y,model_validate)

    # Saving the model
    model.save(f'{model_name}.h5')

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
    'train_time': [total_time],
    'modelFilePath' :[file_path]
    }
    
    # Check environment variable
    model_env = os.getenv('modelenv', 'development')

    if model_env == 'production':
        # Return as a request for production
        return make_response(jsonify(output), 200)
    else:
        # Return JSON output for development
        return json.dumps(output)

# @app.route('/model/predict', methods=['POST'])
def predict_neural_model():
    action='predict'

    #starting the timer
    start_time = time.time()

    data = {
    "modelDetails": {
        "_id": "666e5423feceb4da8cc03f2b",
        "name": "rr_2024-06-16_abstergo-dolor-molestias_4JoPwb",
        "filePath": "/Users/sanjanavishwanath/Desktop/RCA Project/Summer ACOSUS/Summer_new_ACOSUS/nn_2024-06-15_abstergo-dolor-molestias_F7IZuu.h5",
        "absolute": 0.32596810976974666,
        "squared": 0.12619933090172708,
        "r2": 0.44392069708555937,
        "isCurrent": "true",
        "quizId": "666d4157932a2f10fa42d716",
        "modelUniqueId": "4JoPwb",
        "trainTime": 279,
        "trainData": "Vestigium decimus calamitas odio. Causa dolorum sed desino aliqua varietas somnus creo animus delectus. Territo unus acsi cohibeo curiositas velut talis conduco aeger crastinus.",
        "other": "Thorax cattus una apparatus audentia vindico triumphus.",
        "createdBy": "666d4155932a2f10fa42d6a9",
        "createdAt": "2024-06-16T02:55:31.325Z",
        "updatedAt": "2024-06-16T02:55:31.325Z",
        "__v": 0
    },
    "quizId": "666d4157932a2f10fa42d716",
    "allAnswers": {
        "alius-excepturi-suspendo": "false",
        "valeo-subseco-audentia": "true"
    },
    "studentId": "666d4154932a2f10fa42d69b"
    }

    # Extract the model path
    model_path = data['modelDetails']['filePath']

    # Extract the allAnswers dictionary
    all_answers = data['allAnswers']

    # Convert the allAnswers dictionary to a DataFrame
    df = pd.DataFrame([all_answers])

    # Extract only the questions (keys)
    questions = list(all_answers.keys())

    #convert all answers to numeric
    numeric_data = convert_to_numeric(df)

    #scaling the answers between 0 and 1
    scaled_data = scale(action, numeric_data, questions)

    # Load the model
    model = load_model(model_path)

    # Print the DataFrame
    print(scaled_data)

    # Predict the result
    predictions = model.predict(scaled_data)

    # End timing
    end_time = time.time()

    # Calculate the elapsed time
    total_time = end_time - start_time

    # Round and calculate the success rate
    predicted_successRate = round(predictions[0][0] * 100, 2)
    # predicted_successRate = round((predictions[0][0],2)*100).tolist()

    output = {
    'Success_Rate': predicted_successRate,
    'Predict_time': [total_time]
    }

    # Check environment variable
    model_env = os.getenv('modelenv', 'development')

    if model_env == 'production':
        # Return as a request for production
        return make_response(jsonify(output), 200)
    else:
        # we might want to send the make_response(jsonify(output), 200) and json.dumps(output)
        # Return JSON output for development
        return json.dumps(output)

@app.route('/trainNeural', methods=['POST'])
def train_neural_route():
    return trainNeural()

@app.route('/predict', methods=['POST'])
def predict_neural_route():
    return predict_neural_model()

if __name__ == '__main__':
    app.run(debug=True)


