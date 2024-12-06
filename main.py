from json.decoder import JSONDecoder
import random
from flask import Flask, request
from flask_cors import CORS
import json
import os
from flask import Flask, request, make_response, jsonify
import pandas as pandas
from dotenv import load_dotenv
import openai
import regression
import neural_network

# Load the environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

useHardCodedData = os.getenv('USE_HARDCODED_DATA', default=True)
current_model = None

def genai_sucess_rate(prompt):
    print(f"generating text based output using GENAI..GPT 3.5")
    openai.api_key = os.getenv('OPENAI_API_KEY')
    # client = OpenAI(os.getenv('OPENAI_API_KEY'))
    print(f"Prompt: {prompt}")

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # print(completion.choices[0].message)
    print(completion.choices[0].message.content)
    return make_response(jsonify({"message": completion.choices[0].message.content}), 200)


@app.route('/genai', methods=['POST'])
def genai_sucess_rate_route():
    data = request.json
    print(data)
    return genai_sucess_rate(data['prompt'])


def generate_seed_data(data):
    seed_qty = int(data["seedQty"])
    questionSlugAndType = data["questionSlugAndType"][0]
    questions = questionSlugAndType["questions"]
    answers_list = []
    for _ in range(seed_qty):
        answers = {}
        for question, details in questions.items():
            if "seedData" not in details or not details["seedData"]:
                continue
            seed_data = details["seedData"]
            seed_values = [item["seedValue"] for item in seed_data]
            seed_probabilities = [float(item["seedProbability"]) for item in seed_data]
            selected_value = random.choices(seed_values, seed_probabilities)[0]
            answers[question] = selected_value
        answers_list.append({
            "answers": answers
        })
    return {
        "data": {
            "answers": answers_list
        }
    }

def train_with_seed_data(data):
    # Generate seed data
    seed_data = generate_seed_data(data['data'])

    # Update the original data with generated seed data
    data['data']['answers'] = seed_data['data']['answers']

    model_name = data['data']['modelName']
    model_dir = os.path.join('models', model_name)
    os.makedirs(model_dir, exist_ok=True)
    training_data_path = os.path.join(model_dir, 'SeededTrainingData_350.json')
    with open(training_data_path, 'w') as f:
        json.dump(data, f, indent=2)
    if data['data']['algorithmType'] == 'nn':
        return neural_network.prepare_training_data(data, model_dir)
    else:
        return regression.prepare_training_data(data, model_dir)

@app.route('/seed', methods=['POST'])
def trainSeededModel_with_options_route():
    print("Training the model when options are given..")
    data = request.json if not useHardCodedData else None
    if useHardCodedData == 'True':
       file_path = 'models/input_test_files/SeededTrainingData_new.json'
       # Reading the JSON file
       with open(file_path, 'r') as file:
        data = json.load(file)

    else:
        data = request.json
        print(data)

    return train_with_seed_data(data)

@app.route('/trainNeural', methods=['POST'])
def train_neural_route_from_request():
    print("Training the model")
    print(request.json)
    data = request.json
    model_name = data['data']['modelName']
    model_dir = os.path.join('models', model_name)
    return neural_network.prepare_training_data(data, model_dir)

@app.route('/trainRegression', methods=['POST'])
def train_regression_route_from_request():
    print("Training the model")
    print(request.json)
    data = request.json
    model_name = data['data']['modelName']
    model_dir = os.path.join('models', model_name)
    return regression.prepare_training_data(data, model_dir)

@app.route('/predict_neural', methods=['POST'])
def predict_neural_route():
    print("Predicting the Neural Network model....\n")
    return neural_network.predict_neural_model()

@app.route('/predict_regression', methods = ['POST'])
def predict_regression_route():
    print("Predicting the Regression model..\n")
    return regression.predict_regression_model()

if __name__ == '__main__':
    print(trainSeededModel_with_options_route())
    # model_env = os.getenv('MODELENV', 'local')
    # host = os.getenv('FLASK_HOST', '127.0.0.1')
    # port = int(os.getenv('FLASK_PORT', 5000))
    # debug = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    # print(f"Model environment: {model_env}")
    # app.run(host=host, port=port, debug=debug)
