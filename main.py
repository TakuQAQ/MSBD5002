from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import model_from_json
import time
import pandas as pd
import numpy as np

# Load, transform and split the original dataset
def load_trans(filename):
    # Load the dataset for trainning
    df = pd.read_csv(filename, delimiter="\t", header=None)
    # Split the dataset into input and target

    # Transform the non-numeric attribute into numeric value
    # 9: Age (Range), 10: Gender, 11: Country
    age = sorted(df[9].unique())
    gender = sorted(df[10].unique())
    country = sorted(df[11].unique())
    # Build mapping from categorical value to numerical value
    age_map = {age[i]: i for i in range(len(age))}
    gender_map = {gender[i]: i for i in range(len(gender))}
    country_map = {country[i]: i for i in range(len(country))}
    # Label encoding
    df[9] = df[9].map(age_map)
    df[10] = df[10].map(gender_map)
    df[11] = df[11].map(country_map)
    
    return df

def split(df):
    X, y = df.loc[:,:11], df.loc[:,[12]]
    return X, y



# Train and save models
# Input hidden_layer_param is a list of tuples (number of neurons, activation function)
# Similarly output_layer_param is a tuple
def train_model(X, y,
                hidden_layer_param, output_layer_param,
                loss_func, opt_method,
                val_splt, epoch, batch):
    np.random.seed(int(time.time()))
    
    # Construct the model
    model = Sequential()
    # Implied input layer
    model.add(Input(shape=(len(X.columns),)))
    # Hidden layers
    for n,f in hidden_layer_param:
        model.add(Dense(n, activation=f))
    # Output layer
    n0, f0 = output_layer_param
    model.add(Dense(n0, activation=f0))
    
    # Compile the model
    model.compile(loss=loss_func, optimizer=opt_method, metrics=["accuracy"])
    
    # Fit the model
    model.fit(X, y, validation_split=val_splt, epochs=epoch, batch_size=batch)
    
    # Evaluation
    scores = model.evaluate(X, y)
    print("")
    print(f"{model.metrics_names[0]}: {scores[0]}, {model.metrics_names[1]}: {round(scores[1]*100,2)}%")
    
    return model

def save_model(model, model_name):
    # Save the model structure to a file in the JSON format
    structure_file = model_name + ".json"
    model_json = model.to_json()
    with open(structure_file, "w") as f:
        f.write(model_json)

    # Save the model weight information to a file in the HDF5 format
    weight_file = model_name + ".weights.h5"
    model.save_weights(weight_file)

def save_all_models(training_file, model_param):
    # Load, transform and split dataset for training
    X, y = split(load_trans(training_file))
    
    i = 1
    for hidden_layer_param, output_layer_param, loss_func, opt_method, val_splt, epoch, batch in model_param:
        print(f"Training model {i}...")
        model = train_model(X, y,
                            hidden_layer_param, output_layer_param,
                            loss_func, opt_method,
                            val_splt, epoch, batch)

        print(f"Saving model {i}...")
        save_model(model, f"ModelNN_{i}")
        print(f"Model {i} saved")
        print("")
        i += 1
    
    print("All models saved!")
    print("------------------------------")



# Read models and save prediction
def read_model(model_name):
    # Load the model structure from a file in the JSON format
    structure_file = model_name + ".json"
    with open(structure_file, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    
    # Load the model weight information from a file in the HDF5 format
    weight_file = model_name + ".weights.h5"
    model.load_weights(weight_file)
    
    return model

def predict_from_model(newX, new_target_file, model, threshold):
    # Predict the target attribute of the new data based on a model
    y_pred = np.where(model.predict(newX) > threshold, 1, 0)

    # Save the predicted target attribute of the new data into a file
    np.savetxt(new_target_file, y_pred, delimiter="\t", fmt="%d")
    
    return y_pred

def save_all_prediction(new_input_file, threshold, n):
    # Load the new data (input attribute)
    print(f"Loading input attributes dataset...")
    print("")
    newX = load_trans(new_input_file)
    
    for i in range(1,n+1):
        print(f"Loading model {i}...")
        model = read_model(f"ModelNN_{i}")
        
        print(f"Predicting from model {i}...")
        predict_from_model(newX, f"predicted{i}.txt", model, threshold)
        print(f"Prediction from model {i} saved")
        print("")
        
    print("All prediction saved!")



# The main function
def main():
    training_file = "firstData.txt"
    new_input_file = "secondData.txt"
    threshold = 0.5
    
    # Define parameters for 5 different NN models, which are supposed to be a list of tuples
    # The ith entry corresponds to (hidden_layer_param, output_layer_param, loss_func, opt_method, val_splt, epoch, batch)
    model_param = [([(12,"relu"), (8,"relu")], (1,"sigmoid"), "binary_crossentropy", "adam", 0.2, 50, 10),
                   ([(4,"relu"), (2,"relu")], (1,"sigmoid"), "binary_crossentropy", "adam", 0.2, 30, 30),
                   ([(4,"relu"), (2,"relu")], (1,"sigmoid"), "binary_crossentropy", "SGD", 0.2, 30, 30),
                   ([(4,"sigmoid"), (2,"sigmoid")], (1,"sigmoid"), "binary_crossentropy", "adam", 0.2, 30, 30),
                  ([(8,"relu")], (1,"sigmoid"), "binary_crossentropy", "adam", 0.2, 50, 10)]
    
    # Train and save models
    save_all_models(training_file, model_param)
    
    # Read models and save prediction from models
    n = len(model_param)
    save_all_prediction(new_input_file, threshold, n)

main()