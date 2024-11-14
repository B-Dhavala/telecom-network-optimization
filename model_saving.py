#model_saving.py
import pickle

# Function to save models
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Function to load models
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model
