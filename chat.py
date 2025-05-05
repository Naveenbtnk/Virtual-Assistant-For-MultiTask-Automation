import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents JSON
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Apple"


# Adjust input size function
def adjust_input_size(X, expected_size):
    """Adjust input size by truncating or padding the bag-of-words vector."""
    if X.shape[1] > expected_size:
        return X[:, :expected_size]  # Truncate
    elif X.shape[1] < expected_size:
        padding = np.zeros((1, expected_size - X.shape[1]))
        return np.concatenate((X, padding), axis=1)  # Pad
    return X


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])

    # Debugging: Check input size
    print(f"Bag-of-words size: {X.shape[1]}, Expected size: {input_size}")

    # Adjust input size if needed
    if X.shape[1] != input_size:
        X = adjust_input_size(X, input_size)

    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Debugging: Print predicted tag and confidence
    print(f"Predicted tag: {tag}, Confidence: {prob.item()}")

    if prob.item() > 0.5:  # Lowered threshold
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."


# Main loop
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        resp = get_response(sentence)
        print(f"{bot_name}: {resp}")
