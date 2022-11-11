import time

import torch
from happytransformer import HappyWordPrediction

# detect hardware gpu or cpu
def detect():
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")


# Load the model
def load_model():
    happy_wp = HappyWordPrediction("ALBERT”, “albert-xxlarge-v2")
    return happy_wp


# create a function to get user input and append it to the input buffer of the standard out
def get_input():
    user_input = input("")
    return user_input


def setup_model():
    wait_time = 2
    print("Downloading model...")
    happy_wp = load_model()
    print("Waiting for model to load...")
    # wait x seconds
    time.sleep(wait_time)
    print("Model loaded!")
    return happy_wp


def predict(user_input, predictor):
    result = predictor.predict_mask(user_input + "[MASK]", top_k=20)
    return result


# main function
def main():
    detect()
    predictor = setup_model()

    print("Enter sentence with a blank space where you want to predict the next word: ")
    user_input = get_input()

    while True:
        new_input = predict(user_input, predictor)
        print(user_input)
        user_input = user_input + new_input[5].token
        next_words = get_input()
        user_input = user_input + " " + next_words

        # if user presses q, exit the program
        if next_words == "q":
            print("Exiting program...")
            break


# call main function
if __name__ == "__main__":
    main()
