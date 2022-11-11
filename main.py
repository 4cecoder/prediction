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


def predict(user_input, predictor, max_predictions):
    result = predictor.predict_mask(user_input + " [MASK]", top_k=max_predictions)
    return result


def check_exit(user_input):
    if user_input == "q":
        print("Exiting program...")
        exit()


# main function
def main():
    prediction_variant = 0
    max_predictions = 20
    detect()
    predictor = setup_model()

    print("Enter sentence with a blank space where you want to predict the next word: ")
    user_input = get_input()
    check_exit(user_input)

    while True:
        new_input = predict(user_input, predictor, max_predictions)
        # if new_input[].token is not greater than 1 character long, then add 1 to predictionVariant
        if len(new_input[prediction_variant].token) == 1 and prediction_variant < max_predictions:
            prediction_variant += 1
        if len(new_input[prediction_variant].token) > 1:
            user_input = user_input + " " + new_input[prediction_variant].token
            print(user_input)
            next_words = get_input()
            user_input = user_input + " " + next_words
            check_exit(next_words)


# call main function
if __name__ == "__main__":
    main()
