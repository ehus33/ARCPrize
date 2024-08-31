import json
import numpy as np
from model import build_cnn
from evolutionary_algorithm import evolve_population
from data_loader import load_data

def main():
    train_data, train_labels, val_data, val_labels, test_data = load_data()
    cnn_model = build_cnn(input_shape=(64, 64, 1))
    cnn_model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
    
    submission = evolve_population(test_data, cnn_model)
    
    with open("submission.json", "w") as f:
        json.dump(submission, f)

if __name__ == "__main__":
    main()

