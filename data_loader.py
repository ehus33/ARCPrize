import numpy as np

def load_data():
    train_data = np.random.rand(1000, 64, 64, 1)  # Example shape
    train_labels = np.random.randint(0, 10, 1000)
    val_data = np.random.rand(200, 64, 64, 1)
    val_labels = np.random.randint(0, 10, 200)
    test_data = {
        "task_1": [np.random.rand(64, 64, 1)],
        "task_2": [np.random.rand(64, 64, 1)]
    }
    return train_data, train_labels, val_data, val_labels, test_data
