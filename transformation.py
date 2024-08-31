import numpy as np

def apply_transformations(grid, transformations):
    for transformation in transformations:
        if transformation == 1:
            grid = np.rot90(grid)  # Example transformation: rotate 90 degrees
        elif transformation == 0:
            grid = np.flip(grid, axis=1)  # Example transformation: flip horizontally
    return grid
