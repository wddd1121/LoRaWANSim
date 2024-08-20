import numpy as np
import matplotlib.pyplot as plt


def generateENCoordinates(radius, EN_num, rng):
    u = rng.uniform(0, 1, EN_num)
    R = radius * np.sqrt(u)
    theta = 2 * np.pi * rng.uniform(0, 1, EN_num)
    xs = R * np.cos(theta)
    ys = R * np.sin(theta)
    return xs, ys


