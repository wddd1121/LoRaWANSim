import numpy as np
import matplotlib.pyplot as plt
import math




def friis_pathloss(d):
    if d == 0:
        return 1
    # pathloss = np.power((299792.458) / (4 * np.pi * d * 470000000), 2)
    pathloss = np.square((299.792458) / (4 * np.pi * d * 470000))
    if pathloss > 1.0 :
        pathloss = 1.0
    return pathloss


def pathloss_calculate(d):
    return friis_pathloss(d)

