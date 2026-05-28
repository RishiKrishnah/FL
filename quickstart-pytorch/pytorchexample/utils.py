import numpy as np


def poison_parameters(parameters):

    poisoned = []

    for param in parameters:

        noise = np.random.normal(

            0,

            0.5,

            param.shape
        )

        poisoned.append(
            param + noise
        )

    return poisoned