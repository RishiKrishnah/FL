import numpy as np

def poison_parameters(parameters):

    print("\nPOISONING STARTED")

    for i, param in enumerate(parameters):

        before = np.mean(param)

        param += np.random.normal(
            0,
            0.05,
            param.shape
        ).astype(param.dtype)

        after = np.mean(param)

        print(
            f"Layer {i}: "
            f"{before:.6f} -> {after:.6f}"
        )

    print("POISONING FINISHED\n")

    return parameters