from collections import OrderedDict
import numpy as np
import torch

from flwr.common import Context, parameters_to_ndarrays

from flwr.server import (
    ServerApp,
    ServerAppComponents,
    ServerConfig,
)

from flwr.server.strategy import FedAvg

from .model import load_model
from .task import save_model
from flwr.common import ndarrays_to_parameters


def weighted_average(metrics):

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


def fedavg_weights(weight_sets, num_examples):

    total_examples = sum(num_examples)

    aggregated = []

    for layer_idx in range(len(weight_sets[0])):
        layer_sum = sum(
            weights[layer_idx] * n for weights, n in zip(weight_sets, num_examples)
        )

        aggregated.append(layer_sum / total_examples)

    return aggregated


class SaveModelStrategy(FedAvg):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)

        self.model_name = model_name

    def aggregate_fit(self, server_round, results, failures):
        print(
            f"\nRound {server_round}: results={len(results)}, failures={len(failures)}"
        )

        aggregated = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated is None:
            print(f"Round {server_round}: aggregation returned None")
            return None

        parameters, metrics = aggregated

        if parameters is None:
            print(f"Round {server_round}: parameters are None")
            return aggregated

        ndarrays = parameters_to_ndarrays(parameters)

        model = load_model(self.model_name)

        model_ndarrays = ndarrays[: len(model.state_dict())]

        params_dict = zip(
            model.state_dict().keys(),
            model_ndarrays,
        )

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        model.load_state_dict(state_dict, strict=True)

        save_model(model, server_round)

        print(f"Round {server_round}: model saved successfully")

        return aggregated


def server_fn(context: Context):

    print("\n===== RUN CONFIG =====")
    print(context.run_config)
    print("======================\n")

    local_epochs = context.run_config["local-epochs"]
    model_name = context.run_config["model-name"]
    num_rounds = context.run_config["num-server-rounds"]

    def fit_config(server_round):

        config = {"local_epochs": local_epochs}

        return config

    if model_name == "faft":
        strategy = FAFTStrategy(
            model_name=model_name,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=5,
            min_evaluate_clients=5,
            min_available_clients=5,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    else:
        strategy = SaveModelStrategy(
            model_name=model_name,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=5,
            min_evaluate_clients=5,
            min_available_clients=5,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


class FAFTStrategy(SaveModelStrategy):
    def __init__(self, model_name, **kwargs):

        super().__init__(
            model_name=model_name,
            **kwargs,
        )

        self.global_memory = None

        model = load_model(model_name)
        self.num_model_tensors = len(model.state_dict())
        del model

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        print(
            f"\nFAFT Round {server_round}: "
            f"results={len(results)}, "
            f"failures={len(failures)}"
        )

        if failures:
            print("\n========== FAILURES ==========")

            for i, failure in enumerate(failures):
                print(f"\nFailure {i + 1}:")
                print(type(failure))
                print(repr(failure))

            print("\n==============================")

        if not results:
            print("No client results received")
            return None

        # --------------------------------------------------
        # Separate model weights and prototypes
        # --------------------------------------------------

        expected_tensors = self.num_model_tensors + 1

        weight_sets = []
        prototypes = []
        example_counts = []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)

            if len(ndarrays) != expected_tensors:
                raise ValueError(
                    f"Expected {expected_tensors} tensors but received {len(ndarrays)}"
                )

            model_weights = ndarrays[:-1]

            prototype = ndarrays[-1]

            weight_sets.append(model_weights)

            prototypes.append(prototype)

            example_counts.append(fit_res.num_examples)
        # --------------------------------------------------
        # FedAvg for model weights
        # --------------------------------------------------
        global_weights = fedavg_weights(
            weight_sets,
            example_counts,
        )

        # --------------------------------------------------
        # Prototype aggregation
        # --------------------------------------------------
        global_proto = np.average(
            prototypes,
            axis=0,
            weights=example_counts,
        )

        # --------------------------------------------------
        # Update global memory
        # --------------------------------------------------
        if self.global_memory is None:
            self.global_memory = global_proto

            print("Initialized global memory")

        else:
            self.global_memory = 0.9 * self.global_memory + 0.1 * global_proto

            print("Updated global memory")

        print(f"\nRound {server_round} Prototype Aggregated")

        print(f"Memory Shape: {self.global_memory.shape}")

        print(f"Memory Norm: {np.linalg.norm(self.global_memory):.4f}")

        # --------------------------------------------------
        # Save aggregated global model
        # --------------------------------------------------
        model = load_model(self.model_name)

        params_dict = zip(
            model.state_dict().keys(),
            global_weights,
        )

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        save_model(
            model,
            server_round,
        )

        print(f"Round {server_round}: model saved successfully")

        # Broadcast:
        #   global model weights
        #   global memory vector
        combined_parameters = global_weights + [self.global_memory.astype(np.float32)]
        print(
            f"Broadcasting "
            f"{len(combined_parameters)} tensors "
            f"({len(global_weights)} weights + memory)"
        )
        parameters = ndarrays_to_parameters(combined_parameters)

        return (
            parameters,
            {"memory_norm": float(np.linalg.norm(self.global_memory))},
        )


app = ServerApp(server_fn=server_fn)
