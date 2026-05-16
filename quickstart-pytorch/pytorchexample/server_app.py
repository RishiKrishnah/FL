import flwr as fl

print("SERVER FILE LOADED")

NUM_CLIENTS = 2


# METRIC AGGREGATION
def weighted_average(metrics):

    accuracies = [
        num_examples * m["accuracy"]
        for num_examples, m in metrics
    ]

    examples = [
        num_examples
        for num_examples, _ in metrics
    ]

    return {
        "accuracy": sum(accuracies) / sum(examples)
    }


# SERVER FUNCTION
def server_fn(context):

    print("[SERVER] Creating strategy")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = fl.server.ServerConfig(
        num_rounds=5,
    )

    return fl.server.ServerAppComponents(
        strategy=strategy,
        server_config=config,
    )


# MODERN FLOWER SERVER APP
app = fl.server.ServerApp(server_fn=server_fn)