import flwr as fl
print("CLIENT FILE LOADED")
from pytorchexample.task import (
    DEVICE,
    Net,
    get_parameters,
    load_data,
    set_parameters,
    test,
    train,
)

NUM_CLIENTS = 2


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, partition_id: int):

        self.partition_id = partition_id

        self.net = Net().to(DEVICE)

        self.trainloader = load_data(
            partition_id,
            NUM_CLIENTS,
        )

    def get_parameters(self, config):

        print(f"[CLIENT {self.partition_id}] Sending parameters")

        return get_parameters(self.net)

    def fit(self, parameters, config):

        print(f"[CLIENT {self.partition_id}] Training started")

        set_parameters(self.net, parameters)

        train(self.net, self.trainloader)

        print(f"[CLIENT {self.partition_id}] Training completed")

        return (
            get_parameters(self.net),
            len(self.trainloader.dataset),
            {},
        )

    def evaluate(self, parameters, config):

        set_parameters(self.net, parameters)

        loss, accuracy = test(
            self.net,
            self.trainloader,
        )

        print(
            f"[CLIENT {self.partition_id}] Accuracy: {accuracy:.4f}"
        )

        return (
            float(loss),
            len(self.trainloader.dataset),
            {"accuracy": float(accuracy)},
        )


# CLIENT FACTORY
def client_fn(context):

    partition_id = context.node_config["partition-id"]

    return FlowerClient(partition_id).to_client()


# FLOWER CLIENT APP
app = fl.client.ClientApp(client_fn=client_fn)