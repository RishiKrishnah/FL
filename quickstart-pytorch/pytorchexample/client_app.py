import flwr as fl
import torch

from flwr.client import NumPyClient

from .model import load_model

from .task import (
    load_data,
    train,
    test,
    DEVICE
)

from .utils import poison_parameters


class FlowerClient(NumPyClient):

    def __init__(self, client_id):

        self.client_id = client_id

        self.model = load_model().to(DEVICE)

        self.trainloader, self.testloader = (
            load_data(client_id)
        )

    def get_parameters(self, config):

        return [

            val.cpu().numpy()

            for _, val in
            self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):

        params_dict = zip(
            self.model.state_dict().keys(),
            parameters
        )

        state_dict = {

            k: torch.tensor(v).to(DEVICE)

            for k, v in params_dict
        }

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

    def fit(self, parameters, config):

        print(
            f"Client {self.client_id} "
            f"training..."
        )

        self.set_parameters(parameters)

        train(
            self.model,
            self.trainloader,
            epochs=1
        )

        updated_params = self.get_parameters({})

        # Simulate malicious client
        if self.client_id == 2:

            print(
                "Client 2 is malicious!"
            )

            updated_params = (
                poison_parameters(
                    updated_params
                )
            )

        return (

            updated_params,

            len(self.trainloader.dataset),

            {
                "client_id":
                self.client_id
            }
        )

    def evaluate(
        self,
        parameters,
        config
    ):

        self.set_parameters(parameters)

        loss, accuracy = test(
            self.model,
            self.testloader
        )

        return (

            float(loss),

            len(self.testloader.dataset),

            {
                "accuracy":
                float(accuracy)
            }
        )


def client_fn(context):

    client_id = (
        context.node_config["partition-id"]
        + 1
    )

    return FlowerClient(
        client_id
    ).to_client()


app = fl.client.ClientApp(
    client_fn=client_fn
)