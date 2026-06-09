import flwr as fl
import torch

from flwr.client import NumPyClient

from .model import load_model

from .task import load_data, train, test, DEVICE

MALICIOUS_CLIENTS = {}

from .utils import poison_parameters


class FlowerClient(NumPyClient):
    def __init__(self, client_id, model_name):

        self.client_id = client_id

        print(f"Client {client_id} using model: {model_name}")

        self.model = load_model(model_name).to(DEVICE)

        (self.trainloader, self.valloader, self.testloader) = load_data(client_id)

    def get_parameters(self, config):

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in params_dict}

        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        print(f"Client {self.client_id} starting fit...")

        self.set_parameters(parameters)

        print(f"Client {self.client_id} starting training...")

        local_epochs = config["local_epochs"]

        train(self.model, self.trainloader, epochs=local_epochs)

        print(f"Client {self.client_id} finished training")

        print(f"Client {self.client_id} extracting parameters...")

        updated_params = self.get_parameters({})

        print(f"Client {self.client_id} parameter extraction complete")

        if self.client_id in MALICIOUS_CLIENTS:
            print(f"Client {self.client_id} is malicious!")

            updated_params = poison_parameters(updated_params)

            print(f"Client {self.client_id} poisoning complete")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Client {self.client_id} returning results")

        return (
            updated_params,
            len(self.trainloader.dataset),
            {"client_id": self.client_id},
        )

    def evaluate(self, parameters, config):

        try:
            print(f"\nCLIENT {self.client_id} STARTING EVALUATION")

            self.set_parameters(parameters)

            loss, accuracy = test(self.model, self.testloader)

            print(f"\nCLIENT {self.client_id} EVALUATION COMPLETE")

            return (
                float(loss),
                len(self.testloader.dataset),
                {"accuracy": float(accuracy)},
            )

        except Exception as e:
            print(f"\nCLIENT {self.client_id} EVALUATION ERROR:")

            print(e)

            raise


def client_fn(context):

    client_id = context.node_config["partition-id"] + 1

    model_name = context.run_config["model-name"]

    return FlowerClient(client_id, model_name).to_client()


app = fl.client.ClientApp(client_fn=client_fn)
