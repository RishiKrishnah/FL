import flwr as fl
import torch

from flwr.client import NumPyClient
from .model import load_model
from .task import load_data, train, test, DEVICE
from .utils import poison_parameters

MALICIOUS_CLIENTS = {}


class FlowerClient(NumPyClient):
    def __init__(self, client_id, model_name):

        self.client_id = client_id
        print(f"Client {client_id} using model: {model_name}")
        self.model = load_model(model_name).to(DEVICE)
        (self.trainloader, self.valloader, self.testloader) = load_data(client_id)

    def get_parameters(self, config):

        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        if hasattr(self.model, "get_prototype"):
            prototype = self.model.get_prototype()

            if prototype is None:
                prototype = torch.zeros(
                    768,
                    dtype=torch.float32,
                )

            params.append(prototype.detach().cpu().numpy().astype("float32"))

        return params

    def set_parameters(
        self,
        parameters,
    ):

        num_model_params = len(self.model.state_dict())
        model_params = parameters[:num_model_params]

        params_dict = zip(
            self.model.state_dict().keys(),
            model_params,
        )

        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in params_dict}

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        # ----------------------------------
        # Extract global memory if present
        # ----------------------------------

        if len(parameters) == num_model_params + 1 and hasattr(
            self.model,
            "set_global_memory",
        ):
            memory = torch.tensor(
                parameters[-1],
                dtype=torch.float32,
                device=DEVICE,
            )

            self.model.set_global_memory(memory)

            print(
                f"Client {self.client_id} "
                f"received global memory "
                f"shape={memory.shape}, "
                f"norm={memory.norm().item():.4f}"
            )

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

            # Last element is the prototype
            weights = updated_params[:-1]
            prototype = updated_params[-1]

            # Poison only model weights
            weights = poison_parameters(weights)

            # Reassemble update
            updated_params = weights + [prototype]

            print(f"Client {self.client_id} poisoning complete")

        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        print(f"Client {self.client_id} returning results")

        metrics = {
            "client_id": self.client_id,
        }

        return (
            updated_params,
            len(self.trainloader.dataset),
            metrics,
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
