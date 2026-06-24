import flwr as fl
import torch

from flwr.client import NumPyClient
from .model import load_model
from .task import load_data, train, test
from .utils import poison_parameters

MALICIOUS_CLIENTS = {}


class FlowerClient(NumPyClient):
    def __init__(
        self,
        client_id,
        model_name,
        dataset_name,
    ):

        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Client {client_id} using {self.device}")

        self.model = load_model(model_name).to(self.device)

        (
            self.trainloader,
            self.valloader,
            self.testloader,
        ) = load_data(
            client_id,
            dataset_name,
        )

    def get_parameters(self, config):

        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        if hasattr(self.model, "get_prototypes"):
            prototypes = self.model.get_prototypes()

            real_proto = prototypes["real"]
            fake_proto = prototypes["fake"]

            if real_proto is None:
                real_proto = torch.zeros(
                    768,
                    dtype=torch.float32,
                )

            if fake_proto is None:
                fake_proto = torch.zeros(
                    768,
                    dtype=torch.float32,
                )

            params.append(real_proto.detach().cpu().numpy().astype("float32"))
            params.append(fake_proto.detach().cpu().numpy().astype("float32"))

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

        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        # ----------------------------------
        # Extract global memory if present
        # ----------------------------------

        if len(parameters) == num_model_params + 2 and hasattr(
            self.model, "set_global_memory"
        ):
            real_memory = torch.tensor(
                parameters[-2],
                dtype=torch.float32,
                device=self.device,
            )

            fake_memory = torch.tensor(
                parameters[-1],
                dtype=torch.float32,
                device=self.device,
            )

            self.model.set_global_memory(
                real_memory,
                fake_memory,
            )

            print(f"Client {self.client_id} received class memories")
            print(f"Real memory norm={real_memory.norm().item():.4f}")
            print(f"Fake memory norm={fake_memory.norm().item():.4f}")

    def fit(self, parameters, config):

        print(f"Client {self.client_id} starting fit...")
        self.set_parameters(parameters)

        print(f"Client {self.client_id} starting training...")
        local_epochs = config["local_epochs"]

        train(
            self.model,
            self.trainloader,
            self.device,
            epochs=local_epochs,
        )
        print(f"Client {self.client_id} finished training")
        print(f"Client {self.client_id} extracting parameters...")
        updated_params = self.get_parameters({})

        print(f"Client {self.client_id} parameter extraction complete")

        if self.client_id in MALICIOUS_CLIENTS:
            print(f"Client {self.client_id} is malicious!")

            if hasattr(self.model, "get_prototypes") and len(updated_params) >= 2:
                weights = updated_params[:-2]
                real_proto = updated_params[-2]
                fake_proto = updated_params[-1]
                weights = poison_parameters(weights)
                updated_params = weights + [real_proto] + [fake_proto]

            else:
                updated_params = poison_parameters(updated_params)

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
            loss, accuracy = test(
                self.model,
                self.testloader,
                self.device,
            )
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

    dataset_name = context.run_config["dataset-name"]

    return FlowerClient(
        client_id,
        model_name,
        dataset_name,
    ).to_client()


app = fl.client.ClientApp(client_fn=client_fn)
