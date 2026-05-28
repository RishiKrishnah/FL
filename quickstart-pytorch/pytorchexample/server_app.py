from collections import OrderedDict

import flwr as fl
import torch

from flwr.common import (

    Context,

    parameters_to_ndarrays
)

from flwr.server import (

    ServerApp,

    ServerAppComponents
)

from flwr.server.strategy import (
    FedAvg
)

from .model import load_model

from .task import save_model


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

        "accuracy":

        sum(accuracies)

        / sum(examples)
    }


class SaveModelStrategy(FedAvg):

    def aggregate_fit(

        self,

        server_round,

        results,

        failures
    ):

        aggregated = super().aggregate_fit(

            server_round,

            results,

            failures
        )

        if aggregated is not None:

            parameters, _ = aggregated

            ndarrays = (

                parameters_to_ndarrays(
                    parameters
                )
            )

            model = load_model()

            params_dict = zip(

                model.state_dict().keys(),

                ndarrays
            )

            state_dict = OrderedDict({

                k: torch.tensor(v)

                for k, v in params_dict
            })

            model.load_state_dict(

                state_dict,

                strict=True
            )

            save_model(

                model,

                server_round
            )

        return aggregated


def server_fn(context: Context):

    strategy = SaveModelStrategy(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=2,

        min_evaluate_clients=2,

        min_available_clients=2,

        evaluate_metrics_aggregation_fn=
        weighted_average
    )

    return ServerAppComponents(

        strategy=strategy
    )


app = ServerApp(

    server_fn=server_fn
)