import torch
import torch.nn as nn
import random
import numpy as np
from quantization import scalar_quantize

# Can this be simplified to only return a dictionary that is the average of client states?
def federated_averaging(global_model, client_states, setup="standard"):
    
    # Initialize a dictionary to store averaged weights
    averaged_state = global_model.state_dict()

    for key in averaged_state:
        if any(substr in key for substr in ["activation_post_process", "fake_quant"]):
            continue

        # Compute the average of the weights for each layer
        try:
            averaged_state[key] = torch.mean(torch.stack([client_state[key] for client_state in client_states]), dim=0)

        except KeyError:
            print(f"Warning: Key {key} not found in one or more client states. Skipping.")
            continue
    
    return averaged_state


def client_selection_with_constraints(client_resources, deadline):
    """
    Select clients based on their resource availability and time constraints.
    """
    selected_clients = []
    total_time = 0  # Track elapsed time
    remaining_clients = list(range(len(client_resources)))  # Indices of available clients

    while remaining_clients:
        # Sort remaining clients by minimum time to complete training and upload
        best_client = None
        min_time = float('inf')

        for client in remaining_clients:
            resource = client_resources[client]
            update_time = resource["data_size"] / resource["comp_capacity"]  # Simplified time calculation
            if total_time + update_time < deadline and update_time < min_time:
                best_client = client
                min_time = update_time

        if best_client is None:
            break  # No more clients can be selected within the deadline

        # Select the best client
        selected_clients.append(best_client)
        total_time += min_time
        remaining_clients.remove(best_client)

    return selected_clients


def select_indices(n, k):
    return random.sample(range(n), k)
