import torch




# Can this be simplified to only return a dictionary that is the average of client states?
def federated_averaging(global_model, client_states):
    # Initialize a dictionary to store averaged weights
    averaged_state = global_model.state_dict()
    
    for key in averaged_state:
        # Compute the average of the weights for each layer
        averaged_state[key] = torch.mean(
            torch.stack([client_state[key] for client_state in client_states]), dim=0
        )
    
    return averaged_state
