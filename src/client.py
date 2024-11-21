import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Client:
    def __init__(self, id, speed_factor, dataset, batch_size=32):
        """
        Initialize a Client object.

        Parameters:
        - id (int): The ID of the client.
        - speed_factor (float): The speed factor of the client, which determines the training delay.
        - dataset (torch.utils.data.Dataset): The dataset used for training.
        - batch_size (int, optional): The batch size for the dataloader. Default is 32.
        """
        self.id = id
        self.speed_factor = speed_factor
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, global_model, epochs=1):
        """
        Train the global model on the client's local dataset using Adam optimizer.

        Parameters:
        - global_model (torch.nn.Module): The global model to be trained.
        - epochs (int, optional): The number of training epochs. Default is 1.

        Returns:
        - state_dict (dict): The updated model parameters.
        """
        # Directly copy the global model
        local_model = global_model
        local_model.load_state_dict(global_model.state_dict())
        
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            local_model.parameters(), 
            lr=0.001, 
            # learning hyperparameters can be set later
            # betas=(0.9, 0.99), 
            # eps=1e-7, 
            # weight_decay=1e-4
        )
        
        # Simulate training delay based on speed_factor
        local_model.train()
        for epoch in range(int(epochs * self.speed_factor)):
            for inputs, labels in self.dataloader:
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Return updated model parameters
        return local_model.state_dict()
    