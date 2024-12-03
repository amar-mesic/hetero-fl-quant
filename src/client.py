import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from dataclasses import dataclass, field
import random
import fl



@dataclass
class ClientResources:
    speed_factor: float
    battery_level: float
    bandwidth: float
    dataset_size: int
    CPU_available: bool
    CPU_memory_availability: float
    GPU_available: bool
    GPU_memory_availability: float

    def __post_init__(self):
        # Validation logic
        if not (1 <= self.speed_factor):
            raise ValueError("speed_factor must be greater than or equal to 1")
        if not (1 <= self.battery_level <= 100):
            raise ValueError("battery_level must be between 0 and 100")
        if self.bandwidth < 0:
            raise ValueError("bandwidth must be non-negative")
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if not (0 <= self.CPU_memory_availability <= 128):  # Example: assuming max 128GB
            raise ValueError("CPU_memory_availability must be between 0 and 128")
        if not (0 <= self.GPU_memory_availability <= 32):  # Example: assuming max 32GB
            raise ValueError("GPU_memory_availability must be between 0 and 32")



    
    @staticmethod
    def generate_random(dataset_size_range=(500, 2000)):
        """
        Generate random valid ClientResources.
        """

        GPU_available = random.choice([True, False])

        return ClientResources(
            speed_factor=random.uniform(1.0, 2.0),  # Speed factor in range [0.1, 2.0]
            battery_level=random.uniform(1, 100),   # Battery level in range [0, 100]
            bandwidth=random.uniform(1, 100),       # Bandwidth in range [1, 100] Mbps
            dataset_size=random.randint(*dataset_size_range),  # Dataset size
            CPU_available=random.choice([True, False]),        # Random CPU availability
            CPU_memory_availability=random.uniform(0, 128),    # CPU memory in GB
            GPU_available=GPU_available,        # Random GPU availability
            GPU_memory_availability=random.uniform(0, 32) if GPU_available else 0,  # GPU memory if available
        )








class Client:
    def __init__(self, id, resources: ClientResources, dataset, dataloader, val_loader):
        """
        Initialize a Client object.

        Parameters:
        - id (int): The ID of the client.
        - speed_factor (float): The speed factor of the client, which determines the training delay. It must be greater than 1
        - dataset (torch.utils.data.Dataset): The dataset used for training.
        - batch_size (int, optional): The batch size for the dataloader. Default is 32.
        """

        self.id = id
        self.resources = resources

        self.dataset = dataset
        self.dataloader = dataloader
        self.val_loader = val_loader

    def train(self, global_model, epochs=1):
        """
        Train the global model on the client's local dataset using Adam optimizer.

        Parameters:
        - global_model (torch.nn.Module): The global model to be trained.
        - epochs (int, optional): The number of training epochs. Default is 1.

        Returns:
        - state_dict (dict): The updated model parameters.
        """
        # track start time
        start_time = time.time()

        # Directly copy the global model
        local_model = fl.create_model()
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
        for epoch in range(epochs):
            for inputs, labels in self.dataloader:
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"Training round complete in {time_elapsed:.2f}: seconds")

        time.sleep((self.resources.speed_factor - 1) * time_elapsed)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"Client simulated to take {time_elapsed:.2f} seconds for training")

        # Return updated model parameters
        return local_model.state_dict()
    