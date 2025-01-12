import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from dataclasses import dataclass, field
import random
import fl
from models import *
from quantization import *


@dataclass
class ClientResources:
    """
    A dataclass representing the computational and resource capabilities of a simulated client
    in a federated learning environment. This class is designed to model client heterogeneity
    by including various attributes related to processing speed, power levels, memory, and network bandwidth.

    Attributes:
        speed_factor (float): 
            A multiplier representing the client's processing speed relative to a baseline.
            Higher values indicate faster processing.
        
        battery_level (float): 
            The current battery level of the client as a percentage (0.0 to 100.0).
            Used to simulate throttling or resource limitations based on power availability.
        
        bandwidth (float): 
            The network bandwidth available to the client, measured in Mbps.
            Impacts the time required for model upload and download.
        
        dataset_size (int): 
            The size of the local dataset available to the client, measured in the number of samples.
            Larger datasets may increase the time required for local model updates.
        
        CPU_available (bool): 
            A flag indicating whether the client has a CPU available for computation.
            If False, the client cannot perform local updates.
        
        CPU_memory_availability (float): 
            The amount of available memory (in GB) on the client's CPU.
            Determines whether the client can handle memory-intensive computations.
        
        GPU_available (bool): 
            A flag indicating whether the client has a GPU available for computation.
            If True, the client is expected to process faster compared to CPU-only clients.
        
        GPU_memory_availability (float): 
            The amount of available memory (in GB) on the client's GPU.
            Critical for handling large models or datasets during training.

    Example:
        >>> client = ClientResources(
        ...     speed_factor=2.5,
        ...     battery_level=75.0,
        ...     bandwidth=50.0,
        ...     dataset_size=10000,
        ...     CPU_available=True,
        ...     CPU_memory_availability=16.0,
        ...     GPU_available=True,
        ...     GPU_memory_availability=8.0,
        ... )
        >>> print(client)
        ClientResources(
            speed_factor=2.5,
            battery_level=75.0,
            bandwidth=50.0,
            dataset_size=10000,
            CPU_available=True,
            CPU_memory_availability=16.0,
            GPU_available=True,
            GPU_memory_availability=8.0
        )
    """
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


    def get_slowdown_factor(self):
        """
        Calculate the slowdown factor for the client based on battery level,
        CPU/GPU availability, and other attributes.
        """
        # Battery level-based slowdown
        if self.battery_level > 50:
            battery_slowdown = 1.0  # Full speed
        elif 20 <= self.battery_level <= 50:
            battery_slowdown = 0.5  # Half speed
        else:
            battery_slowdown = 0.0  # No processing

        # Processing availability-based slowdown
        if self.GPU_available:
            processing_slowdown = 1.0 * self.GPU_memory_availability / 32  # Full speed with GPU
        elif self.CPU_available:
            processing_slowdown = 0.3 * self.CPU_memory_availability / 128 # Reduced speed with only CPU
        else:
            processing_slowdown = 0.0  # No processing available

        # Bandwidth influence (optional): Assume speed reduces slightly with lower bandwidth
        bandwidth_slowdown = max(0.5, min(1.0, self.bandwidth / 100))

        # Combine all slowdown factors
        total_slowdown = battery_slowdown * processing_slowdown * bandwidth_slowdown

        # Apply speed_factor for additional adjustments
        adjusted_speed = self.speed_factor * total_slowdown

        return adjusted_speed





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

    def train(self, global_model, epochs=1, lr=0.001, quantize=False, lambda_kure=0.0, delta=0.0, setup='standard', bit_widths=[32]):
        """
        Train the global model on the client's local dataset using Adam optimizer.

        Parameters:
        - global_model (torch.nn.Module): The global model to be trained.
        - epochs (int, optional): The number of training epochs. Default is 1.

        Returns:
        - state_dict (dict): The updated model parameters.
        """
        # track start time
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        # Directly copy the global model
        local_model = QuantStubModel(q=quantize)
        if(quantize):
            local_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
            torch.quantization.prepare_qat(local_model, inplace=True)
        local_model.load_state_dict(global_model.state_dict())
        
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            local_model.parameters(), 
            lr=lr, 
            # learning hyperparameters can be set later
            # betas=(0.9, 0.99), 
            # eps=1e-7, 
            weight_decay=1e-4
        )
        
        # Simulate training delay based on speed_factor
        local_model.train()
        for epoch in range(epochs):
            for inputs, labels in self.dataloader:
                optimizer.zero_grad()

                if setup == 'mqat':
                    # Apply Pseudo-Quantization Noise (APQN)
                    if delta is not None:
                        for param in local_model.parameters():
                            param.data = add_pseudo_quantization_noise(param, delta)

                    # Apply Multi-Bit Quantization (MQAT)
                    if bit_widths is not None:
                        bit_width = random.choice(bit_widths)
                        for param in local_model.parameters():
                            param.data = quantize_multi_bit(param, bit_width)

                outputs = local_model(inputs)
                loss = criterion(outputs, labels)

                # Kurtosis Regularization
                if setup == 'kure':
                    for param in local_model.parameters():
                        loss += lambda_kure * kurtosis_regularization(param)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        end_time = time.time()
        time_elapsed = end_time - start_time
        #print(f"Training round complete in {time_elapsed:.2f}: seconds")

        # time.sleep((self.resources.speed_factor - 1) * time_elapsed)
        time.sleep(self.resources.get_slowdown_factor() * time_elapsed)

        end_time = time.time()
        time_elapsed = end_time - start_time
        #print(f"Client simulated to take {time_elapsed:.2f} seconds for training")

        # Return updated model parameters
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return local_model.state_dict(), avg_loss
    