import torch
import torch.nn as nn
from torch.nn.init import normal_

class SiameseNetwork(nn.Module):
    """
    Siamese Network architecture for one-shot learning tasks. This model uses a pair of images and 
    calculates a similarity score between them based on their Euclidean distance in an embedding space.
    """

    def __init__(self):
        """
        Initializes the Siamese Network by defining a convolutional network (ConvNet) to extract 
        features and two fully connected layers to produce a similarity score.
        """
        super(SiameseNetwork, self).__init__()

        # Define the ConvNet 
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10, stride=1, padding=0),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
    
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),  
            nn.ReLU(inplace=True),
            nn.Flatten(),  # Flatten layer
        )

        # Calculate the output size of convnet with a dummy input
        dummy_input = torch.randn(1, 1, 105, 105)  # Assuming input image size is 105x105
        output_size = self.convnet(dummy_input).shape[1]

        # Fully connected layer with 4096 units
        self.fc = nn.Sequential(
            nn.Linear(output_size, 4096),  # Adjust input size dynamically
            nn.Sigmoid()  # Sigmoid activation function
        )

        # Output layer to predict the similarity score
        self.final_fc = nn.Sequential(
            nn.Linear(1, 1),  # After Euclidean distance, we have a scalar, so input is 1
            nn.Sigmoid()  # For binary classification (similar/not similar)
        )

        # Custom weight initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        """
        Initializes weights for convolutional and linear layers with custom parameters.

        Parameters:
        - m (torch.nn.Module): The layer to initialize.

        """
        if isinstance(m, nn.Conv2d):
            normal_(m.weight, mean=0.0, std=0.01)  # W_init_1 for Conv2D
        elif isinstance(m, nn.Linear):
            normal_(m.weight, mean=0.0, std=0.2)  # W_init_2 for Dense layer
            normal_(m.bias, mean=0.5, std=0.01)   # b_init for biases

    def forward(self, input1, input2):
        """
        Forward pass for the Siamese Network. Takes two images, passes each through the ConvNet 
        and fully connected layers, and computes the Euclidean distance between their embeddings 
        as a measure of similarity.

        Parameters:
        - input1 (torch.Tensor): First input image tensor of shape (batch_size, channels, height, width).
        - input2 (torch.Tensor): Second input image tensor of shape (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Output similarity score of shape (batch_size, 1) with values between 0 and 1, 
          indicating the similarity between input1 and input2.
        """
        # Pass both inputs through the same convnet
        encoded_l = self.convnet(input1)
        encoded_r = self.convnet(input2)

        # Reshape the encoded vectors
        encoded_l = encoded_l.view(encoded_l.size(0), -1)  
        encoded_r = encoded_r.view(encoded_r.size(0), -1)  

        # Pass through the fully connected layer (fc) before calculating distance
        encoded_l = self.fc(encoded_l)
        encoded_r = self.fc(encoded_r)

        # Euclidean distance between the two encoded vectors
        euclidean_distance = torch.sqrt(torch.sum((encoded_l - encoded_r) ** 2, dim=1))

        # Pass the euclidean distance through a final fully connected layer for prediction
        output = self.final_fc(euclidean_distance.unsqueeze(1))  # Unsqueeze to match (batch_size, 1) for final layer
        return output
