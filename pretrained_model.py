    """You are building an application to label images from the social media. 
        This task requires high accuracy and speed. You are going to use a pre-trained ResNet18 model to infer image classes.
    """


# Import resnet18 model
from torchvision.models import (
    resnet18, ResNet18_Weights
)

# Initialize model with default weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Set model to evaluation mode
model.eval()

# Initialize the transforms
transform = weights.transforms()