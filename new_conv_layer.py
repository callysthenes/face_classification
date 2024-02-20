"""
Instantiate a model from the CNNModel class and access the convolutional layers.
Create a new convolutional layer with in_channels equal to existing layer's out_channels, out_channels set to 32, and stride and padding both set to 1, and assign it to conv2.
Append the new layer to the model, calling it "conv2".
"""


# Create a model
model = CNNModel()
print("Original model: ", model)

# Create a new convolutional layer
conv2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, padding=1)

# Append the new layer to the model
model.add_module('conv2', conv2)
print("Extended model: ", model)