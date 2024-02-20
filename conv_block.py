"""
You decided to redesign your binary CNN model template by creating a 
block of convolutional layers. This will help you stack multiple layers s
sequentially. With this improved model, you will be able to easily design 
various CNN architectures.
"""

class BinaryImageClassification(nn.Module):
  def __init__(self):
    super(BinaryImageClassification, self).__init__()
    # Create a convolutional block
    self.conv_block = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
    )
    
  def forward(self, x):
    # Pass inputs through the convolutional block
    x = self.conv_block(x)
    return x