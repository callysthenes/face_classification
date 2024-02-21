# Save the model
torch.save(model.state_dict(), "ModelCNN.pth")

# Create a new model
loaded_model = ManufacturingCNN()

# Load the saved model
loaded_model.load_state_dict(torch.load('ModelCNN.pth'))
print(loaded_model)