# Get model's prediction
with torch.no_grad():
    output = model(test_image)

# Extract boxes from the output
boxes = output[0]["boxes"]

# Extract scores from the output
scores = output[0]["scores"]

print(boxes, scores)