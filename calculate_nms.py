# Import nms
from torchvision.ops import nms

# Set the IoU threshold
iou_threshold = 0.5

# Apply non-max suppression
box_indices = nms(
    boxes=boxes,
    scores=scores,
    iou_threshold = iou_threshold
)

# Filter boxes
filtered_boxes = boxes[box_indices]

print("Filtered Boxes:", filtered_boxes)