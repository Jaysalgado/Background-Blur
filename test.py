import torchvision
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torch
import numpy as np
import matplotlib.pyplot as plt

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
model.eval()

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image = transform(image)
    return image

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def visualize_prediction(image, prediction, threshold=0.7):
    image = image.permute(1, 2, 0).mul(255).byte().numpy()
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for element in range(len(prediction[0]["boxes"])):
        score = prediction[0]["scores"][element].item()
        if score > threshold: # Use the threshold here
            box = prediction[0]["boxes"][element].tolist()
            label = COCO_INSTANCE_CATEGORY_NAMES[prediction[0]["labels"][element].item()]
            mask = prediction[0]["masks"][element, 0]

            # Draw bounding box
            draw.rectangle(box, outline="red", width=3)

            # Draw label
            draw.text((box[0], box[1]), f'{label}: {score:.2f}', fill="black", font=font)
            # Draw mask
            mask = mask.mul(255).byte().cpu().numpy()
            mask = np.array(mask, dtype=np.uint8)
            # Assuming you want to overlay mask on the image, you might need a more complex logic than provided
            image.paste(Image.fromarray(mask, mode="L"), (0, 0), Image.fromarray(mask, mode="L"))

    plt.imshow(image)
    plt.axis("off")
    plt.show()

image_path = 'image.jpg' 
image = load_image(image_path)

with torch.no_grad():
    prediction = model([image])

visualize_prediction(image, prediction)
