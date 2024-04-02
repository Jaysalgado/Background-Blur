# import torchvision
# from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
# import torchvision.transforms as T
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
# model.eval()

# COCO_INSTANCE_CATEGORY_NAMES = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
#     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
#     'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
#     'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]
# def load_image(image_path):
#     image = Image.open(image_path).convert("RGB") 
#     transform = T.Compose([T.ToTensor()])
#     image = transform(image)
#     return image

# def onclick(event, predictions, image):
#     x, y = int(event.xdata), int(event.ydata)
#     clicked_on_object = False
    
#     masks = predictions[0]['masks'] > 0.7 
#     for i in range(masks.shape[0]):  # Iterate through all detected objects
#         mask = masks[i, 0]  # Extract the i-th mask 
#         if mask[y, x].item() == 1:  # Check if the clicked point is within the object
#             print(f'Clicked on {COCO_INSTANCE_CATEGORY_NAMES[prediction[0]["labels"][i].item()]} at ({x}, {y})')
#             clicked_on_object = True
#             mask = mask.mul(255).byte().cpu().numpy()
#             mask = np.array(mask, dtype=np.uint8)
#             mask = Image.fromarray(mask, mode="L")
#             inverted_mask = ImageOps.invert(mask.convert('L')) 
#             blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2))
#             # Create a composite image using the inverted mask to combine
#             # the blurred image with the original image in the area defined by the mask.
#             composite_image = Image.composite(blurred_image, image,   inverted_mask)
#             composite_image.show() 
#             break  
    
#     if not clicked_on_object:
#         print(f'Clicked on the background at ({x}, {y})')


# def visualize_prediction(image, prediction, threshold=0.7):
#     image = image.permute(1, 2, 0).mul(255).byte().numpy() #back into visualization format
#     image = Image.fromarray(image)
#     draw = ImageDraw.Draw(image)
#     font = ImageFont.load_default()

#     for element in range(len(prediction[0]["boxes"])):
#         score = prediction[0]["scores"][element].item()
#         if score > threshold: # Use the threshold here
#             box = prediction[0]["boxes"][element].tolist()
#             label = COCO_INSTANCE_CATEGORY_NAMES[prediction[0]["labels"][element].item()]
#             mask = prediction[0]["masks"][element, 0]

#             # Draw bounding box
#             # draw.rectangle(box, outline="red", width=3)

#             # Draw label
#             # draw.text((box[0], box[1]), f'{label}: {score:.2f}', fill="black", font=font)
#             # Draw mask
#             # mask = mask.mul(255).byte().cpu().numpy()
#             # mask = np.array(mask, dtype=np.uint8)
          
#             # image.paste(Image.fromarray(mask, mode="L"), (0, 0), Image.fromarray(mask, mode="L"))

#     fig, ax = plt.subplots() # Create a figure and an axes to plot
#     fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, prediction, image)) # Connect the onclick event
#     plt.imshow(image)
#     plt.axis("off")
#     plt.show()

# image_path = 'image.png' 
# image = load_image(image_path)

# with torch.no_grad():
#     prediction = model([image])

# visualize_prediction(image, prediction)
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import torchvision.transforms as T
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Load Mask R-CNN model with pre-trained weights
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
model.eval()

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

def load_image(image_path):
    image = Image.open(image_path).convert("RGB") 
    transform = T.Compose([T.ToTensor()])
    image = transform(image)
    return image

def onclick(event, predictions, image, blur_radius):
    x, y = int(event.xdata), int(event.ydata)
    clicked_on_object = False
    
    masks = predictions[0]['masks'] > 0.7 
    for i in range(masks.shape[0]):  # Iterate through all detected objects
        mask = masks[i, 0]  # Extract the i-th mask 
        if mask[y, x].item() == 1:  # Check if the clicked point is within the object
            print(f'Clicked on {COCO_INSTANCE_CATEGORY_NAMES[predictions[0]["labels"][i].item()]} at ({x}, {y})')
            clicked_on_object = True
            mask = mask.mul(255).byte().cpu().numpy()
            mask = np.array(mask, dtype=np.uint8)
            mask = Image.fromarray(mask, mode="L")
            inverted_mask = ImageOps.invert(mask.convert('L')) 
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            # Create a composite image using the inverted mask to combine
            # the blurred image with the original image in the area defined by the mask.
            composite_image = Image.composite(blurred_image, image, inverted_mask)
            composite_image.show() 
            break  
    
    if not clicked_on_object:
        print(f'Clicked on the background at ({x}, {y})')


def visualize_prediction(image, prediction, blur_radius, threshold=0.7):
    image = image.permute(1, 2, 0).mul(255).byte().numpy() #back into visualization format
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for element in range(len(prediction[0]["boxes"])):
        score = prediction[0]["scores"][element].item()
        if score > threshold: # Use the threshold here
            box = prediction[0]["boxes"][element].tolist()
            label = COCO_INSTANCE_CATEGORY_NAMES[prediction[0]["labels"][element].item()]
            mask = prediction[0]["masks"][element, 0]

    fig, ax = plt.subplots() # Create a figure and an axes to plot
    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, prediction, image, blur_radius)) # Connect the onclick event
    plt.imshow(image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Check if the blur radius argument is provided when running the script
    if len(sys.argv) >= 2:
        blur_radius = int(sys.argv[2])  # Assuming blur_radius is the first argument
    else:
        blur_radius = 2  # Default blur radius value if not provided

    image_path = 'image.png' 
    image = load_image(image_path)

    with torch.no_grad():
        prediction = model([image])

    visualize_prediction(image, prediction, blur_radius)

