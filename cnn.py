import torchvision
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as T
import torch
import numpy as np

# detectable objects
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

class Vision:
    def __init__(self):
        # model for predictions
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
        self.model.eval()

    # load image to be visible by NN
    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB") 
        transform = T.Compose([T.ToTensor()])
        image = transform(image)
        
        with torch.no_grad():
            prediction = self.model([image])
        
        return (image, prediction)

    # blur the image
    def blur_image(self, position, predictions, image, blur_radius):
        x, y = int(position[0]), int(position[1])
        clicked_on_object = False
        
        # make image visible again
        image = image.permute(1, 2, 0).mul(255).byte().numpy()
        image = Image.fromarray(image)
        
        # mask and blur image
        masks = predictions[0]['masks'] > 0.7 
        for i in range(masks.shape[0]):  # Iterate through all detected objects
            mask = masks[i, 0]  # Extract the i-th mask 
            if mask[y, x].item() == 1:  # Check if the clicked point is within the object
                clicked_on_object = True
                
                # create the mask around the subject
                mask = mask.mul(255).byte().cpu().numpy()          # turns subject's pixel values into 1's, to distinctly separate subject from background (which are 0's)
                mask = np.array(mask, dtype=np.uint8)              # turn the points in the mask into an array
                mask = Image.fromarray(mask, mode="L")             # create an image from the array
                inverted_mask = ImageOps.invert(mask.convert('L')) # flip the mask so the rest of the image is turned into 1's, and subject is 0's
                blurred_image = image.filter(                      # blur the original image
                    ImageFilter.GaussianBlur(radius=blur_radius)
                )
                
                # based on the inverted mask make a composite image by selecting
                # pixels from the blurred image for background (1's), and pixels from
                # the original image from the subject (0's)
                composite_image = Image.composite(blurred_image, image, inverted_mask)
                return composite_image
        
        if not clicked_on_object:
            return None