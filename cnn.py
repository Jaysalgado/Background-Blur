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
        image = image.permute(1, 2, 0).mul(255).byte().numpy() #back into visualization format
        image = Image.fromarray(image)
        
        # mask and blur image
        masks = predictions[0]['masks'] > 0.7 
        for i in range(masks.shape[0]):  # Iterate through all detected objects
            mask = masks[i, 0]  # Extract the i-th mask 
            if mask[y, x].item() == 1:  # Check if the clicked point is within the object
                clicked_on_object = True
                
                # create the mask around the subject
                mask = mask.mul(255).byte().cpu().numpy()
                mask = np.array(mask, dtype=np.uint8)
                mask = Image.fromarray(mask, mode="L")
                inverted_mask = ImageOps.invert(mask.convert('L')) 
                blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                # Create a composite image using the inverted mask to combine
                # the blurred image with the original image in the area defined by the mask.
                composite_image = Image.composite(blurred_image, image, inverted_mask)
                return composite_image
        
        if not clicked_on_object:
            return None