import torch
import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ImageMask():
    
    def __init__(self, image, predictor):
        self.image = self.image_mask = image

        self.predictor = predictor
        self.predictor.set_image(self.image)

        self.input_points = np.empty((0, 2), dtype=float)
        self.input_labels = np.empty((0), dtype=int)

        self.masks = None
        self.scores = None
        self.logits = None

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            
            self.input_points = np.vstack([self.input_points, [x, y]])
            self.input_labels = np.append(self.input_labels, 1)

            print(self.input_points, self.input_labels, '\n')

            self.update_mask()
            self.show_masks()


    # def set_image(self):
    #     # TODO: update mask_image given the input_points/input_labels
        
    #     return
    
    def update_mask(self):
        self.masks, self.scores, self.logits = self.predictor.predict(
            point_coords=self.input_points,
            point_labels=self.input_labels,
            multimask_output=False,
        )
        







    def show_mask(self, mask, image):

        mask = mask.astype(np.uint8) * 255  # Convert mask to binary
        mask_colored = np.zeros_like(image, dtype=np.uint8)
        mask_colored[:, :, :] = np.array([30, 144, 255], dtype=np.uint8)  # BGR for OpenCV
        overlay = cv2.addWeighted(image, 1, mask_colored, 0.6, 0, dtype=cv2.CV_8U)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01 * cv2.arcLength(contour, True), closed=True) for contour in contours]
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), thickness=2)
        
        return overlay
    

    def show_points(self, image, coords, labels, marker_size=4):

        for point, label in zip(coords, labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for positive, Red for negative
            cv2.drawMarker(image, tuple(point), color, markerType=cv2.MARKER_STAR, markerSize=marker_size, thickness=2, line_type=cv2.LINE_AA)
        
        return image
    

    def show_masks(self): # , image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):

        image = self.image.copy()

        for mask, score in zip(self.masks, self.scores):
            image = self.show_mask(mask, image)

        # if self.input_points is not None:
        #     image = self.show_points(image, self.input_points, self.input_labels)

        self.image_mask = image






def select_device():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    
    return device



if __name__=="__main__": 
    
    # np.random.seed(3)
    
    device = select_device()

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    imageMask = ImageMask(
        image=cv2.imread('ExamplesFromDataRelease/Rogue_Devil_05012015.jpg', 1),
        predictor=SAM2ImagePredictor(sam2_model)
    )

    # TODO: can i load the model once for many images?
    # TODO: begin while loop
    # TODO: implement glob (or similar) to get images

    cv2.namedWindow(winname='image')
    cv2.setMouseCallback('image', imageMask.click_event) 

    while True:
        cv2.imshow('image', imageMask.image_mask)
        key = cv2.waitKey(1) 
        if key == ord('q'):
            break

    cv2.destroyAllWindows() 

    # np.save(
    #     f'output/{image_name}_label.npy',
    #     masks, 
    # )



