import cv2
import torch
import numpy as np
from glob import glob

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Image():
    
    def __init__(self, image, predictor):
        self.image = self.image_mask = image

        self.predictor = predictor
        self.predictor.set_image(self.image)

        self.input_points = np.empty((0, 2), dtype=int)
        self.input_labels = np.empty((0), dtype=int)

        self.masks = None
        self.scores = None
        self.logits = None


    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.input_points = np.vstack([self.input_points, [x, y]])
            self.input_labels = np.append(self.input_labels, 1)

            self.update_masks()
            self.show_masks()

        
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.input_points = np.vstack([self.input_points, [x, y]])
            self.input_labels = np.append(self.input_labels, 0)

            self.update_masks()
            self.show_masks()


    def remove_last(self):   
        # TODO: fix out-of-bounds error         
        self.input_points = self.input_points[:-1]
        self.input_labels = self.input_labels[:-1]

        self.update_masks()
        self.show_masks()
            

    def update_masks(self):
        self.masks, self.scores, self.logits = self.predictor.predict(
            point_coords=self.input_points,
            point_labels=self.input_labels,
            multimask_output=False,
        )
        
    
    def show_masks(self):
        masks = np.logical_or.reduce(self.masks).astype(np.uint8)

        new_masks = np.zeros((*masks.shape, 4), dtype=np.uint8)
        new_masks[masks == 1] = [255, 127, 0, 127]

        image_alpha = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)

        alpha_mask = new_masks[:, :, 3] / 255.0  # Normalize to [0,1]
        alpha_mask = alpha_mask[..., np.newaxis]  # Expand dims to match shape

        blended = (new_masks[:, :, :3] * alpha_mask + image_alpha[:, :, :3] * (1 - alpha_mask)).astype(np.uint8)
        image_alpha = np.dstack([blended, (alpha_mask * 255).astype(np.uint8)])

        self.image_mask = cv2.cvtColor(image_alpha, cv2.COLOR_BGRA2BGR)

        for i in range(len(self.input_points)):
            if self.input_labels[i] == 1:
                self.image_mask = cv2.circle(self.image_mask, self.input_points[i].tolist(), radius=8, color=(31, 223, 31), thickness=-1)
            else:
                self.image_mask = cv2.circle(self.image_mask, self.input_points[i].tolist(), radius=8, color=(31, 31, 223), thickness=-1)
            self.image_mask = cv2.circle(self.image_mask, self.input_points[i].tolist(), radius=9, color=(255, 255, 255), thickness=2)


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
    np.random.seed(3)
    
    device = select_device()

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    model = SAM2ImagePredictor(sam2_model)

    print()
    print('Left click to add a positive point.')
    print('Right click to add a negative point.')
    print('Press "z" to remove the most recent point.')
    print('Press "q" to exit and save masks.')

    all_images = glob("input/*")
    finished_images = glob("output/*")

    files = []
    for i in all_images:
        if i not in finished_images:
            files.append(i)

    for file in files:
        image_name = file.split("/")[-1].strip(".jpg")
        print(f'\n\nImage: {image_name}')

        my_image = Image(
            image=cv2.imread(file, 1),
            predictor=model
        )

        cv2.namedWindow(winname='image')
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('image', my_image.click_event) 

        while True:
            cv2.imshow('image', my_image.image_mask)
            key = cv2.waitKey(1) 
            if key == ord('z'):
                my_image.remove_last()
            elif key == ord('q'):
                break

        cv2.destroyAllWindows() 

        np.save(
            f'output/{image_name}.npy',
            my_image.masks, 
        )

        again = input("\nWould you like to continue? [y/n] ")
        if again.lower() == 'n' or again.lower() == 'no':
            break
