import cv2
import numpy as np

class RapidsImage():
    
    def __init__(self, image, predictor=None, has_textbox=False, msg=''):
        self.image = self.image_mask = image

        self.predictor = predictor
        if self.predictor != None:
            self.predictor.set_image(self.image)

        self.input_points = np.empty((0, 2), dtype=int)
        self.input_labels = np.empty((0), dtype=int)

        self.masks = None
        self.scores = None
        self.logits = None

        self.rapid_class = -1

        self.has_textbox = has_textbox
        self.textmsg = msg

        self.display_image()


    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.input_points = np.vstack([self.input_points, [x, y]])
            self.input_labels = np.append(self.input_labels, 1)

            self.display_image()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.input_points = np.vstack([self.input_points, [x, y]])
            self.input_labels = np.append(self.input_labels, 0)

            self.display_image()


    def remove_last(self):   
        self.input_points = self.input_points[:-1]
        self.input_labels = self.input_labels[:-1]

        self.display_image()
            

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


    def show_textbox(self):
        textbox = np.zeros((40, self.image_mask.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            textbox,
            text = self.textmsg,
            org = (8, 28),
            fontFace = cv2.FONT_HERSHEY_COMPLEX,
            fontScale = 1,
            color = (255, 255, 255),
            thickness = 1,
        )
        self.image_mask = np.vstack((self.image_mask, textbox))


    def display_image(self):
        self.image_mask = self.image
        if self.predictor != None:
            self.update_masks()
            self.show_masks()
        if self.has_textbox:
            self.show_textbox()


    def set_textmsg(self, msg):
        self.textmsg = msg


    def set_rapid_class(self, value):
        self.rapid_class = value
