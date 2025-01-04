

import torch # court line model is written in torch
import torchvision.transforms as transforms # to apply same transformations on images
import torchvision.models as models
import cv2 # to manipulate images
import numpy as np
#from trackNet import BallTrackerNet

class CourtLineDetector:
    def __init__(self, model_path): 
        self.model = models.resnet50(pretrained=True) # using resnet50 as our base model
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) # change last layer of model for keypoints
        #self.model = BallTrackerNet(out_channels = 15)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')) # loading in our trained weights to modify resnet50

        self.transform = transforms.Compose([ # standardize image into same size and normalize
                                        # transforms.Compose is a callable object
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def predict(self,image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # change image from BGR to RGB
        image_tensor = self.transform(img_rgb).unsqueeze(0) # transform image and unsqueeze (adding dimensions)

        with torch.no_grad(): # disables gradient calculation and is useful for inference when you know you will not call tensor.backward()
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]

        keypoints[::2] *= original_w/224.0
        keypoints[1::2] *= original_h/224.0

        return keypoints # list of keypoints on court
    
    def draw_keypoints(self, image, keypoints):
        '''
        Draws the keypoints on the court.
        image - the image of the tennis court from aerial view
        keypoints - a list of the keypoints
        returns: the image with the keypoints drawn on the tennis court
        '''
         
        for i in range(0, len(keypoints), 2): # iterate with step 2 to have each x and y 
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 2)
            cv2.circle(image, (x,y), 5, (0,0,255), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        '''
        draws keypoints on the whole video frame by frame
        '''
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
    


# import torch
# import torchvision.transforms as transforms
# import cv2
# from torchvision import models
# import numpy as np

# class CourtLineDetector:
#     def __init__(self, model_path):
#         self.model = models.resnet50(pretrained=True)
#         self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
#         self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def predict(self, image):

    
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_tensor = self.transform(image_rgb).unsqueeze(0)
#         with torch.no_grad():
#             outputs = self.model(image_tensor)
#         keypoints = outputs.squeeze().cpu().numpy()
#         original_h, original_w = image.shape[:2]
#         keypoints[::2] *= original_w / 224.0
#         keypoints[1::2] *= original_h / 224.0

#         return keypoints

#     def draw_keypoints(self, image, keypoints):
#         # Plot keypoints on the image
#         for i in range(0, len(keypoints), 2):
#             x = int(keypoints[i])
#             y = int(keypoints[i+1])
#             cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#             cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
#         return image
    
#     def draw_keypoints_on_video(self, video_frames, keypoints):
#         output_video_frames = []
#         for frame in video_frames:
#             frame = self.draw_keypoints(frame, keypoints)
#             output_video_frames.append(frame)
#         return output_video_frames