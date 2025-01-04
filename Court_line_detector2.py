import os
import cv2
import numpy as np
import torch
from trackNet import BallTrackerNet
import torch.nn.functional as F
from tqdm import tqdm
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps
import argparse
from utils import read_video_keypoints, write_video_keypoints

class CourtLineDetector2:
    def __init__(self, model_path):
        self.model = BallTrackerNet(out_channels=15)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def infer_video(self, frames):
        OUTPUT_WIDTH = 640
        OUTPUT_HEIGHT = 360
    
        self.frames_upd = [] # updated frames
        for image in tqdm(frames):
            img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            inp = (img.astype(np.float32) / 255.)
            inp = torch.tensor(np.rollaxis(inp, 2, 0))
            inp = inp.unsqueeze(0)

            out = self.model(inp.float().to(self.device))[0]
            pred = F.sigmoid(out).detach().cpu().numpy()

            points = []
            for kps_num in range(14):
                heatmap = (pred[kps_num] * 255).astype(np.uint8)
                x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
                if kps_num not in [8, 12, 9] and x_pred and y_pred:
                    x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
                points.append((x_pred, y_pred))

            # matrix_trans = get_trans_matrix(points)
            # if matrix_trans is not None:
            #     points = cv2.perspectiveTransform(refer_kps, matrix_trans)
            #     points = [np.squeeze(x) for x in points]

            for j in range(len(points)):
                if points[j][0] is not None:
                    image = cv2.circle(image, (int(points[j][0]), int(points[j][1])),
                                    radius=0, color=(0, 0, 255), thickness=10)
            self.frames_upd.append(image)

        return self.frames_upd
    