from ultralytics import YOLO
import cv2
import pickle # library used for serializing and deserializing python objects
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox


class PlayerTracker:
    '''
    This class is meant to detect players in a video. it should be initialized with a model and it has a function that 
    takes in a list of frames from read_video in utils that returns a list of dictionaries mapping player IDs to their 
    bounding boxes
    ''' 
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        '''
        takes in court_keypoints - 
        '''
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        #choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        # dont have to detect_frames again if already been detected
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        # serialize player_detections into file in stub_path
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        result = self.model.track(frame, persist=True)[0] # result is a result object from YOLO which has many attributes on the metadata of the frame
        id_name_dict = result.names # dictionary mapping of class ids to class names

        player_dict = {} # key is player ID and values are bounding boxes
        for box in result.boxes:
            track_id = int(box.id.tolist()[0]) # takes the track id for the box
            bounding_box = box.xyxy.tolist()[0] # takes the box in xmin xmax ymin ymax format as a list
            object_cls_id = box.cls.tolist()[0] # class labels for each box as a list
            object_cls_name = id_name_dict[object_cls_id] # class names for each box
            if object_cls_name == "person":
                player_dict[track_id] = bounding_box # add to player_dict if class name is a person

        return player_dict
    
    def draw__bboxes(self, video_frames, player_detections):
        '''
        this function draws the bounding boxes for the players in the video
        video_frames - list of the frames of the video
        player_detections - list of dictionaries that contain trackid and bounding boxes for the players
        '''
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                # add text for the player ID 
                cv2.putText(frame, f"Player ID: {track_id}",(int(x1),int(y1) -10 ), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
                # make a rectangle with the bounding box and make it red
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2) 
            output_video_frames.append(frame)

        return output_video_frames