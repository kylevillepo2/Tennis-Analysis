from ultralytics import YOLO
import cv2
import pickle # library used for serializing and deserializing python objects
import pandas as pd # import pandas for interpolate() function
 

class BallTracker:
    '''
    This class is meant to detect balls in a video. it should be initialized with a model and it has a function that 
    takes in a list of frames from read_video in utils that returns a list of dictionaries mapping ball IDs to their 
    bounding boxes
    ''' 
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions] # list of bounding boxes which is an empty list when there are no detections
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2']) # converts list to pandas dataframe

        # interpolate missing values
        df_ball_positions = df_ball_positions.interpolate() # does not interpolate numbers in beginning so need to make sure 
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        # dont have to detect_frames again if already been detected
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # serialize ball_detections into file in stub_path
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        result = self.model.predict(frame, conf=0.15)[0] # results is a result object from YOLO which has many attributes on the metadata of the frame

        ball_dict = {} # key is ball ID and values are bounding boxes
        for box in result.boxes:
            bounding_box = box.xyxy.tolist()[0] # takes the box in xmin xmax ymin ymax format as a list
            ball_dict[1] = bounding_box # add to ball_dict if class name is a person

        return ball_dict
    
    def draw__bboxes(self, video_frames, ball_detections):
        '''
        this function draws the bounding boxes for the balls in the video
        video_frames - list of the frames of the video
        ball_detections - list of dictionaries that contain trackid and bounding boxes for the balls
        '''
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                # add text for the ball ID 
                cv2.putText(frame, f"ball ID: {track_id}",(int(x1),int(y1) -10 ), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
                # make a rectangle with the bounding box and make it yellow
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2) 
            output_video_frames.append(frame)

        return output_video_frames