
from utils import (read_video, 
                   save_video, read_video_keypoints, write_video_keypoints)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from Court_line_detector2 import CourtLineDetector2
from mini_court import MiniCourt
import cv2



def main():
    # reading video
    input_video_path = "input_data/input_video.mp4"
    video_frames = read_video(input_video_path)

    # detecting players
    player_tracker = PlayerTracker(model_path='models/yolo11x.pt') # create PlayerTracker object
    player_detections = player_tracker.detect_frames(video_frames,  # list of player detection dictionaries
                                                     read_from_stub=True,  
                                                     stub_path="tracker_stubs/player_detections.pkl")
    
    # detecting balls
    ball_tracker = BallTracker(model_path='models/last.pt') # create BallTracker object
    
    ball_detections = ball_tracker.detect_frames(video_frames,  # list of ball detection dictionaries
                                                     read_from_stub=True, 
                                                     stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # Court Line Detector model
    court_model_path = "models/keypoints_model7.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # filter out people not playing
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    #initialize mini court
    mini_court = MiniCourt(video_frames[0])



    # Draw output

    # Draw player Bounding Boxes
    output_video_frames = player_tracker.draw__bboxes(video_frames, player_detections)

    # Draw ball bounding boxes
    output_video_frames = ball_tracker.draw__bboxes(output_video_frames, ball_detections)

    # Draw court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    # court_line_detector = CourtLineDetector2(court_model_path)
    # output_video_frames = court_line_detector.infer_video(output_video_frames)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court_video(output_video_frames)

    # Draw frame numbers 
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    save_video(output_video_frames, "output_videos/output_video3.avi")

if __name__ == "__main__":
    main()