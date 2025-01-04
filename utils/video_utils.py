import cv2
def read_video_keypoints(path_video):
    """ Read video file
    :params
        path_video: path to video file
    :return
        frames: list of video frames
        fps: frames per second
    """
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def write_video_keypoints(imgs_new, fps, path_output_video):
    height, width = imgs_new[0].shape[:2]
    out = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'DIVX'),
                          fps, (width, height))
    for num in range(len(imgs_new)):
        frame = imgs_new[num]
        out.write(frame)
    out.release() 


def read_video(video_path):
    '''
    Ready video file
    params:
        video_path: path to video file
    returns:
        frames: list of frames
    '''
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    '''
    saves video to output_video_path
    '''
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()