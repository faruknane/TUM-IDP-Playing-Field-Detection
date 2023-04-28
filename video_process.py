import cv2


def TakeFrameFromVideo(video_path, frame_index):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Set the frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    # Read the frame
    ret, frame = cap.read()
    # Close the video file
    cap.release()
    # Return the frame
    return frame

def ReadAllFramesFromVideo(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get the number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Read all the frames
    frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        frames.append(frame)
    # Close the video file
    cap.release()
    # Return the frames
    return frames