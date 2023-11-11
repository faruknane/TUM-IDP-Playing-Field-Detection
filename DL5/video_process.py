import cv2
import time
import os
import pathlib
import glob

def TakeFrameFromVideo(video_path, frame_index):
    # print(video_path, frame_index)
    # get the directory of the video_path
    video_dir = os.path.dirname(video_path)

    # get the file name of the video_path without extension
    file_name = pathlib.Path(video_path).stem
    # get the path of the folder
    folder_path = os.path.join(video_dir, file_name)

    # check if frame_index.png exists
    if not os.path.exists(os.path.join(folder_path, str(frame_index) + ".png")):
       
        # create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # extract the target frame from the video
        frame = TakeFrameFromVideo2(video_path, frame_index)
        # save the frame as a png image
        cv2.imwrite(os.path.join(folder_path, str(frame_index) + ".png"), frame)

        #rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame
    else:
        # read the frame from the folder
        frame = cv2.imread(os.path.join(folder_path, str(frame_index) + ".png"), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

        
def ExtractFramesFromVideo(video_path, folder_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get the number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Read all the frames
    for i in range(frame_count):
        ret, frame = cap.read()
        # Save the frame as a png image
        cv2.imwrite(os.path.join(folder_path, str(i) + ".png"), frame)
    # Close the video file
    cap.release()


def TakeFrameFromVideo2(video_path, frame_index):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(frame_count)
    # Set the frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    # Read the frame
    ret, frame = cap.read()
    # Close the video file
    cap.release()
    # Return the frame
    return frame


if __name__ == "__main__":

    # x = TakeFrameFromVideo2("data/train/videos/x.mov", 255)
    # print(x.shape)

    import random

    my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # your list here
    random.seed(0) # set the seed to a fixed value
    chosen_elements = random.sample(my_list, k=5) # choose 20 elements from the list

    print(chosen_elements)
