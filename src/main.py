from utils import ReadVideo, SaveVideo
from trackers import Tracker
import cv2

def GetCroppedImageOfPlayer(videoFrames, tracks):
    for trackID, player in tracks['players'][0].items():
        boundingBox = player['bbox']
        frame = videoFrames[0]

        # Crop boundingBox from frame
        croppedImage = frame[int(boundingBox[1]):int(boundingBox[3]), int(boundingBox[0]):int(boundingBox[2])]
        cv2.imwrite(f'data/croppedPlayer.jpg', croppedImage)
        break

    print(f'Image saved.')

def main():
    # Read video
    videoFrames = ReadVideo('./data/Bundesliga.mp4')

    # Initialise Tracker
    tracker = Tracker('./src/models/best_yolo11s.pt')
    tracks = tracker.GetObjectTracks(videoFrames, read_from_data=True, data_path='./data/track_data.pkl')

    # Save cropped image of a player for clustering
    #GetCroppedImageOfPlayer(videoFrames, tracks)

    # Draw output
    ## Draw object Tracks
    outputVideoFrames = tracker.DrawAnnotations(videoFrames, tracks)

    # Save video
    SaveVideo(outputVideoFrames, './data/Bundesliga-output.mp4')

if __name__ == '__main__':
    main()