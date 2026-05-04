from utils import ReadVideo, SaveVideo
from trackers import Tracker

def main():
    # Read video
    videoFrames = ReadVideo('./data/Bundesliga.mp4')

    # Initialise Tracker
    tracker = Tracker('./src/models/best_yolo11s.pt')
    tracks = tracker.GetObjectTracks(videoFrames, read_from_data=True, data_path='./data/track_data.pkl')

    # Save video
    SaveVideo(videoFrames, './data/Bundesliga-output.mp4')

if __name__ == '__main__':
    main()