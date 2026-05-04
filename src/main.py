from utils import ReadVideo, SaveVideo

def main():
    # Read video
    videoFrames = ReadVideo('./data/Bundesliga.mp4')

    # Save video
    SaveVideo(videoFrames, './data/Bundesliga-output.mp4')

if __name__ == '__main__':
    main()