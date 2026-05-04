import cv2

def ReadVideo(videoPath):
    capture = cv2.VideoCapture(videoPath)
    frames = []
    while True:
        exists, frame = capture.read()
        if not exists:
            break
        frames.append(frame)

    return frames

def SaveVideo(outputVideoFrames, outputVideoPath):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputVideoPath, fourcc, fps=24, frameSize=(outputVideoFrames[0].shape[1], outputVideoFrames[0].shape[0]))
    for frame in outputVideoFrames:
        out.write(frame)
    
    out.release()
