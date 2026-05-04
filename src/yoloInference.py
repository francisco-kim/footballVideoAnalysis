from ultralytics import YOLO

model = YOLO('yolo11x.pt')

results = model.predict('./data/Bundesliga.mp4', stream=True, save=True)

# First frame of the video
firstFrame = results[0]
print(firstFrame)
print('==================================')
for box in firstFrame.boxes:
    print(box)
