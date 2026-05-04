from ultralytics import YOLO

#model = YOLO('yolo11x.pt')
#model = YOLO('./src/models/best_yolo5x.pt')
model = YOLO('./src/models/best_yolo11s.pt')

results = model.predict('./data/Bundesliga.mp4', save=True)

# First frame of the video
firstFrame = results[0]
print(firstFrame)
print('==================================')
for box in firstFrame.boxes:
    print(box)
