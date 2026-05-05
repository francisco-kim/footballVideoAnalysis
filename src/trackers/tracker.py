from ultralytics import YOLO
import supervision as sv
import os, sys, pickle, cv2
import numpy as np
import pandas as pd

sys.path.append('../')
from utils import GetBoundingBoxCentre, GetBoundingBoxWidth

class Tracker:
    def __init__(self, modelPath):
        self.model = YOLO(modelPath)
        self.tracker = sv.ByteTrack()

    def interpolateBallPositionsWithoutOutliners(self, ballPositions, remove_outliers=False):
        ballPositions = [p.get(1, {}).get('bbox', []) for p in ballPositions]
        positionsColumns = ['x1', 'y1', 'x2', 'y2']
        dfBallPositions = pd.DataFrame(ballPositions, columns=positionsColumns)

        # Remove outliers 
        D_THRESHOLD = 70
        if remove_outliers:
            cx = (dfBallPositions['x1'] + dfBallPositions['x2']) / 2
            cy = (dfBallPositions['y1'] + dfBallPositions['y2']) / 2

            dx = cx.diff().abs()   # current minus previous
            dy = cy.diff().abs()
            distance = np.sqrt(dx**2 + dy**2)

            outlier_mask = distance > D_THRESHOLD
            dfBallPositions.loc[outlier_mask, positionsColumns] = np.nan
            # first row distance is NaN; it stays untouched

        # if remove_outliers:
        #     Z_THRESHOLD = 3
        #     WINDOW = 2
        #     for column in positionsColumns:
        #         rollingMean = dfBallPositions[column].rolling(window=WINDOW, center=True).mean()
        #         rollingStd = dfBallPositions[column].rolling(window=WINDOW, center=True).std()
        #         rollingZScores = np.abs((dfBallPositions[column] - rollingMean) / rollingStd)
        #         dfBallPositions[column] = dfBallPositions[column].where(rollingZScores < Z_THRESHOLD, np.nan)

        # Interpolate missing values
        dfBallPositions = dfBallPositions.interpolate()
        dfBallPositions = dfBallPositions.bfill()

        ballPositions = [{1: {"bbox": p}} for p in dfBallPositions.to_numpy().tolist()]

        return ballPositions

    def DetectFrames(self, frames):
        batchSize = 20
        detections = []

        for i in range(0, len(frames), batchSize):
            detectionsBatch = self.model.predict(frames[i:i+batchSize], conf=0.1)
            detections += detectionsBatch
        return detections

    def GetObjectTracks(self, frames, read_from_data=False, data_path=None):

        if read_from_data and data_path is not None and os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.DetectFrames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "ball":[],
        }

        for frameIndex, detection in enumerate(detections):
            classNames = detection.names                                            # {0:person, ...}
            classNamesInverse = {value:key for key, value in classNames.items()}    # {person:0, ...}

            # Convert to supervision detection format
            detectionSupervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for objectIndex, classID in enumerate(detectionSupervision.class_id):
                if classNames[classID] == 'goalkeeper':
                    detectionSupervision.class_id[objectIndex] = classNamesInverse["player"]

            # Track objects
            detectionWithTracks = self.tracker.update_with_detections(detectionSupervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frameDetection in detectionWithTracks:
                boundingBox = frameDetection[0].tolist()
                classID = frameDetection[3]
                trackID = frameDetection[4]

                if classID == classNamesInverse['player']:
                    tracks['players'][frameIndex][trackID] = {'bbox':boundingBox}

                if classID == classNamesInverse['referee']:
                    tracks['referees'][frameIndex][trackID] = {'bbox':boundingBox}

            for frameDetection in detectionSupervision:
                boundingBox = frameDetection[0].tolist()
                classID = frameDetection[3]

                if classID == classNamesInverse['ball']:
                    tracks['ball'][frameIndex][1] = {'bbox':boundingBox}

        if data_path is not None:
            with open(data_path, 'wb') as f:
                pickle.dump(tracks, f)

        print(detectionSupervision)
    
    def DrawEllipse(self, frame, boundingBox, colour, trackID=None):
        y2 = int(boundingBox[3])
        xCentre, _ = GetBoundingBoxCentre(boundingBox)
        width = GetBoundingBoxWidth(boundingBox)

        cv2.ellipse(
            frame,
            center=(xCentre, y2),
            axes=(int(width), int(0.41 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=colour,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangleWidth = 41
        rectangleHeight = 20
        x1Rectangle = xCentre - rectangleWidth // 2
        x2Rectangle = xCentre + rectangleWidth // 2
        y1Rectangle = (y2 - rectangleHeight // 2) + 14
        y2Rectangle = (y2 + rectangleHeight // 2) + 14

        if trackID is not None:
            cv2.rectangle(frame,
                          (int(x1Rectangle), int(y1Rectangle)),
                          (int(x2Rectangle), int(y2Rectangle)),
                          colour,
                          cv2.FILLED
                          )

            x1Text = x1Rectangle + 14
            if trackID > 99:
                x1Text -= 10

            cv2.putText(
                frame,
                f"{trackID}",
                (int(x1Text), int(y1Rectangle + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def DrawTriangle(self, frame, boundingBox, colour):
        y = int(boundingBox[1])
        x, _ = GetBoundingBoxCentre(boundingBox)

        trianglePoints = np.array([
            [x, y],
            [x - 10, y -20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [trianglePoints], 0, colour, cv2.FILLED)
        cv2.drawContours(frame, [trianglePoints], 0, (0, 0, 0), 2)

        return frame

    def DrawAnnotations(self, videoFrames, tracks):
        outputVideoFrames = []

        for frameIndex, frame in enumerate(videoFrames):
            frame = frame.copy()

            playerDictionary = tracks['players'][frameIndex]
            refereeDictionary = tracks['referees'][frameIndex]
            ballDictionary = tracks['ball'][frameIndex]

            # Draw players
            for trackID, player in playerDictionary.items():
                colour = player.get('teamColour', (0, 0, 255))
                frame = self.DrawEllipse(frame, player['bbox'], colour, trackID)

            # Draw referees
            for trackID, referee in refereeDictionary.items():
                frame = self.DrawEllipse(frame, referee['bbox'], (0, 255, 255))

            # Draw ball
            for trackID, ball in ballDictionary.items():
                frame = self.DrawTriangle(frame, ball['bbox'], (0, 255, 0))

            outputVideoFrames.append(frame)

        return outputVideoFrames
