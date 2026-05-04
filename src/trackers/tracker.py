from ultralytics import YOLO
import supervision as sv
import pickle, os

class Tracker:
    def __init__(self, modelPath):
        self.model = YOLO(modelPath)
        self.tracker = sv.ByteTrack()

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