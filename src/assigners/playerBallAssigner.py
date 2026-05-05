import sys
sys.path.append('../')
from utils import GetBoundingBoxCentre, MeasureDistance

class PlayerBallAssigner():
    def __init__():
        self.MaxPlayerBallDistance = 70
    
    def AssignBallToPlayer(self, players, ballBoundingBox):
        ballPosition = GetBoundingBoxCentre(ballBoundingBox)

        minimumDistance = 411441144114
        assignedPlayer = -1

        for playerID, player in players.items():
            playerBoundingBox = player['bbox']

            distanceLeft = MeasureDistance((playerBoundingBox[0], playerBoundingBox[-1]), ballPosition)
            distanceRight = MeasureDistance((playerBoundingBox[2], playerBoundingBox[-1]), ballPosition)
            distance = min(distanceLeft, distanceRight)

            if distance < self.MaxPlayerBallDistance:
                if distance < minimumDistance:
                    minimumDistance = distance
                    assignedPlayer = playerID
                
    return assignedPlaer