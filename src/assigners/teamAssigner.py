from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.TeamColours = {}
        self.PlayerTeamDictionary = {}

    def GetClusteringModel(self, image):
        # Reshape the image into 2D array from (height, width, channels) to (height * width, channels)
        image2D = image.reshape(-1, 3)
        
        # Perform k-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(image2D)
        
        return kmeans

    def GetPlayerColour(self, frame, boundingBox):
        image = frame[int(boundingBox[1]):int(boundingBox[3]), int(boundingBox[0]):int(boundingBox[2])]

        topHalfImage = image[0:int(image.shape[0]/2), :]

        kmeans = self.GetClusteringModel(topHalfImage)

        # Get the cluster labels
        labels = kmeans.labels_
        
        # Reshape the labels into the image shape
        #clusteredImage = labels.reshape(image.shape[0], image.shape[1])[0: int(image.shape[0]/2), :] 
        clusteredImage = labels.reshape(topHalfImage.shape[0], topHalfImage.shape[1]) 

        # Get the player cluster
        cornerClusters = [clusteredImage[0, 0], clusteredImage[0, -1], clusteredImage[-1, 0], clusteredImage[-1, -1]]
        nonPlayerCluster = max(set(cornerClusters), key=cornerClusters.count)
        playerCluster = 1 - nonPlayerCluster

        playerColour = kmeans.cluster_centers_[playerCluster]

        return playerColour

    def AssignTeamColour(self, frame, playerDetections):

        playerColours = []
        for _, playerDetection in playerDetections.items():
            boundingBox = playerDetection['bbox']
            playerColour = self.GetPlayerColour(frame, boundingBox)
            playerColours.append(playerColour)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(playerColours)

        self.KMeans = kmeans

        self.TeamColours[1] = kmeans.cluster_centers_[0]
        self.TeamColours[2] = kmeans.cluster_centers_[1]

    def GetPlayerTeam(self, frame, playerBoundingBox, playerID):
        if playerID in self.PlayerTeamDictionary:
            return self.PlayerTeamDictionary[playerID]

        playerColour = self.GetPlayerColour(frame, playerBoundingBox)

        teamID = self.KMeans.predict(playerColour.reshape(1, -1))[0]
        teamID += 1

        self.PlayerTeamDictionary[playerID] = teamID

        return teamID
