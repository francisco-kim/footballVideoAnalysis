def GetBoundingBoxCentre(boundingBox):
    x1, y1, x2, y2 = boundingBox

    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def GetBoundingBoxWidth(boundingBox):
    return boundingBox[2] - boundingBox[0]