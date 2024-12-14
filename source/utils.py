import cv2
import matplotlib.pyplot as plt
import numpy as np

# Draws detected and tracked features on a frame (motion vector is drawn as a line).
# @param frame Frame to be used for drawing (will be converted to RGB).
# @param prevPts Previous frame keypoints.
# @param currPts Next frame keypoints.
def drawFrameFeatures(frame, prevPtsGFTT, currPtsGFTT, prevPtsSP, currPtsSP, frameIdx):
    currFrameRGB = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    for i in range(len(currPtsGFTT)-1):
        cv2.circle(currFrameRGB, (int(currPtsGFTT[i][0]),int(currPtsGFTT[i][1])), radius=3, color=(200, 100, 0))
        cv2.line(currFrameRGB, (int(prevPtsGFTT[i][0]),int(prevPtsGFTT[i][1])), (int(currPtsGFTT[i][0]),int(currPtsGFTT[i][1])), color=(200, 100, 0))
    cv2.putText(currFrameRGB, "Frame: {}".format(frameIdx), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))
        #cv2.putText(currFrameRGB, "Features: {}".format(len(currPts)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200)) 
    for i in range(len(currPtsSP)-1):
        cv2.circle(currFrameRGB, (int(currPtsSP[i][0]),int(currPtsSP[i][1])), radius=3, color=(200, 200, 0))
        cv2.line(currFrameRGB, (int(prevPtsSP[i][0]),int(prevPtsSP[i][1])), (int(currPtsSP[i][0]),int(currPtsSP[i][1])), color=(200, 200, 0))
    
    cv2.imshow("Frame with keypoints", currFrameRGB)

#
# @param trackedPoints
# @param groundtruthPoints
def updateTrajectoryDrawing(trackedPointsGFTT, trackedPointsSP, groundtruthPoints):
    plt.cla()
    plt.plot(trackedPointsGFTT[:,0], trackedPointsGFTT[:,2], c='blue', label="Tracking GFTT")
    plt.plot(trackedPointsSP[:,0], trackedPointsSP[:,2], c='red', label="Tracking SP")
    plt.plot(groundtruthPoints[:,0], groundtruthPoints[:,2], c='green', label="Ground truth")
    plt.title("Trajectory")
    plt.legend()
    plt.draw()
    plt.pause(0.001)

def savePly(points, colors, output_file):
    vertexes = [ (p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(points, colors)]
    vertexes = [ v for v in vertexes if v[2] >= 0 ] # Discard negative z
    dtypes = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    array = np.array(vertexes, dtype=dtypes)
    element = plyfile.PlyElement.describe(array, "vertex")
    plyfile.PlyData([element]).write(output_file)