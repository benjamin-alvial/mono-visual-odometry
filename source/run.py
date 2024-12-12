import cv2
import plyfile
import numpy as np
import matplotlib.pyplot as plt

from kitti_reader import DatasetReaderKITTI
from feature_tracking import FeatureTracker
from utils import drawFrameFeatures, updateTrajectoryDrawing, savePly

if __name__ == "__main__":
    tracker = FeatureTracker()
    detector = cv2.GFTTDetector_create()
    dataset_reader = DatasetReaderKITTI("../../data/")

    K = dataset_reader.readCameraMatrix()

    prev_points = np.empty(0)
    prev_frame_BGR = dataset_reader.readFrame(0)
    kitti_positions, track_positions = [], []
    camera_rot, camera_pos = np.eye(3), np.zeros((3,1))

    plt.show()

    # Process next frames
    N=5
    for frame_no in range(1, N+1):
        curr_frame_BGR = dataset_reader.readFrame(frame_no)
        prev_frame = cv2.cvtColor(prev_frame_BGR, cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)

        # Feature detection & filtering
        prev_points = detector.detect(prev_frame) # tuple of cv2.KeyPoint
        prev_points = cv2.KeyPoint_convert(sorted(prev_points, key = lambda p: p.response, reverse=True)) # numpy.ndarray of numpy.ndarray

        # Feature tracking (optical flow)
        prev_points, curr_points = tracker.trackFeatures(prev_frame, curr_frame, prev_points, removeOutliers=True)
        print (f"{len(curr_points)} features left after feature tracking.")

        # Essential matrix, pose estimation
        E, mask = cv2.findEssentialMat(curr_points, prev_points, K, cv2.RANSAC, 0.99, 1.0, None)
        prev_points = np.array([pt for (idx, pt) in enumerate(prev_points) if mask[idx] == 1])
        curr_points = np.array([pt for (idx, pt) in enumerate(curr_points) if mask[idx] == 1])
        _, R, T, _ = cv2.recoverPose(E, curr_points, prev_points, K)
        print(f"{len(curr_points)} features left after pose estimation.")

        # Read groundtruth translation T and absolute scale for computing trajectory
        kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(frame_no)
        if kitti_scale <= 0.1:
            continue

        camera_pos = camera_pos + kitti_scale * camera_rot.dot(T)
        camera_rot = R.dot(camera_rot)

        kitti_positions.append(kitti_pos)
        track_positions.append(camera_pos)
        updateTrajectoryDrawing(np.array(track_positions), np.array(kitti_positions))
        drawFrameFeatures(curr_frame, prev_points, curr_points, frame_no)

        if frame_no == N:
            plt.cla()
            plt.plot(np.array(track_positions)[:,0], np.array(track_positions)[:,2], c='blue', label="Tracking")
            plt.plot(np.array(kitti_positions)[:,0], np.array(kitti_positions)[:,2], c='green', label="Ground truth")
            plt.title("Trajectory")
            plt.legend()
            plt.draw()
            plt.savefig("../results/trajectories.png")

        if cv2.waitKey(1) == ord('q'):
            break
            
        prev_points, prev_frame_BGR = curr_points, curr_frame_BGR
        
    cv2.destroyAllWindows()
    print("Done")
