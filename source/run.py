import cv2
import plyfile
import numpy as np
import matplotlib.pyplot as plt
import sys

from kitti_reader import DatasetReaderKITTI
from feature_tracking import FeatureTracker
from utils import drawFrameFeatures, updateTrajectoryDrawing, savePly

from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Must specify sequence: python run.py seq with seq in [00,01,...,11]')
        sys.exit()
    seq = sys.argv[1]
    if seq not in ['00','01','02','03','04','05','06','07','08','09','10','11']:
        print('Sequence must be in [00,01,...,11]')
        sys.exit()

    # Initialize the SuperPoint model
    weights_path = 'SuperPointPretrainedNetwork/superpoint_v1.pth'
    superpoint = SuperPointFrontend(weights_path, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=True)

    tracker = FeatureTracker()
    detector = cv2.GFTTDetector_create()
    dataset_reader = DatasetReaderKITTI("../../data/", seq)

    K = dataset_reader.readCameraMatrix()

    prev_points_gftt = np.empty(0)
    prev_points_sp = np.empty(0)
    prev_frame_BGR = dataset_reader.readFrame(0)
    kitti_positions, track_positions_gftt, track_positions_sp = [], [], []
    camera_rot_gftt, camera_pos_gftt = np.eye(3), np.zeros((3,1))
    camera_rot_sp, camera_pos_sp = np.eye(3), np.zeros((3,1))

    plt.show()

    # Process next frames
    N = dataset_reader.number_images()

    for frame_no in range(1, N):
        print(f"Processing frame {frame_no}")
        curr_frame_BGR = dataset_reader.readFrame(frame_no)
        prev_frame = cv2.cvtColor(prev_frame_BGR, cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)

        # Feature detection & filtering
        prev_points_gftt = detector.detect(prev_frame) # tuple of cv2.KeyPoint
        prev_points_gftt = cv2.KeyPoint_convert(sorted(prev_points_gftt, key = lambda p: p.response, reverse=True)) # numpy.ndarray of numpy.ndarray

        prev_points_sp, descriptors, heatmap = superpoint.run(prev_frame.astype(np.float32) / 255.0)
        prev_points_sp = np.float32(prev_points_sp.T[:,:2])

        # Feature tracking (optical flow)
        prev_points_gftt, curr_points_gftt = tracker.trackFeatures(prev_frame, curr_frame, prev_points_gftt, removeOutliers=True)
        prev_points_sp, curr_points_sp = tracker.trackFeatures(prev_frame, curr_frame, prev_points_sp, removeOutliers=True)
        #print (f"{len(curr_points)} features left after feature tracking.")

        # Essential matrix, pose estimation
        E_gftt, mask_gftt = cv2.findEssentialMat(curr_points_gftt, prev_points_gftt, K, cv2.RANSAC, 0.99, 1.0, None)
        prev_points_gftt = np.array([pt for (idx, pt) in enumerate(prev_points_gftt) if mask_gftt[idx] == 1])
        curr_points_gftt = np.array([pt for (idx, pt) in enumerate(curr_points_gftt) if mask_gftt[idx] == 1])
        _, R_gftt, T_gftt, _ = cv2.recoverPose(E_gftt, curr_points_gftt, prev_points_gftt, K)

        E_sp, mask_sp = cv2.findEssentialMat(curr_points_sp, prev_points_sp, K, cv2.RANSAC, 0.99, 1.0, None)
        prev_points_sp = np.array([pt for (idx, pt) in enumerate(prev_points_sp) if mask_sp[idx] == 1])
        curr_points_sp = np.array([pt for (idx, pt) in enumerate(curr_points_sp) if mask_sp[idx] == 1])
        _, R_sp, T_sp, _ = cv2.recoverPose(E_sp, curr_points_sp, prev_points_sp, K)
        #print(f"{len(curr_points)} features left after pose estimation.")

        """
        if frame_no == N-1:
            print("Plotting last frame...")
            plt.cla()
            plt.plot(np.array(track_positions)[:,0], np.array(track_positions)[:,2], c='blue', label="Tracking")
            plt.plot(np.array(kitti_positions)[:,0], np.array(kitti_positions)[:,2], c='green', label="Ground truth")
            plt.title("Trajectory")
            plt.legend()
            plt.draw()
            plt.savefig("../results/trajectories_"+seq+".png")
        """

        # Read groundtruth translation T and absolute scale for computing trajectory
        kitti_pos, kitti_scale = dataset_reader.readGroundtuthPosition(frame_no)
        if kitti_scale <= 0.1:
            continue

        camera_pos_gftt = camera_pos_gftt + kitti_scale * camera_rot_gftt.dot(T_gftt)
        camera_rot_gftt = R_gftt.dot(camera_rot_gftt)

        camera_pos_sp = camera_pos_sp + kitti_scale * camera_rot_sp.dot(T_sp)
        camera_rot_sp = R_sp.dot(camera_rot_sp)

        kitti_positions.append(kitti_pos)
        track_positions_gftt.append(camera_pos_gftt)
        track_positions_sp.append(camera_pos_sp)
        updateTrajectoryDrawing(np.array(track_positions_gftt), np.array(track_positions_sp), np.array(kitti_positions))
        drawFrameFeatures(curr_frame, prev_points_gftt, curr_points_gftt, prev_points_sp, curr_points_sp, frame_no)

        if cv2.waitKey(1) == ord('q'):
            break
            
        prev_points_gftt, prev_points_sp, prev_frame_BGR = curr_points_gftt, curr_points_sp, curr_frame_BGR
        
    cv2.destroyAllWindows()
    print("Done")
