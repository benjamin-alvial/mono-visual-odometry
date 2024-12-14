import cv2
import plyfile
import numpy as np
import matplotlib.pyplot as plt
import sys

from kitti_reader import DatasetReaderKITTI
from feature_tracking import FeatureTracker
from utils import drawFrameFeatures, updateTrajectoryDrawing, savePly

from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend

from PIL import Image

from UNet.unet_model import UNet
import torch

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Must specify sequence: python run.py seq with seq in [00,01,...,11]')
        sys.exit()
    seq = sys.argv[1]
    if seq not in ['00','01','02','03','04','05','06','07','08','09','10','11']:
        print('Sequence must be in [00,01,...,11]')
        sys.exit()

    # Initialize pre-trained U-Net for inference
    checkpoint_file = 'UNet/checkpoint_epoch7.pth'
    net = UNet(n_channels=3, n_classes=12, bilinear=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(checkpoint_file, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    net.cpu()

    # Initialize the SuperPoint model for SuperPoint keypoints
    weights_path = 'SuperPointPretrainedNetwork/superpoint_v1.pth'
    superpoint = SuperPointFrontend(weights_path, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=True)

    # Initialize the GFTT detector for GFTT keypoints
    detector = cv2.GFTTDetector_create()

    # Initialize the tracker to find matches
    tracker = FeatureTracker()

    # Initialize reader for Kitti images, poses and camera calibration
    dataset_reader = DatasetReaderKITTI("../../data/", seq)
    K = dataset_reader.readCameraMatrix()

    kitti_positions, track_positions_gftt, track_positions_sp = [], [], []
    prev_frame_BGR = dataset_reader.readFrame(0)
    prev_points_gftt, prev_points_sp = np.empty(0), np.empty(0)
    camera_rot_gftt, camera_rot_sp = np.eye(3), np.eye(3)
    camera_pos_gftt, camera_pos_sp = np.zeros((3,1)), np.zeros((3,1))

    #plt.show()

    # Process next frames
    N = dataset_reader.number_images()
    N=8

    for frame_no in range(1, N):
        print(f"Processing frame {frame_no}")
        curr_frame_BGR = dataset_reader.readFrame(frame_no, "General")
        curr_frame = cv2.cvtColor(curr_frame_BGR, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.cvtColor(prev_frame_BGR, cv2.COLOR_BGR2GRAY)

        curr_frame_UNet = dataset_reader.readFrame(frame_no, "U-Net")
        img_pred = net.forward(curr_frame_UNet) # 1x12xHxW
        img_pred = img_pred.detach().numpy()
        img_pred = img_pred.squeeze(0) # 12xHxW
        img_pred = np.transpose(img_pred, (1, 2, 0))
        img_pred = np.argmax(img_pred, 2)

        # Detection with GFTT
        prev_points_gftt = detector.detect(prev_frame) # tuple of cv2.KeyPoint
        prev_points_gftt = cv2.KeyPoint_convert(sorted(prev_points_gftt, key = lambda p: p.response, reverse=True)) # numpy.ndarray of numpy.ndarray

        # Detection with SuperPoint
        prev_points_sp, descriptors, heatmap = superpoint.run(prev_frame.astype(np.float32) / 255.0)
        prev_points_sp = np.float32(prev_points_sp.T[:,:2])

        # Feature tracking (optical flow)
        prev_points_gftt, curr_points_gftt = tracker.trackFeatures(prev_frame, curr_frame, prev_points_gftt, removeOutliers=True)
        prev_points_sp, curr_points_sp = tracker.trackFeatures(prev_frame, curr_frame, prev_points_sp, removeOutliers=True)

        # Essential matrix, pose estimation
        E_gftt, mask_gftt = cv2.findEssentialMat(curr_points_gftt, prev_points_gftt, K, cv2.RANSAC, 0.99, 1.0, None)
        prev_points_gftt = np.array([pt for (idx, pt) in enumerate(prev_points_gftt) if mask_gftt[idx] == 1])
        curr_points_gftt = np.array([pt for (idx, pt) in enumerate(curr_points_gftt) if mask_gftt[idx] == 1])
        _, R_gftt, T_gftt, _ = cv2.recoverPose(E_gftt, curr_points_gftt, prev_points_gftt, K)

        E_sp, mask_sp = cv2.findEssentialMat(curr_points_sp, prev_points_sp, K, cv2.RANSAC, 0.99, 1.0, None)
        prev_points_sp = np.array([pt for (idx, pt) in enumerate(prev_points_sp) if mask_sp[idx] == 1])
        curr_points_sp = np.array([pt for (idx, pt) in enumerate(curr_points_sp) if mask_sp[idx] == 1])
        _, R_sp, T_sp, _ = cv2.recoverPose(E_sp, curr_points_sp, prev_points_sp, K)

        # Read ground truth translation T and absolute scale for computing trajectory
        kitti_pos, kitti_scale = dataset_reader.readGroundTruthPosition(frame_no)
        if kitti_scale <= 0.1:
            continue

        # Calculate new camera positions
        camera_pos_gftt = camera_pos_gftt + kitti_scale * camera_rot_gftt.dot(T_gftt)
        camera_rot_gftt = R_gftt.dot(camera_rot_gftt)

        camera_pos_sp = camera_pos_sp + kitti_scale * camera_rot_sp.dot(T_sp)
        camera_rot_sp = R_sp.dot(camera_rot_sp)

        # Accumulate successive positions of camera for ground truth and both GFTT and SuperPoint methods
        kitti_positions.append(kitti_pos)
        track_positions_gftt.append(camera_pos_gftt)
        track_positions_sp.append(camera_pos_sp)

        # Live plotting of trajectories and keypoints detection
        updateTrajectoryDrawing(np.array(track_positions_gftt), np.array(track_positions_sp), np.array(kitti_positions))
        drawFrameFeatures(curr_frame, prev_points_gftt, curr_points_gftt, prev_points_sp, curr_points_sp, frame_no)

        if cv2.waitKey(1) == ord('q'):
            break
            
        prev_points_gftt, prev_points_sp, prev_frame_BGR = curr_points_gftt, curr_points_sp, curr_frame_BGR

    print("Plotting full trajectories...")
    plt.cla()
    plt.plot(np.array(track_positions_gftt)[:,0], np.array(track_positions_gftt)[:,2], c='blue', label="Tracking GFTT")
    plt.plot(np.array(track_positions_sp)[:,0], np.array(track_positions_sp)[:,2], c='red', label="Tracking SP")
    plt.plot(np.array(kitti_positions)[:,0], np.array(kitti_positions)[:,2], c='green', label="Ground truth")
    plt.title("Trajectories")
    plt.legend()
    plt.draw()
    plt.savefig("../results/trajectories_"+seq+".png")

    print("MSE results...")
    track_positions_gftt = np.array(np.squeeze(track_positions_gftt))
    track_positions_sp = np.array(np.squeeze(track_positions_sp))
    kitti_positions = np.array(kitti_positions)
    MSE_gftt = np.mean(np.sum((kitti_positions[:,:2] - track_positions_gftt[:,:2])**2, axis=1))
    MSE_sp = np.mean(np.sum((kitti_positions[:,:2] - track_positions_sp[:,:2])**2, axis=1))
    print(f"MSE for GFTT: {MSE_gftt}")
    print(f"MSE for SP: {MSE_sp}")
    with open("../results/MSE_"+seq+".txt", 'w') as output:
        output.write(str(MSE_gftt))
        output.write("\n")
        output.write(str(MSE_sp))
        
   # cv2.destroyAllWindows()
    print("Done")
