import torch
import cv2
import numpy as np
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend

# Initialize the SuperPoint model
weights_path = 'SuperPointPretrainedNetwork/superpoint_v1.pth'
superpoint = SuperPointFrontend(weights_path, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=True)

def process_image(image_path):
    # Read and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be read")
    image = image.astype(np.float32) / 255.0

    # Extract keypoints and descriptors
    keypoints, descriptors, heatmap = superpoint.run(image)
    return keypoints, descriptors, heatmap

# Example: Process a single KITTI image
image_path = 'path_to_kitti_image.png'
image_path = "../../data/data_odometry_gray/dataset/sequences/00/image_0/000000.png"
keypoints, descriptors, heatmap = process_image(image_path)
print(type(keypoints))

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# Visualize keypoints
for pt in keypoints.T:
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
cv2.imshow("Keypoints", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

