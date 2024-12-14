import numpy as np

def calculate_reprojection_errors(essential_matrix, keypoints1, keypoints2, camera_matrix):
    """
    Calculate reprojection errors for matched keypoints using the essential matrix.
    
    Parameters:
    -----------
    essential_matrix : numpy.ndarray
        3x3 Essential matrix describing the epipolar geometry
    keypoints1 : numpy.ndarray
        Array of first set of keypoints (n x 2)
    keypoints2 : numpy.ndarray
        Array of second set of keypoints (n x 2)
    camera_matrix : numpy.ndarray
        3x3 Camera intrinsic matrix
    
    Returns:
    --------
    reprojection_errors : numpy.ndarray
        Array of reprojection errors for each point pair
    mean_error : float
        Mean reprojection error
    """
    # Ensure inputs are numpy arrays
    essential_matrix = np.array(essential_matrix)
    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)
    camera_matrix = np.array(camera_matrix)
    
    # Validate input shapes
    assert keypoints1.shape == keypoints2.shape, "Keypoint arrays must have the same shape"
    assert keypoints1.shape[1] == 2, "Keypoints must be 2D (x, y)"
    assert essential_matrix.shape == (3, 3), "Essential matrix must be 3x3"
    assert camera_matrix.shape == (3, 3), "Camera matrix must be 3x3"
    
    # Normalize keypoints using camera matrix
    def normalize_points(points):
        # Convert to homogeneous coordinates
        homo_points = np.column_stack((points, np.ones(points.shape[0])))
        
        # Apply inverse of camera matrix to normalize
        inv_camera_matrix = np.linalg.inv(camera_matrix)
        normalized_points = (inv_camera_matrix @ homo_points.T).T
        
        return normalized_points[:, :2]
    
    # Normalize keypoints
    norm_points1 = normalize_points(keypoints1)
    norm_points2 = normalize_points(keypoints2)
    
    # Calculate fundamental matrix (F = inv(K2)^T * E * inv(K1))
    fundamental_matrix = np.linalg.inv(camera_matrix).T @ essential_matrix @ np.linalg.inv(camera_matrix)
    
    # Reprojection error calculation for each point
    reprojection_errors = []
    for pt1, pt2 in zip(norm_points1, norm_points2):
        # Compute point to line distance for epipolar constraint
        # Line equation: ax + by + c = 0
        line2 = fundamental_matrix @ np.append(pt1, 1)
        a, b, c = line2
        
        # Distance calculation
        distance = np.abs(a * pt2[0] + b * pt2[1] + c) / np.sqrt(a**2 + b**2)
        reprojection_errors.append(distance)
    
    # Convert to numpy array
    reprojection_errors = np.array(reprojection_errors)
    
    return reprojection_errors, np.mean(reprojection_errors)