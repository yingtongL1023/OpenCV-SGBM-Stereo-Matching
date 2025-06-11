import numpy as np
import cv2 as cv
import json
import os

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def load_params(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        K = np.array(data['K'], dtype=np.float64).reshape(3, 3)
        D = np.array(data['D'], dtype=np.float64)
        R = np.array(data['R'], dtype=np.float64).reshape(3, 3)
        P = np.array(data['P'], dtype=np.float64).reshape(3, 4)
        return K, D, R, P

def main():
    # === Load stereo images ===
    imgL = cv.imread('/path/to/the/left/image/left.png')
    imgR = cv.imread('/path/to/the/right/image/right.png')

    # === Load calibration parameters ===
    K1, D1, R1_raw, P1 = load_params('/path/to/the/left/camera/info/left_camera_info.json')
    K2, D2, R2_raw, P2 = load_params('/path/to/the/right/camera/info/right_camera_info.json')

    # === Estimate R and T from P1 and P2 ===
    fx = P1[0, 0]
    Tx = P2[0, 3] / -fx
    R = np.eye(3, dtype=np.float64)
    T = np.array([Tx, 0, 0], dtype=np.float64)
    print(f"Estimated baseline T = {T}")

    # === Get image size ===
    h, w = imgL.shape[:2]

    # === Stereo Rectification ===
    R1, R2, P1_rect, P2_rect, Q, _, _ = cv.stereoRectify(K1, D1, K2, D2, (w, h), R, T, flags=0)

    # === Rectify images ===
    map1x, map1y = cv.initUndistortRectifyMap(K1, D1, R1, P1_rect, (w, h), cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(K2, D2, R2, P2_rect, (w, h), cv.CV_32FC1)
    imgL_rect = cv.remap(imgL, map1x, map1y, cv.INTER_LINEAR)
    imgR_rect = cv.remap(imgR, map2x, map2y, cv.INTER_LINEAR)

    # === Stereo Matching ===
    stereo = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=9,
        P1=8 * 3 * 9 ** 2,
        P2=32 * 3 * 9 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disp = stereo.compute(imgL_rect, imgR_rect).astype(np.float32) / 16.0

    # === Reproject to 3D ===
    points = cv.reprojectImageTo3D(disp, Q)
    
    colors = cv.cvtColor(imgL_rect, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()

    # === Remove NaNs and Infs ===
    points_filtered = points[mask]
    colors_filtered = colors[mask]
    print(points_filtered.shape)
    valid_mask = np.isfinite(points_filtered).all(axis=1)
    out_points = points_filtered[valid_mask]
    out_colors = colors_filtered[valid_mask]

    # === Save point cloud ===
    output_folder = '/output/point_cloud/folder/path'
    os.makedirs(output_folder, exist_ok=True)
    out_fn = os.path.join(output_folder, 'point_cloud_name.ply')
    write_ply(out_fn, out_points, out_colors)
    print(f'{out_fn} saved.')

    # === Show results ===
    cv.imshow('Left Image (Rectified)', imgL_rect)
    cv.imshow('Disparity', (disp - disp.min()) / (disp.max() - disp.min()))
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
