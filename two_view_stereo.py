import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from numpy.linalg import norm, inv
from numpy import cross
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
from cv2 import warpPerspective
import open3d as o3d


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32(
        [[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(
        h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(
        h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - \
        v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    """Student Code Starts"""
    h_i = K_i_corr @ R_irect @ inv(K_i)
    h_j = K_j_corr @ R_jrect @ inv(K_j)
    rgb_i_rect = cv2.warpPerspective(rgb_i, h_i, (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, h_j, (w_max, h_max))
    """Student Code Ends"""

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_iw, T_iw, R_jw, T_jw):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_iw, R_jw : [3,3]
    T_iw, T_jw : [3,1]
        p_i = R_iw @ p_w + T_iw
        p_j = R_jw @ p_w + T_jw
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ij @ p_j + T_ij, B is the baseline
    """

    """Student Code Starts"""
    R_ij = R_iw @ R_jw.T
    T_ij = - R_iw @ R_jw.T @ T_jw + T_iw
    B = norm(T_ij)
    """Student Code Ends"""

    return R_ij, T_ij, B


def compute_rectification_R(T_ij):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ij : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ij.squeeze(-1) / (T_ij.squeeze(-1)[1] + EPS)

    """Student Code Starts"""
    norm_ei = e_i / norm(e_i)
    r1 = cross(norm_ei, np.array([0., 0., 1.]))
    norm_r1 = r1 / norm(r1)
    r3 = cross(norm_r1, norm_ei)
    R_irect = np.vstack([norm_r1, norm_ei, r3])
    """Student Code Ends"""

    return R_irect


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    src = np.expand_dims(src, axis=1)
    dst = np.expand_dims(dst, axis=0)
    diff = src - dst
    ssd = np.sum(np.square(diff), axis=(2, 3))
    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    src = np.expand_dims(src, axis=1)
    dst = np.expand_dims(dst, axis=0)
    sad = np.sum(np.abs(src - dst), axis=(2, 3))
    """Student Code Ends"""

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    src = np.expand_dims(src, axis=1)
    dst = np.expand_dims(dst, axis=0)

    src_std = np.std(src, axis=2, keepdims=True)
    dst_std = np.std(dst, axis=2, keepdims=True)

    src_mean = np.mean(src, axis=2, keepdims=True)
    dst_mean = np.mean(dst, axis=2, keepdims=True)

    zncc = np.sum((src - src_mean) * (dst - dst_mean) /
                  (src_std * dst_std + EPS), axis=(2, 3))
    """Student Code Ends"""

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    """Student Code Starts"""
    # If kernel size is 1, add a new dimension to make it 4D
    if k_size == 1:
        patch_buffer = np.expand_dims(image, axis=2)
    # If kernel size is not 1, extract patches and store in patch_buffer
    if k_size != 1:
        height, width, _ = image.shape
        step_size = (k_size - 1) // 2
        pad_width = [(step_size, step_size), (step_size, step_size), (0, 0)]
        # pad image
        pad_image = np.pad(image, pad_width, mode='constant')
        # create an empty array to store the patches
        patch_buffer = np.zeros((height, width, k_size**2, 3))
        # extract and store
        for x in range(height):
            for y in range(width):
                patch_buffer[x, y] = pad_image[x:x +
                                               k_size, y:y+k_size].reshape(-1, 3)

    """Student Code Starts"""

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func : function, optional
        this is for auto-grader purpose, in grading, we will use our correct implementation 
        of the image2path function to exclude double count for errors in image2patch function

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    """Student Code Starts"""
    height, width, _ = rgb_i.shape

    normalized_i = rgb_i.astype(float) / 255.0
    normalized_j = rgb_j.astype(float) / 255.0

    patches_i = image2patch(normalized_i, k_size)
    patches_j = image2patch(normalized_j, k_size)

    vi = np.arange(height)
    vj = vi

    diff = np.outer(vi, np.ones_like(vj)) - \
        np.outer(np.ones_like(vi), vj)
    disp_candidates = diff + d0
    valid_disp_mask = disp_candidates > 0.0

    disp_map = np.zeros((height, width))
    lr_consistency_mask = np.zeros((height, width))

    for u in range(width):
        # get buf_i, j
        buf_i = patches_i[:, u]
        buf_j = patches_j[:, u]
        buf = kernel_func(buf_i, buf_j)

        # Set invalid disparities to a large number
        buf[~valid_disp_mask] = buf.max() + 1.0

        for v in range(height):
            # get best match
            best_right = np.argmin(buf[v])
            best_left = np.argmin(buf[:, best_right])
            # populate
            disp_map[v, u] = disp_candidates[v, best_right]
            lr_consistency_mask[v, u] = (best_left == v)
    """Student Code Ends"""

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    """Student Code Starts"""
    height, width = disp_map.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    focal_x = K[0][0]
    focal_y = K[1][1]
    ppoint_x = K[0][2]
    ppoint_y = K[1][2]

    # compute depth map from disparity map
    dep_map = (focal_y * B) / disp_map

    # 3D point cloud in camera coord
    Xc = (u - ppoint_x) * dep_map / focal_x
    Yc = (v - ppoint_y) * dep_map / focal_y
    Zc = dep_map

    # Stack
    xyz_cam = np.stack((Xc, Yc, Zc), axis=-1)

    """Student Code Ends"""

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_wc,
    T_wc,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(
        mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(
        nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""
    pcl_world = pcl_cam @ R_wc - np.transpose(T_wc) @ R_wc
    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_iw, T_iw = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_jw, T_jw = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ij, T_ij, B = compute_right2left_transformation(R_iw, T_iw, R_jw, T_jw)
    assert T_ij[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ij)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ij,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1,
                                      1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(
    ), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_iw,
        T_wc=R_irect @ T_iw,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
