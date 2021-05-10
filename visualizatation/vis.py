import os
import cv2
import tqdm
import argparse
import numpy as np
import open3d as o3d
from mmcv import Config
from mmcv.runner import obj_from_dict

from lib.datasets import dataset
from lib.utils import object3d_kitti
from lib.datasets import build_dataloader
from lib.datasets.kitti import kitti_dataset
from lib.utils.box_utils import boxes_to_corners_3d
from lib.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

__all__ = {
    'DatasetTemplate': dataset,
    'KittiDataset': kitti_dataset,
}


def parse_args():
    parser = argparse.ArgumentParser(description='visualizatation config')
    parser.add_argument('--cfg', default='../train_info/20210226-113025/car_cfg_FFPS.py',
                        help='config file path')
    parser.add_argument('--eval_res_path', default='../train_info/20210226-113025/checkpoint_92/results/',
                        help='evaluation result path')
    parser.add_argument('--cg_point_path', default='../train_info/20210226-113025/checkpoint_92/candidate_points/',
                        help='evaluation result path')
    parser.add_argument('--img_dump_path', default='../train_info/20210226-113025/checkpoint_92/vis/pred_with_keys/',
                        help='the save path of visualization image')

    args = parser.parse_args()
    return args


def vis_point(pcd):
    """
    visualize points
    Args:
        pcd: ([N, 4])
    Returns:
        None
    """
    # o3d.visualization.draw_geometries([pcd])
    pcd.paint_uniform_color([1, 1, 1])
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Car')
    vis.get_render_option().point_size = 1
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    vis.get_view_control().set_front([-0.99030783944282885, -0.026955850580090156, 0.13624890919776253])
    vis.get_view_control().set_lookat([34.682845105205296, 2.1901744029098298, -1.7519006546743743])
    vis.get_view_control().set_up([0.13617305134165394, 0.0046386846820924621, 0.99067420613071555])
    vis.get_view_control().set_zoom(0.38)

    vis.run()
    vis.destroy_window()


def vis_boxes(pts, boxes, output_path, window_name, keypoints=None):
    """
    Args:
        pts: ([N, 4])
        boxes: (gt_num, 8)
        output_path:
        window_name:
        keypoints:
    Returns:
        None
    """
    gt_num = boxes.shape[0]
    num_points = pts.shape[0]
    pcd_colors = np.ones([num_points, 3], dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])

    # points info
    box_idxs_of_point = points_in_boxes_cpu(pts[:, :3], boxes[:, :7])  # (gt_num, N)
    box_idxs_of_point = box_idxs_of_point.transpose()  # (N, gt_num)
    box_fg_mask = np.sum(box_idxs_of_point, axis=1)
    box_fg_mask = box_fg_mask > 0
    pcd_colors[box_fg_mask, :] = [1.0, 1.0, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.get_render_option().point_size = 1

    vis.add_geometry(pcd)

    # draw keypoints
    if keypoints is not None:
        pcd_num = keypoints.shape[0]
        pcd_keys_color = [[1., 0., 0.] for i in range(pcd_num)]

        pcd_keys = o3d.geometry.PointCloud()
        pcd_keys.points = o3d.utility.Vector3dVector(keypoints[:, :3])
        pcd_keys.colors = o3d.utility.Vector3dVector(pcd_keys_color)

        vis.get_render_option().point_size = 2
        vis.add_geometry(pcd_keys)

    # boxes info
    corners3d = boxes_to_corners_3d(boxes[:, :7])
    line_color = [[0., 1., 0.] for i in range(len(lines))]
    for i in range(gt_num):
        tmp_corners = corners3d[i]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(tmp_corners),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(line_color)
        vis.add_geometry(line_set)

        # orientation
        orientation_bias = (tmp_corners[3, :] + tmp_corners[1, :] - 2 * tmp_corners[2, :]) / 2
        orientation = tmp_corners[1:3, :] + orientation_bias
        orientation_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(orientation),
            lines=o3d.utility.Vector2iVector([[1, 0]]),
        )
        orientation_set.colors = o3d.utility.Vector3dVector([[0.0, 1., 0.]])
        vis.add_geometry(orientation_set)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    vis.get_view_control().set_front([-0.99030783944282885, -0.026955850580090156, 0.13624890919776253])
    vis.get_view_control().set_lookat([34.682845105205296, 2.1901744029098298, -3.2082727163913023])
    vis.get_view_control().set_up([0.13617305134165394, 0.0046386846820924621, 0.99067420613071555])
    vis.get_view_control().set_zoom(0.36)

    # tmp_img = vis.capture_screen_float_buffer(False)
    # plt.imsave(output_path, np.asarray(tmp_img), dpi=1)
    # plt.imshow(np.asarray(tmp_img))
    # plt.show()
    vis.run()
    vis.capture_screen_image(output_path)
    vis.destroy_window()


def vis_prediction(pts, gt_boxes, pred_boxes, win_name, output_path):
    """

    Args:
        pts:
        gt_boxes:
        pred_boxes:
        win_name:
        output_path:
    Returns:

    """

    def draw_boxes(vis, boxes, boxes_color=None):
        if boxes_color is None:
            boxes_color = [0., 1., 0.]

        gt_num = boxes.shape[0]
        # boxes info
        corners3d = boxes_to_corners_3d(boxes[:, :7])
        line_color = [boxes_color for i in range(len(lines))]
        for i in range(gt_num):
            tmp_corners = corners3d[i]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(tmp_corners),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(line_color)
            vis.add_geometry(line_set)

            # orientation
            orientation_bias = (tmp_corners[3, :] + tmp_corners[1, :] - 2 * tmp_corners[2, :]) / 2
            orientation = tmp_corners[1:3, :] + orientation_bias
            orientation_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(orientation),
                lines=o3d.utility.Vector2iVector([[1, 0]]),
            )
            orientation_set.colors = o3d.utility.Vector3dVector([boxes_color])
            vis.add_geometry(orientation_set)

    num_points = pts.shape[0]
    pcd_colors = np.ones([num_points, 3], dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])

    # points info
    box_idxs_of_point = points_in_boxes_cpu(pts[:, :3], gt_boxes[:, :7])  # (gt_num, N)
    box_idxs_of_point = box_idxs_of_point.transpose()  # (N, gt_num)
    box_fg_mask = np.sum(box_idxs_of_point, axis=1)
    box_fg_mask = box_fg_mask > 0
    pcd_colors[box_fg_mask, :] = [1.0, 1.0, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=win_name)
    vis.get_render_option().point_size = 1

    vis.add_geometry(pcd)

    # boxes information
    draw_boxes(vis, gt_boxes, [0., 1., 0.])
    draw_boxes(vis, pred_boxes, [1., 0., 0.])

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    vis.get_view_control().set_front([-0.99030783944282885, -0.026955850580090156, 0.13624890919776253])
    vis.get_view_control().set_lookat([34.682845105205296, 2.1901744029098298, -3.2082727163913023])
    vis.get_view_control().set_up([0.13617305134165394, 0.0046386846820924621, 0.99067420613071555])
    vis.get_view_control().set_zoom(0.36)

    vis.run()
    vis.capture_screen_image(output_path)
    vis.destroy_window()


def vis_img(img, gt_boxes, pred_boxes, calib, output_path, cg_pts=None):
    # ------- boxes to img_corners -------
    gt_corners_lidar = boxes_to_corners_3d(gt_boxes[:, :7])  # (gt_num, 8, 3)
    pred_corners_lidar = boxes_to_corners_3d(pred_boxes[:, :7])  # (pred_num, 8, 3)

    # gt boxes
    gt_num = gt_boxes.shape[0]
    for i in range(gt_num):
        temp_corners_lidar = gt_corners_lidar[i]
        temp_corners_img, _ = calib.lidar_to_img(temp_corners_lidar)  # (8, 2)
        for line in lines:
            cv2.line(img, tuple(temp_corners_img[line[0], :].astype(int)),
                     tuple(temp_corners_img[line[1], :].astype(int)), (0, 0, 255), thickness=1)

    # pred boxes
    pred_num = pred_boxes.shape[0]
    for i in range(pred_num):
        temp_corners_lidar = pred_corners_lidar[i]
        temp_corners_img, _ = calib.lidar_to_img(temp_corners_lidar)
        for line in lines:
            cv2.line(img, tuple(temp_corners_img[line[0], :].astype(int)),
                     tuple(temp_corners_img[line[1], :].astype(int)), (0, 255, 0), thickness=1)

    if cg_pts is not None:
        cg_img_pts, _ = calib.lidar_to_img(cg_pts)
        for pts in cg_img_pts:
            cv2.circle(img, tuple(pts.astype(np.int)), 3, (255, 0, 0), -1)

    cv2.imwrite(output_path, img)


def parse_prediction(pred_file, calib):
    pred_obj_list = object3d_kitti.get_objects_from_label(pred_file)
    if len(pred_obj_list) == 0:
        return np.zeros([0, 7])

    loc = np.concatenate([obj.loc.reshape(1, 3) for obj in pred_obj_list], axis=0)
    dims = np.array([[obj.l, obj.h, obj.w] for obj in pred_obj_list])  # lhw(camera) format
    ry = np.array([obj.ry for obj in pred_obj_list])

    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    pred_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + ry[..., np.newaxis])], axis=1)
    return pred_boxes_lidar


def draw_all(args, dataset):
    pbar = tqdm.trange(dataset.__len__())
    os.makedirs(args.img_dump_path, exist_ok=True)
    for i in pbar:
        per_data = dataset[i]
        pts = per_data['points']
        calib = per_data['calib']
        gt_boxes = per_data['gt_boxes']
        output_img_path = os.path.join(args.img_dump_path, '%s.png' % dataset.sample_id_list[i])

        cg_point_path = os.path.join(args.cg_point_path, '%s.txt' % dataset.sample_id_list[i])
        cg_pts = np.loadtxt(cg_point_path)

        # draw pts
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        # vis_point(pcd)
        # vis_boxes(pts, gt_boxes, output_img_path, dataset.sample_id_list[i])

        # draw pred boxes with gt boxes in LiDAR
        # pred_file = os.path.join(args.eval_res_path, '%s.txt' % dataset.sample_id_list[i])
        # pred_boxes = parse_prediction(pred_file, calib)
        # vis_prediction(pts, gt_boxes, pred_boxes, dataset.sample_id_list[i], output_img_path)

        # draw pred boxes with gt boxes in IMAGE
        image = dataset.get_image(dataset.sample_id_list[i])
        pred_file = os.path.join(args.eval_res_path, '%s.txt' % dataset.sample_id_list[i])
        pred_boxes = parse_prediction(pred_file, calib)
        vis_img(image, gt_boxes, pred_boxes, calib, output_img_path, cg_pts)


def draw_single(args, dataset, idx):
    """
    draw single point clouds with boxes
    Args:
        args: arguments
        dataset: KittiDataset class
        idx: idx or name of data
    Returns:
        None:
    """
    img_name = '%06d' % idx
    i = dataset.sample_id_list.index(img_name)

    per_data = dataset[i]

    pts = per_data['points']
    calib = per_data['calib']
    gt_boxes = per_data['gt_boxes']

    output_img_path = os.path.join(args.img_dump_path, '%s.png' % img_name)

    keypoints_path = os.path.join(args.eval_res_path, 'candidate_points', '%s.txt' % img_name)
    if os.path.exists(keypoints_path):
        keypoints = np.loadtxt(keypoints_path)
    else:
        keypoints = None

    # draw pts
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    # vis_point(pcd)
    vis_boxes(pts, gt_boxes, output_img_path, img_name, keypoints)

    # draw pred boxes with gt boxes in LiDAR
    # pred_file = os.path.join(args.eval_res_path, '%s.txt' % dataset.sample_id_list[i])
    # pred_boxes = parse_prediction(pred_file, calib)
    # vis_prediction(pts, gt_boxes, pred_boxes, dataset.sample_id_list[i], output_img_path)

    # draw pred boxes with gt boxes in IMAGE
    # image = dataset.get_image(dataset.sample_id_list[i])
    # pred_file = os.path.join(args.eval_res_path, '%s.txt' % dataset.sample_id_list[i])
    # pred_boxes = parse_prediction(pred_file, calib)
    # vis_img(image, gt_boxes, pred_boxes, calib, output_img_path)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    cfg.DATASET.training = False

    dataset = obj_from_dict(cfg.DATASET, __all__[cfg.DATASET.type])

    draw_all(args, dataset)

    # for id in img_list:
    #     draw_single(args, dataset, id)


if __name__ == '__main__':
    lines = [[0, 1],
             [0, 3],
             [0, 4],
             [1, 2],
             [1, 5],
             [2, 3],
             [2, 6],
             [3, 7],
             [4, 5],
             [4, 7],
             [5, 6],
             [6, 7],
             [0, 5],
             [1, 4],
             [3, 6],
             [2, 7]]

    img_list = [8, 21, 31, 39, 47, 52, 104, 107, 140, 152, 159, 161, 169, 173, 181,
                191, 192, 201, 213, 236, 246, 248, 252, 266, 273, 301]

    main()
