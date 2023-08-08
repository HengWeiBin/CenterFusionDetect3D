from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import cv2
import os
from collections import defaultdict
import pycocotools.coco as coco
import torch
from timeit import default_timer as timer
from torchvision.transforms import \
    ColorJitter, Normalize, Lambda, Compose, RandomOrder

from utils.image import getAffineTransform, affineTransform, lightingAug
# from utils.image import flip, color_aug
# from utils.image import affine_transform
# from utils.image import gaussian_radius, draw_umich_gaussian, gaussian2D
# from utils.pointcloud import map_pointcloud_to_image, pc_dep_to_hm
# from utils.pointcloud import RadarPointCloudWithVelocity as RadarPointCloud
# from nuscenes.utils.data_classes import Box
# from pyquaternion import Quaternion
# from nuscenes.utils.geometry_utils import view_points
# from utils.ddd_utils import compute_box_3d, project_to_image, draw_box_3d
# from utils.ddd_utils import comput_corners_3d, alpha2rot_y, get_pc_hm

# def get_dist_thresh(calib, ct, dim, alpha):
#     rotation_y = alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0])
#     corners_3d = comput_corners_3d(dim, rotation_y)
#     dist_thresh = max(corners_3d[:, 2]) - min(corners_3d[:, 2]) / 2.0
#     return dist_thresh


class GenericDataset(torch.utils.data.Dataset):
    # default_resolution = None
    # class_name = None
    num_categories = None
    class_ids = None
    max_objs = None
    # rest_focal_length = 1200
    # num_joints = 17
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                [11, 12], [13, 14], [15, 16]]
    edges = [[0, 1], [0, 2], [1, 3], [2, 4],
             [4, 6], [3, 5], [5, 6],
             [5, 7], [7, 9], [6, 8], [8, 10],
             [6, 12], [5, 11], [11, 12],
             [12, 14], [14, 16], [11, 13], [13, 15]]
    ignore_val = 1

    # change these vectors to actual mean and std to normalize
    pc_mean = np.zeros((18, 1))
    pc_std = np.ones((18, 1))
    img_ind = 0

    def __init__(self, config=None, split=None, ann_path=None, img_dir=None, device=None):
        super(GenericDataset, self).__init__()
        if config is not None and split is not None:
            self.split = split
            self.config = config
            self.enable_meta =  (config.TEST.OFFICIAL_EVAL and
                split in ["val", "mini_val", "test"]) or config.EVAL

        if ann_path is not None and img_dir is not None:
            self.coco = coco.COCO(ann_path)
            self.images = self.coco.getImgIds()
            self.img_dir = img_dir

        self.device = device if device is not None else torch.device('cpu')

        # initiaize the color augmentation
        mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self.colorAugmentor = Compose([
            RandomOrder([
                ColorJitter(brightness=0.4),
                ColorJitter(contrast=0.4),
                ColorJitter(saturation=0.4),]
            ),
            Lambda(lightingAug),
            Normalize(mean, std)])

    def __getitem__(self, index):
        # t1 = timer()
        img, anns, img_info, img_path = \
            self.loadImageAnnotation(self.images[index], self.img_dir)
        # t2 = timer()
        center = np.array([img_info['width'] / 2., img_info['height'] / 2.], dtype=np.float32)
        if self.config.DATASET.MAX_CROP:
            scale = max(img_info['height'], img_info['width']) * 1.0
        else:
            scale = np.array([img_info['width'], img_info['height']], dtype=np.float32)

        # data augmentation for training set
        scaleFactor, rotateFactor, isFliped = 1, 0, True
        if 'train' in self.split:
            # t6 = timer()
            center, scaleFactor, rotateFactor = \
                self.getAugmentParam(center, scale, img_info['width'], img_info['height'])
            # t7 = timer()
            scale *= scaleFactor
            if np.random.random() < self.config.DATASET.FLIP:
                isFliped = True
                img = img[:, ::-1, :]
                anns = self.filpAnnotations(anns, img_info['width'])
            # t8 = timer()

        transMatInput = getAffineTransform(
            center, scaleFactor, rotateFactor,
            [self.config.MODEL.INPUT_SIZE[1], self.config.MODEL.INPUT_SIZE[0]])
        # t9 = timer()
        transMatOutput = getAffineTransform(
            center, scaleFactor, rotateFactor,
            [self.config.MODEL.OUTPUT_SIZE[1], self.config.MODEL.OUTPUT_SIZE[0]])
        # t10 = timer()
        item = {
            'image': self.transformInput(img, transMatInput),
            'origin_img': img,
            'img_info': img_info,
            'transMatInput': transMatInput,
            'transMatOutput': transMatOutput,
            'isFliped': isFliped,
            }
        target = {'bboxes': [], 'scores': [], 'classIds': [], 'centers': []}
        # t3 = timer()

        # #  load point cloud data
        # if self.config.DATASET.NUSCENES.RADAR_PC:
        #     pc_2d, pc_N, pc_dep, pc_3d = \
        #         self.loadRadarPointCloud(
        #         img, img_info, trans_input, transMatOutput, isFliped)
            
        #     item.update({'pc_2d': pc_2d,
        #                 'pc_3d': pc_3d,
        #                 'pc_N': pc_N,
        #                 'pc_dep': pc_dep})
        # t4 = timer()

        # init samples
        self._init_ret(item, target) #下次修 val batchsize問題請看這個
        calib = self._get_calib(img_info, img_info['width'], img_info['height'])

        # get velocity transformation matrix
        if "velocity_trans_matrix" in img_info:
            velocity_mat = np.array(
                img_info['velocity_trans_matrix'], dtype=np.float32)
        else:
            velocity_mat = np.eye(4)

        num_objs = min(len(anns), self.max_objs)
        for k in range(num_objs):
            ann = anns[k]
            classId = int(self.class_ids[ann['category_id']])
            if classId > self.num_categories or classId <= -999:
                continue
            bbox, bbox_amodal = self.transformBbox(ann['bbox'], transMatOutput)
            if classId <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
                self._mask_ignore_or_crowd(item, classId, bbox)
                continue
            self._add_instance(
                item, target, k, classId, bbox, bbox_amodal, ann, transMatOutput, aug_s,
                calib, pre_cts, track_ids)

        if self.opt.debug > 0 or self.enable_meta:
            target = self._format_gt_det(target)
            meta = {
                'c': center,
                's': scale,
                'gt_det': target,
                'img_id': img_info['id'],
                'img_path': img_path,
                'calib': calib,
                'img_width': img_info['width'],
                'img_height': img_info['height'],
                'flipped': isFliped,
                'velocity_mat': velocity_mat
                }
            item['meta'] = meta
            
        item['calib'] = calib
        return item

    def get_default_calib(self, width, height):
        calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                          [0, self.rest_focal_length, height / 2, 0],
                          [0, 0, 1, 0]])
        return calib

    def loadImageAnnotation(self, img_id, img_dir): 
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        img = cv2.imread(img_path)
        return img, anns, img_info, img_path

    def getBorder(self, border:int, size:int) -> int:
        '''
        This function returns the smallest multiple of the border.

        Args:
            border (int): The border.
            size (int): The size of the region that the border contains.

        Returns:
            int: The smallest multiple of the border.
        '''
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def getAugmentParam(self, center, scale, width, height):
        '''
        Generate random augmentation parameters

        Args:
            center(list of int): The center of the bounding box before augmentation.
            scale(float): The scale of the bounding box before augmentation.
            width(int): The width of the image.
            height(int): The height of the image.

        Returns:
            list of int
                The center of the bounding box after augmentation.
            float
                The scale factor of the bounding box after augmentation.
            float
                The rotation factor of the bounding box after augmentation.
        '''
        if self.config.DATASET.RANDOM_CROP:
            scaleFactor = np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self.getBorder(128, width)
            h_border = self.getBorder(128, height)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)

        else:
            scaleFactor = self.config.DATASET.SCALE
            shiftFactor = self.config.DATASET.SHIFT
            scaleFactor = np.clip(
                np.random.randn() * scaleFactor + 1,
                1 - scaleFactor,
                1 + scaleFactor)
            center[0] += scale * np.clip(
                np.random.randn() * shiftFactor,
                -2 * shiftFactor,
                2 * shiftFactor)
            center[1] += scale * np.clip(
                np.random.randn() * shiftFactor,
                -2 * shiftFactor,
                2 * shiftFactor)

        if np.random.random() < self.config.DATASET.ROTATE:
            rotateFactor = self.config.DATASET.ROTATE
            rotateFactor = np.clip(
                np.random.randn() * rotateFactor,
                -rotateFactor * 2,
                rotateFactor * 2)
        else:
            rotateFactor = 0

        return center, scaleFactor, rotateFactor

    def filpAnnotations(self, anns, width):
        '''
        This function flips the annotations horizontally.
        It does this by flipping the bounding boxes, the rotation angles,
        the amodel centers, and the velocities

        Args:
            anns (list): A list of annotations.
            width (int): The width of the image.

        Returns:
            A list of annotations that have been flipped horizontally.
        '''
        for k in range(len(anns)):
            bbox = anns[k]['bbox']
            anns[k]['bbox'] = [
                width - bbox[0] - 1 - bbox[2], bbox[1], bbox[2], bbox[3]]

            if 'rotation' in self.config.heads and 'alpha' in anns[k]:
                anns[k]['alpha'] = np.pi - anns[k]['alpha'] if anns[k]['alpha'] > 0 \
                    else - np.pi - anns[k]['alpha']

            # TODO: nannotation typo amodel -> amodal
            if 'amodal_offset' in self.config.heads and 'amodel_center' in anns[k]:
                anns[k]['amodel_center'][0] = width - \
                    anns[k]['amodel_center'][0] - 1

            if self.config.LOSS.VELOCITY and 'velocity' in anns[k]:
                anns[k]['velocity'][0] *= -1

        return anns

    def create_pc_pillars(self, img, img_info, pc_2d, pc_3d, inp_trans, out_trans):
        pillar_wh = np.zeros((2, pc_3d.shape[1]))
        boxes_2d = np.zeros((0, 8, 2))
        pillar_dim = self.opt.pillar_dims
        v = np.dot(np.eye(3), np.array([1, 0, 0]))
        ry = -np.arctan2(v[2], v[0])

        for i, center in enumerate(pc_3d[:3, :].T):
            # Create a 3D pillar at pc location for the full-size image
            box_3d = compute_box_3d(
                dim=pillar_dim, location=center, rotation_y=ry)
            box_2d = project_to_image(box_3d, img_info['calib']).T  # [2x8]

            # save the box for debug plots
            if self.opt.debug:
                box_2d_img, m = self._transform_pc(box_2d, inp_trans, self.opt.input_w,
                                                   self.opt.input_h, filter_out=False)
                boxes_2d = np.concatenate(
                    (boxes_2d, np.expand_dims(box_2d_img.T, 0)), 0)

            # transform points
            box_2d_t, m = self._transform_pc(
                box_2d, out_trans, self.opt.output_w, self.opt.output_h)

            if box_2d_t.shape[1] <= 1:
                continue

            # get the bounding box in [xyxy] format
            bbox = [np.min(box_2d_t[0, :]),
                    np.min(box_2d_t[1, :]),
                    np.max(box_2d_t[0, :]),
                    np.max(box_2d_t[1, :])]  # format: xyxy

            # store height and width of the 2D box
            pillar_wh[0, i] = bbox[2] - bbox[0]
            pillar_wh[1, i] = bbox[3] - bbox[1]

            # del box_3d, box_2d, box_2d_t, m, bbox # wayne memory check

        ## DEBUG #################################################################
        if self.opt.debug:
            img_2d = img.copy()
            # img_3d = copy.deepcopy(img)
            img_2d_inp = cv2.warpAffine(img, inp_trans,
                                        (self.opt.input_w, self.opt.input_h),
                                        flags=cv2.INTER_LINEAR)
            img_2d_out = cv2.warpAffine(img, out_trans,
                                        (self.opt.output_w, self.opt.output_h),
                                        flags=cv2.INTER_LINEAR)
            img_3d = cv2.warpAffine(img, inp_trans,
                                    (self.opt.input_w, self.opt.input_h),
                                    flags=cv2.INTER_LINEAR)
            blank_image = 255 * \
                np.ones((self.opt.input_h, self.opt.input_w, 3), np.uint8)
            mask = np.zeros(img_2d.shape[:2], np.uint8)
            overlay = img_2d_inp.copy()
            origin_overlay = img.copy()

            pc_inp, _ = self._transform_pc(
                pc_2d, inp_trans, self.opt.input_w, self.opt.input_h)
            pc_inp = pc_inp[:3, :].T
            pc_out, _ = self._transform_pc(
                pc_2d, out_trans, self.opt.output_w, self.opt.output_h)

            pill_wh_inp = pillar_wh * (self.opt.input_w/self.opt.output_w)
            pill_wh_out = pillar_wh
            pill_wh_ori = pill_wh_inp * 2

            # for i, p in reversed(list(enumerate(pc_inp[:3, :].T))):
            for i in range(len(pc_inp) - 1, -1, -1):
                p = pc_inp[i]
                color = int((p[2].tolist() / 60.0) * 255)
                color = (0, color, 0)

                rect_tl = (np.min(int(p[0]-pill_wh_inp[0, i]/2), 0),
                           np.min(int(p[1]-pill_wh_inp[1, i]), 0))
                rect_br = (np.min(int(p[0]+pill_wh_inp[0, i]/2), 0),
                           int(p[1]))
                cv2.rectangle(img_2d_inp, rect_tl, rect_br,
                              (0, 0, 255), 1, lineType=cv2.LINE_AA)
                img_2d_inp = cv2.circle(
                    img_2d_inp, (int(p[0]), int(p[1])), 3, color, -1)

                # On original-sized image
                rect_tl_ori = (np.min(int(pc_2d[0, i] - pill_wh_ori[0, i] / 2), 0),
                               np.min(int(pc_2d[1, i] - pill_wh_ori[1, i]), 0))
                rect_br_ori = (
                    np.min(int(pc_2d[0, i]+pill_wh_ori[0, i]/2), 0), int(pc_2d[1, i]))
                cv2.rectangle(img_2d, rect_tl_ori, rect_br_ori,
                              (0, 0, 255), 2, lineType=cv2.LINE_AA)
                img_2d = cv2.circle(
                    img_2d, (int(pc_2d[0, i]), int(pc_2d[1, i])), 6, color, -1)

                p2 = pc_out[:3, i].T
                rect_tl2 = (np.min(
                    int(p2[0]-pill_wh_out[0, i]/2), 0), np.min(int(p2[1]-pill_wh_out[1, i]), 0))
                rect_br2 = (
                    np.min(int(p2[0]+pill_wh_out[0, i]/2), 0), int(p2[1]))
                cv2.rectangle(img_2d_out, rect_tl2, rect_br2,
                              (0, 0, 255), 1, lineType=cv2.LINE_AA)
                img_2d_out = cv2.circle(
                    img_2d_out, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

                # on blank image
                cv2.rectangle(blank_image, rect_tl, rect_br,
                              color, -1, lineType=cv2.LINE_AA)
                mask[rect_tl_ori[1]:rect_br_ori[1], rect_tl_ori[0]:rect_br_ori[0]] = color[1]

                # plot 3d pillars
                img_3d = draw_box_3d(img_3d, boxes_2d[i].astype(np.int32), [114, 159, 207],
                                     same_color=False)

                # overlay
                cv2.rectangle(overlay, rect_tl, rect_br,
                              color, -1, lineType=cv2.LINE_AA)
                cv2.rectangle(origin_overlay, rect_tl_ori, rect_br_ori,
                              color, -1, lineType=cv2.LINE_AA)

            cv2.imwrite((self.opt.debug_dir + '/{}pc_pillar_2d_inp.' + self.opt.img_format)
                        .format(self.img_ind), img_2d_inp)
            cv2.imwrite((self.opt.debug_dir + '/{}pc_pillar_2d_ori.' + self.opt.img_format)
                        .format(self.img_ind), img_2d)
            cv2.imwrite((self.opt.debug_dir + '/{}pc_pillar_2d_out.' + self.opt.img_format)
                        .format(self.img_ind), img_2d_out)
            cv2.imwrite((self.opt.debug_dir+'/{}pc_pillar_2d_blank.' + self.opt.img_format)
                        .format(self.img_ind), blank_image)
            cv2.imwrite((self.opt.debug_dir+'/{}pc_pillar_2d_ori_overlay.' + self.opt.img_format)
                        .format(self.img_ind), origin_overlay)
            cv2.imwrite((self.opt.debug_dir+'/{}pc_pillar_2d_overlay.' + self.opt.img_format)
                        .format(self.img_ind), overlay)
            cv2.imwrite((self.opt.debug_dir+'/{}pc_pillar_3d.' + self.opt.img_format)
                        .format(self.img_ind), img_3d)
            cv2.imwrite((self.opt.debug_dir+'/{}pc_pillar_mask.' + self.opt.img_format)
                        .format(self.img_ind), mask)
            cv2.imwrite((self.opt.debug_dir+'/{}img.' + self.opt.img_format)
                        .format(self.img_ind), img)
            self.img_ind += 1
        ## DEBUG #################################################################
        
        return pillar_wh

    def transformInput(self, img, transformMat):
        '''
        Affine transform the image and apply color augmentation
        
        Args:
            img: [HxWx3] image
            transformMat: affine transform matrix

        Returns:
            [CxHxW] transformed image
        '''
        # t1 = timer()
        result = cv2.warpAffine(
            img,
            transformMat,
            (self.config.MODEL.OUTPUT_SIZE[1], self.config.MODEL.OUTPUT_SIZE[0]),
            flags=cv2.INTER_LINEAR)
        # t2 = timer()

        result = result.astype(np.float32) / 255.
        result = result.transpose(2, 0, 1)
        result = torch.from_numpy(result).to(self.device)
        if 'train' in self.split and self.config.DATASET.COLOR_AUG:
            # t3 = timer()
            result = self.colorAugmentor(result)
            # t4 = timer()
        # t5 = timer()
        # print('\n' + f'warpAffine: {(t2 - t1) * 1000:.1f}ms')
        # print(f'color_aug: {(t4 - t3) * 1000:.1f}ms')
        # print(f'other: {(t5 - t4) * 1000:.1f}ms')
        return result

    def _init_ret(self, ret, gt_det):
        max_objs = self.max_objs * self.opt.dense_reg
        ret['hm'] = np.zeros(
            (self.opt.num_classes, self.opt.output_h, self.opt.output_w),
            np.float32)
        ret['ind'] = np.zeros((max_objs), dtype=np.int64)
        ret['cat'] = np.zeros((max_objs), dtype=np.int64)
        ret['mask'] = np.zeros((max_objs), dtype=np.float32)

        if self.opt.pointcloud:
            ret['pc_hm'] = np.zeros(
                (len(self.opt.pc_feat_lvl), self.opt.output_h, self.opt.output_w),
                np.float32)

        regression_head_dims = {
            'reg': 2, 'wh': 2, 'tracking': 2, 'ltrb': 4, 'ltrb_amodal': 4,
            'nuscenes_att': 8, 'velocity': 3, 'hps': self.num_joints * 2,
            'dep': 1, 'dim': 3, 'amodel_offset': 2}

        for head in regression_head_dims:
            if head in self.opt.heads:
                ret[head] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                ret[head + '_mask'] = np.zeros(
                    (max_objs, regression_head_dims[head]), dtype=np.float32)
                gt_det[head] = []
        # if self.opt.pointcloud:
        #     ret['pc_dep_mask'] = np.zeros((max_objs, 1), dtype=np.float32)
        #     ret['pc_dep'] = np.zeros((max_objs, 1), dtype=np.float32)
        #     gt_det['pc_dep'] = []

        if 'hm_hp' in self.opt.heads:
            num_joints = self.num_joints
            ret['hm_hp'] = np.zeros(
                (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)
            ret['hm_hp_mask'] = np.zeros(
                (max_objs * num_joints), dtype=np.float32)
            ret['hp_offset'] = np.zeros(
                (max_objs * num_joints, 2), dtype=np.float32)
            ret['hp_ind'] = np.zeros((max_objs * num_joints), dtype=np.int64)
            ret['hp_offset_mask'] = np.zeros(
                (max_objs * num_joints, 2), dtype=np.float32)
            ret['joint'] = np.zeros((max_objs * num_joints), dtype=np.int64)

        if 'rot' in self.opt.heads:
            ret['rotbin'] = np.zeros((max_objs, 2), dtype=np.int64)
            ret['rotres'] = np.zeros((max_objs, 2), dtype=np.float32)
            ret['rot_mask'] = np.zeros((max_objs), dtype=np.float32)
            gt_det.update({'rot': []})

    def _get_calib(self, img_info, width, height):
        if 'calib' in img_info:
            calib = np.array(img_info['calib'], dtype=np.float32)
        else:
            calib = np.array([[self.rest_focal_length, 0, width / 2, 0],
                              [0, self.rest_focal_length, height / 2, 0],
                              [0, 0, 1, 0]])
        return calib

    def _mask_ignore_or_crowd(self, item, classId, bbox):
        '''
        Mask out specific region(bbox) in heatmap.
        Only single class is masked out if classId is specified.

        Args:
            item(dict): data item
            classId(int): class id
            bbox(array): [x1, y1, x2, y2]

        Returns:
            None
        '''
        ignore_val = 1
        if classId == 0:
            # ignore all classes
            # mask out crowd region
            region = item['hm'][:,
                                int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1]
        else:
            # mask out one specific class
            region = item['hm'][abs(classId) - 1,
                                int(bbox[1]): int(bbox[3]) + 1,
                                int(bbox[0]): int(bbox[2]) + 1]
        np.maximum(region, ignore_val, out=region)

    def transformBbox(self, bbox, transMatOut):
        '''
        Transform bbox according to affine transform matrix.

        Args:
            bbox: [x1, y1, w, h]
            transMatOut: affine transform matrix

        Returns:
            bbox: [x1, y1, x2, y2]
            bbox_amodal: [x1, y1, x2, y2]
        '''
        # convert bbox from [x1, y1, w, h] to [x1, y1, x2, y2]
        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                        dtype=np.float32)

        rect = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                        [bbox[2], bbox[3]], [bbox[2], bbox[1]]])
        for i in range(4):
            rect[i] = affineTransform(rect[i], transMatOut)
        bbox = np.array([rect[:, 0].min(), rect[:, 1].min(),
                         rect[:, 0].max(), rect[:, 1].max()])

        bbox_amodal = bbox.copy()
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.config.MODEL.OUTPUT_SIZE[1] - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.config.MODEL.OUTPUT_SIZE[0] - 1)
        return bbox, bbox_amodal

    def _add_instance(
            self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
            aug_s, calib, pre_cts=None, track_ids=None):
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h <= 0 or w <= 0:
            return
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        ret['cat'][k] = cls_id - 1
        ret['mask'][k] = 1
        if 'wh' in ret:
            ret['wh'][k] = 1. * w, 1. * h
            ret['wh_mask'][k] = 1
        ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
        ret['reg'][k] = ct - ct_int
        ret['reg_mask'][k] = 1
        draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

        gt_det['bboxes'].append(
            np.array([ct[0] - w / 2, ct[1] - h / 2,
                      ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
        gt_det['scores'].append(1)
        gt_det['classIds'].append(cls_id - 1)
        gt_det['centers'].append(ct)

        if 'tracking' in self.opt.heads:
            if ann['track_id'] in track_ids:
                pre_ct = pre_cts[track_ids.index(ann['track_id'])]
                ret['tracking_mask'][k] = 1
                ret['tracking'][k] = pre_ct - ct_int
                gt_det['tracking'].append(ret['tracking'][k])
            else:
                gt_det['tracking'].append(np.zeros(2, np.float32))

        if 'ltrb' in self.opt.heads:
            ret['ltrb'][k] = bbox[0] - ct_int[0], bbox[1] - ct_int[1], \
                bbox[2] - ct_int[0], bbox[3] - ct_int[1]
            ret['ltrb_mask'][k] = 1

        # ltrb_amodal is to use the left, top, right, bottom bounding box representation
        # to enable detecting out-of-image bounding box (important for MOT datasets)
        if 'ltrb_amodal' in self.opt.heads:
            ret['ltrb_amodal'][k] = \
                bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1], \
                bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]
            ret['ltrb_amodal_mask'][k] = 1
            gt_det['ltrb_amodal'].append(bbox_amodal)

        if 'nuscenes_att' in self.opt.heads:
            if ('attributes' in ann) and ann['attributes'] > 0:
                att = int(ann['attributes'] - 1)
                ret['nuscenes_att'][k][att] = 1
                ret['nuscenes_att_mask'][k][self.nuscenes_att_range[att]] = 1
            gt_det['nuscenes_att'].append(ret['nuscenes_att'][k])

        if 'velocity' in self.opt.heads:
            if ('velocity_cam' in ann) and min(ann['velocity_cam']) > -1000:
                ret['velocity'][k] = np.array(
                    ann['velocity_cam'], np.float32)[:3]
                ret['velocity_mask'][k] = 1
            gt_det['velocity'].append(ret['velocity'][k])

        if 'hps' in self.opt.heads:
            self._add_hps(ret, k, ann, gt_det,
                          trans_output, ct_int, bbox, h, w)

        if 'rot' in self.opt.heads:
            self._add_rot(ret, ann, k, gt_det)

        if 'dep' in self.opt.heads:
            if 'depth' in ann:
                ret['dep_mask'][k] = 1
                ret['dep'][k] = ann['depth'] * aug_s
                gt_det['dep'].append(ret['dep'][k])
            else:
                gt_det['dep'].append(2)

        if 'dim' in self.opt.heads:
            if 'dim' in ann:
                ret['dim_mask'][k] = 1
                ret['dim'][k] = ann['dim']
                gt_det['dim'].append(ret['dim'][k])
            else:
                gt_det['dim'].append([1, 1, 1])

        if 'amodel_offset' in self.opt.heads:
            if 'amodel_center' in ann:
                amodel_center = affineTransform(
                    ann['amodel_center'], trans_output)
                ret['amodel_offset_mask'][k] = 1
                ret['amodel_offset'][k] = amodel_center - ct_int
                gt_det['amodel_offset'].append(ret['amodel_offset'][k])
            else:
                gt_det['amodel_offset'].append([0, 0])

        if self.opt.pointcloud:
            # get pointcloud heatmap
            if self.opt.disable_frustum:
                ret['pc_hm'] = ret['pc_dep']
                if opt.normalize_depth:
                    ret['pc_hm'][self.opt.pc_feat_channels['pc_dep']
                                 ] /= opt.max_pc_dist
            else:
                dist_thresh = get_dist_thresh(
                    calib, ct, ann['dim'], ann['alpha'])
                pc_dep_to_hm(ret['pc_hm'], ret['pc_dep'],
                             ann['depth'], bbox, dist_thresh, self.opt)

    def _add_rot(self, ret, ann, k, gt_det):
        if 'alpha' in ann:
            ret['rot_mask'][k] = 1
            alpha = ann['alpha']
            if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                ret['rotbin'][k, 0] = 1
                ret['rotres'][k, 0] = alpha - (-0.5 * np.pi)
            if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                ret['rotbin'][k, 1] = 1
                ret['rotres'][k, 1] = alpha - (0.5 * np.pi)
            gt_det['rot'].append(self._alpha_to_8(ann['alpha']))
        else:
            gt_det['rot'].append(self._alpha_to_8(0))

    def _alpha_to_8(self, alpha):
        ret = [0, 0, 0, 1, 0, 0, 0, 1]
        if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
            r = alpha - (-0.5 * np.pi)
            ret[1] = 1
            ret[2], ret[3] = np.sin(r), np.cos(r)
        if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
            r = alpha - (0.5 * np.pi)
            ret[5] = 1
            ret[6], ret[7] = np.sin(r), np.cos(r)
        return ret

    def _format_gt_det(self, gt_det):
        if (len(gt_det['scores']) == 0):
            gt_det = {'bboxes': np.array([[0, 0, 1, 1]], dtype=np.float32),
                      'scores': np.array([1], dtype=np.float32),
                      'classIds': np.array([0], dtype=np.float32),
                      'centers': np.array([[0, 0]], dtype=np.float32),
                      'pre_cts': np.array([[0, 0]], dtype=np.float32),
                      'tracking': np.array([[0, 0]], dtype=np.float32),
                      'bboxes_amodal': np.array([[0, 0]], dtype=np.float32),
                      'hps': np.zeros((1, 17, 2), dtype=np.float32), }
        gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}
        return gt_det

    def fake_video_data(self):
        self.coco.dataset['videos'] = []
        for i in range(len(self.coco.dataset['images'])):
            img_id = self.coco.dataset['images'][i]['id']
            self.coco.dataset['images'][i]['video_id'] = img_id
            self.coco.dataset['images'][i]['frame_id'] = 1
            self.coco.dataset['videos'].append({'id': img_id})

        if not ('annotations' in self.coco.dataset):
            return

        for i in range(len(self.coco.dataset['annotations'])):
            self.coco.dataset['annotations'][i]['track_id'] = i + 1
