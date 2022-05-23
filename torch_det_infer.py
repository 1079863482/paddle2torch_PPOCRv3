import os
from det.DetModel import DetModel
import torch
from addict import Dict as AttrDict
import cv2
import numpy as np
import math
import time
import pyclipper
from shapely.geometry import Polygon


class DBPostProcess():
    def __init__(self, thresh=0.3, box_thresh=0.4, max_candidates=1000, unclip_ratio=2):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, pred, h_w_list, is_output_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        h_w_list: 包含[h,w]的数组
        pred:
            binary: text region segmentation map, with shape (N, 1,H, W)
        '''
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = h_w_list[batch_index]
            boxes, scores = self.post_p(pred[batch_index], segmentation[batch_index], width, height,is_output_polygon=is_output_polygon)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh

    def post_p(self, pred, bitmap, dest_width, dest_height, is_output_polygon=False):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''
        height, width = pred.shape
        boxes = []
        new_scores = []
        bitmap = bitmap.cpu().numpy()
        if cv2.__version__.startswith('3'):
            _, contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            four_point_box, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            if not is_output_polygon:
                box = np.array(four_point_box)
            else:
                box = box.reshape(-1, 2)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            new_scores.append(score)
        return boxes, new_scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        bitmap = bitmap.detach().cpu().numpy()
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

def narrow(image, expected_size=(224,224)):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    # scale = eh / ih
    scale = min((eh/ih),(ew/iw))
    # scale = eh / max(iw,ih)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = 0
    bottom = eh - nh
    left = 0
    right = ew - nw
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return new_img

def draw_bbox(img_path, result, color=(0, 0, 255), thickness=2):
    import cv2
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path

def img_nchw(img):
    mean = 0.5
    std = 0.5
    resize_ratio = min((640 / img.shape[0]),(640/img.shape[1]))
    img = cv2.resize(img,(0,0),fx=resize_ratio,fy= resize_ratio,interpolation=cv2.INTER_LINEAR)
    h,w= img.shape[:2]
    if h == 640:
        w = (math.ceil(w/32)+1)*32
    elif w == 640:
        h = (math.ceil(h/32)+1)*32
    img1 = narrow(img,(w,h))

    img_data = (img1.astype(np.float32)/255 - mean) / std
    img_np = img_data.transpose(2,0,1)
    img_np = np.expand_dims(img_np,0)
    return img1,img_np


if __name__ == '__main__':
    det_model_path = './weights/ppv3_db.pth'
    test_img = "./det_images"

    post_proess = DBPostProcess()

    db_config = AttrDict(
        in_channels=3,
        backbone=AttrDict(type='MobileNetV3', model_name='large',scale=0.5,pretrained=True),
        neck=AttrDict(type='RSEFPN', out_channels=96),
        head=AttrDict(type='DBHead')
    )

    det_model = DetModel(db_config)
    det_model.load_state_dict(torch.load(det_model_path))
    det_model.eval()

    path_list = os.listdir(test_img)
    for name in path_list:
        img = cv2.imread(os.path.join(test_img, name))
        img0, img_np_nchw = img_nchw(img)

        input_for_torch = torch.from_numpy(img_np_nchw)
        out = det_model(input_for_torch)  # torch model infer

        box_list, score_list = post_proess(out, [img0.shape[:2]], is_output_polygon=False)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []

        img1 = draw_bbox(img0, box_list)
        cv2.imshow("draw", img1)
        cv2.waitKey()


