import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.utils.logging import get_logger

logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        logger.info("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.info("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


# ??????????????????????????????FVC??????
def cut_img_FVC(xps_img):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.175)
    wide2 = int(img_shape[1] * 0.66371)
    long1 = int(img_shape[0] * 0.4281642)
    long2 = int(img_shape[0] * 0.5741163)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_FVC.jpg', xps_img)
    return xps_img


# (62,62) (1185,345)
def cut_img_info(xps_img):
    img_shape = xps_img.shape
    wide1 = int(img_shape[1] * 0.048)
    wide2 = int(img_shape[1] * 0.9556452)
    long1 = int(img_shape[0] * 0.035348)
    long2 = int(img_shape[0] * 0.1967)
    # return xps_img[751: 1007, 184: 823]
    xps_img = xps_img[long1: long2, wide1: wide2]
    cv2.imwrite('D:\\pythonProject\\PaddleOCR-release-2.0\\cut_info.jpg', xps_img)
    return xps_img

# 12??????10???
def fill_FVC(xps_img):
    row_list = ['FVC', 'FEV1', 'FEV1/FVC%', 'PEF', 'FEF25-75%', 'MEF25%', 'MEF50%', 'MEF75%', 'FEV6', 'FEV1/FEV6%',
                'MIF/MEF50%', 'FEV1/VCMAX%']
    column_list = ['PRE_?????????', 'PRE_????????????', 'PRE_Pred', 'PRE_%?????????', 'PRE_z-score', '????????????_?????????', '????????????_??????',
                   '????????????_%??????', '????????????_%?????????', '????????????_z-score']
    fill_excel = dict()
    add1 = xps_img.shape[0] // 12
    # ??????12?????????
    img_row_list = list()
    row_pixel = 0
    for i in range(12):
        img_row_list.append(xps_img[row_pixel: row_pixel + add1, :])
        row_pixel = row_pixel + add1
    return img_row_list


# 1126*283
def fill_info(xps_img):
    img_row_list = list()
    row_rate = 46 / 283 / 2
    row_pixel = 0
    row_add = int(xps_img.shape[0] * row_rate)
    for i in range(4):
        if i == 1:
            row_pixel = row_pixel + row_add * 2
        else:
            img_row_list.append(xps_img[row_pixel + row_add: row_pixel + row_add * 2, :])
            row_pixel = row_pixel + row_add * 2
    return img_row_list




# ?????????????????????????????????????????????????????????10???????????????
def FVC_row_excel(row_img):
    column_list = ['PRE_?????????', 'PRE_????????????', 'PRE_Pred', 'PRE_%?????????', 'PRE_z-score', '????????????_?????????', '????????????_??????',
                   '????????????_%??????', '????????????_%?????????', '????????????_z-score']
    row_ps = [0, 0.0742574, 0.227723, 0.316, 0.409241, 0.514851, 0.6113861, 0.708746, 0.8028, 0.90429, 1]
    img_shape = row_img.shape
    row_excel = list()
    for i in range(10):
        row_excel.append(row_img[:, int(img_shape[1] * row_ps[i]): int(img_shape[1] * row_ps[i+1])])
    return row_excel


# 316*107
def PRE_row_excel(row_img):
    row_ps = [0, 0.17405, 0.46202531, 0.623417724, 0.835443038, 1]
    img_shape = row_img.shape
    row_excel = list()
    for i in range(5):
        row_excel.append(row_img[:, int(img_shape[1] * row_ps[i]): int(img_shape[1] * row_ps[i + 1])])
    return row_excel


def info_row_excel(row_img):
    # r0 = [0.9041, 1]
    # r1 = [0.231, 0.3410, 0.566607, 0.6554174, 0.6971581, 0.739787, 0.79, 0.825, 0.83, 0.91, 0.952, 0.99]
    # r2 = [0.41, 0.4982]
    # row_img = row_img[0]
    row_excel = list()
    img_shape = row_img[0].shape
    r = [[0.9041, 1], [0.231, 0.3410, 0.566607, 0.6554174, 0.6971581, 0.739787, 0.79, 0.825, 0.83, 0.91, 0.952, 1],
         [0.41, 0.4982]]
    for i in range(3):
        rr_len = len(r[i])
        for ii in range(0, rr_len, 2):
            le = int(img_shape[1] * r[i][ii])
            ri = int(img_shape[1] * r[i][ii + 1])
            row_excel.append(row_img[i][:, le: ri])
    return row_excel

def cut_img_SVC(xps_img):
    pass


# 1_1.jpeg 1240*1754
# ?????????ID??????????????????????????????????????????????????????????????????
def main(args):
    row_list = ['??????', 'ID', '??????', '??????', '??????', '??????', '????????????', '????????????']
    te = predict_rec.TextRecognizer(args)
    image_file_list = get_image_file_list(args.image_dir)
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img_ori = cv2.imread(image_file)
            img_PRE = cut_img_info(img_ori)
            img_PRE = fill_info(img_PRE)
            inf_1 = info_row_excel(img_PRE)
        if inf_1 is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        for p in inf_1:
            cv2.imshow('1', p)
            cv2.waitKey(0)
        tb, tr = te(inf_1)
        print(tb)
        print(tr)

if __name__ == "__main__":
    main(utility.parse_args())
