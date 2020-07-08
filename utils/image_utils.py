import cv2
import numpy as np

# conf:float
# bbox: [xmin, xmax, ymin, ymax]
# landm: [x0,y0, x1,y1,...,xn,yn]
def draw_bbox(image, conf, bbox, landm, draw_landm_index=False):
    bbox_color = (0, 0, 255)
    text_color = (255, 255, 255)
    landmark_color = (255, 0, 0)
    b = list(map(int, bbox))
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), bbox_color, 2)

    conf_text = "{:.4f}".format(conf)
    cx = b[0]
    cy = b[1] + 12
    cv2.putText(image, conf_text, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    # landms
    landm = np.array(list(map(int, landm)))
    landm = landm.reshape(-1, 2)
    for i, pt in enumerate(landm):
        x = int(pt[0])
        y = int(pt[1])
        cv2.circle(image, (x, y), 1, landmark_color, 2, cv2.LINE_AA)
        if draw_landm_index:
            text = str(i + 1)
            cx = x + 4
            cy = y + 2
            cv2.putText(image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)


def draw_bboxes(image, confs, bboxes, landms, draw_landm_index=False, conf_thresh=0.5):
    for index, bbox in enumerate(bboxes):
        conf = confs[index]
        if conf < conf_thresh:
            continue
        landm = landms[index]
        draw_bbox(image, conf, bbox, landm, draw_landm_index)


def show_bboxes(image, confs, bboxes, landms,
                draw_landm_index=False, conf_thresh=0.5, show_image_size=(500,500)):
    win_name = "test"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
    dst_w = show_image_size[0]
    dst_h = show_image_size[1]
    cv2.resizeWindow(win_name, dst_w, dst_h)

    src_w = image.shape[1]
    src_h = image.shape[0]
    image = cv2.resize(image, (dst_w, dst_h))
    bboxes[:, 0::2] *= (dst_w / src_w)
    bboxes[:, 1::2] *= (dst_h / src_h)
    landms[:, 0::2] *= (dst_w / src_w)
    landms[:, 1::2] *= (dst_h / src_h)
    draw_bboxes(image, confs, bboxes, landms, draw_landm_index, conf_thresh)
    cv2.imshow(win_name, image)
    cv2.waitKey()

