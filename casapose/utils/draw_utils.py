import cv2
import matplotlib as plt
import numpy as np


def draw_bb(
    xy,
    img,
    color_lines=(0, 0, 255),
    color_points=(0, 255, 255),
    width=2,
    line_type=cv2.LINE_4,
):
    xy = xy.astype(int)
    xy = tuple(map(tuple, xy))
    cv2.line(img, xy[0], xy[1], color_lines, width, line_type)
    cv2.line(img, xy[1], xy[3], color_lines, width, line_type)
    cv2.line(img, xy[3], xy[2], color_lines, width, line_type)
    cv2.line(img, xy[2], xy[0], color_lines, width, line_type)
    cv2.line(img, xy[0], xy[4], color_lines, width, line_type)
    cv2.line(img, xy[4], xy[5], color_lines, width, line_type)
    cv2.line(img, xy[5], xy[7], color_lines, width, line_type)
    cv2.line(img, xy[7], xy[6], color_lines, width, line_type)
    cv2.line(img, xy[6], xy[4], color_lines, width, line_type)
    cv2.line(img, xy[2], xy[6], color_lines, width, line_type)
    cv2.line(img, xy[7], xy[3], color_lines, width, line_type)
    cv2.line(img, xy[1], xy[5], color_lines, width, line_type)

    for p in xy:
        cv2.circle(img, p, 1, color_points, -1)
    return img


def draw_points(xy, img, color_points=(255, 0, 0), size=1, thickness=-1, line_type=cv2.LINE_AA):

    xy = tuple(map(tuple, xy))
    for p in xy:
        cv2.circle(
            img,
            (int(p[0]), int(p[1])),
            size,
            color_points,
            thickness,
            lineType=line_type,
        )
    return img


def draw_lines(xy_1, xy_2, img, color_points=(255, 255, 255)):
    xy_1 = xy_1.astype(int)
    xy_1 = tuple(map(tuple, xy_1))
    xy_2 = xy_2.astype(int)
    xy_2 = tuple(map(tuple, xy_2))
    for idx, p1 in enumerate(xy_1):
        p2 = xy_2[idx]
        cv2.line(img, p1, p2, color_points)
    return img


def pseudocolor_dir(x_dir, y_dir, mask):
    dir_map = np.arctan2(x_dir, y_dir) * 180.0 / np.pi
    dir_map[dir_map < 0.0] += 360.0
    dir_map[dir_map >= 360.0] -= 360.0
    dir_map[mask == 0] = 0.0
    dir_map = dir_map / 360.0  # * 179.0
    # dir_map = dir_map.astype('uint8')
    ones = np.full(mask.shape, 1.0, dtype="float")

    len_map = np.stack([x_dir, y_dir], -1)
    len_map = np.linalg.norm(len_map, axis=-1)
    len_map = np.clip(len_map, 0, 1)
    len_map = np.stack((len_map, len_map, len_map), axis=2) * 255.0

    dir_map = np.stack((dir_map, ones, ones), axis=2)
    dir_map = plt.colors.hsv_to_rgb(dir_map) * 255.0

    dir_map = dir_map.astype("uint8")
    len_map = len_map.astype("uint8")

    dir_map[mask == 0] = 0.0
    return dir_map


def grayscale_dist(dist, mask, clip_max):
    dist = (dist / clip_max) * 255
    dist = np.stack((dist, dist, dist), axis=2)
    dist = 255 - dist.astype("uint8")
    dist[mask == 0] = 0
    return dist
