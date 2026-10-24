"""Unit tests for post-processing NMS and scaling."""

import numpy as np

from core.postprocessing import (
    InferenceResult,
    format_result,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
)


def test_xywh2xyxy_conversion():
    boxes = np.array([[100, 100, 50, 50]], dtype=np.float32)
    out = xywh2xyxy(boxes)
    assert out.shape == (1, 4)
    assert out[0, 0] == 75.0  # x1
    assert out[0, 1] == 75.0  # y1
    assert out[0, 2] == 125.0  # x2
    assert out[0, 3] == 125.0  # y2


def test_nms_filters_overlapping():
    # Two identical high-confidence boxes - NMS should keep one
    num_anchors = 6
    num_classes = 1
    pred = np.zeros((1, num_anchors, 5 + num_classes), dtype=np.float32)
    # Anchor 0 and 1: same box, high objectness + class confidence
    for idx in (0, 1):
        pred[0, idx, 0] = 100  # cx
        pred[0, idx, 1] = 100  # cy
        pred[0, idx, 2] = 50  # w
        pred[0, idx, 3] = 50  # h
        pred[0, idx, 4] = 0.9  # obj conf
        pred[0, idx, 5] = 1.0  # class 0 conf

    out = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    assert len(out) == 1
    assert out[0].shape[0] == 1  # one box kept


def test_scale_boxes():
    boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    scaled = scale_boxes((640, 640), boxes, (480, 640))
    assert scaled.shape == (1, 4)


def test_format_result_empty():
    pred = np.zeros((1, 100, 85), dtype=np.float32)
    result = format_result(
        raw_pred=pred,
        orig_shape=(480, 640, 3),
        names={0: "licence"},
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        inference_ms=10.0,
    )
    assert isinstance(result, InferenceResult)
    assert result.num_detections == 0
    assert result.detections == []
