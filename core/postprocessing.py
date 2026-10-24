"""Post-processing utilities: NMS, box scaling, and result formatting."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """A single detected object."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class InferenceResult:
    """Result of running inference on a single image."""

    detections: list[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    inference_ms: float = 0.0
    num_detections: int = 0

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_ms": round(self.inference_ms, 2),
            "num_detections": self.num_detections,
        }


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: list[int] | None = None,
    agnostic: bool = False,
    max_det: int = 300,
) -> list[np.ndarray]:
    """Perform Non-Maximum Suppression on detection predictions.

    Expects prediction shape [1, num_anchors, 5 + num_classes] or
    [num_anchors, 5 + num_classes]. Returns a list of arrays, one per image,
    each with columns [x1, y1, x2, y2, conf, class].
    """
    if prediction.ndim == 3:
        prediction = prediction[0]  # remove batch dim -> (num_anchors, 5 + nc)

    nc = prediction.shape[1] - 5  # number of classes
    xc = prediction[:, 4] > conf_thres  # objectness confidence filter

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"

    # Settings
    max_wh = 4096  # (pixels) maximum box width/height
    max_nms = 30000  # maximum number of boxes into NMS
    multi_label = nc > 1  # multiple labels per box

    x = prediction[xc]  # filtered anchors -> (n, 5 + nc)
    if not x.shape[0]:
        return [np.zeros((0, 6))]

    # Compute conf = obj_conf * cls_conf
    x[:, 5:] *= x[:, 4:5]

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
        i, j = np.where(x[:, 5:] > conf_thres)
        x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].astype(float)), 1)
    else:  # best class only
        conf = x[:, 5:].max(1, keepdims=True)
        j = x[:, 5:].argmax(1).reshape(-1, 1)
        x = np.concatenate((box, conf, j.astype(float)), 1)[conf.reshape(-1) > conf_thres]

    # Filter by class
    if classes is not None:
        x = x[(x[:, 5:6] == np.array(classes)).any(1)]

    # Apply constraint
    n = x.shape[0]  # number of boxes
    if not n:
        return [np.zeros((0, 6))]
    x = x[np.argsort(-x[:, 4])[:max_nms]]  # sort by conf descending

    # Batched NMS
    c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    i = nms(boxes, scores, iou_thres)  # NMS
    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]

    return [x[i]]


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Pure NumPy Non-Maximum Suppression.

    Returns indices of kept boxes.
    """
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def scale_boxes(
    img1_shape: tuple,
    boxes: np.ndarray,
    img0_shape: tuple,
    ratio_pad: tuple | None = None,
) -> np.ndarray:
    """Rescale bounding boxes from img1_shape (model input) to img0_shape (original)."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes: np.ndarray, shape: tuple) -> None:
    """Clip bounding boxes to image shape (in place)."""
    boxes[:, 0] = np.clip(boxes[:, 0], 0, shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, shape[0])  # y2


def format_result(
    raw_pred: np.ndarray,
    orig_shape: tuple,
    names: dict,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    inference_ms: float,
) -> InferenceResult:
    """Run NMS and format raw model output into an InferenceResult."""
    pred = non_max_suppression(
        raw_pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
    )

    result = InferenceResult(
        image_width=orig_shape[1],
        image_height=orig_shape[0],
        inference_ms=inference_ms,
    )

    if len(pred) == 0 or pred[0].shape[0] == 0:
        result.num_detections = 0
        return result

    det = pred[0]
    # det columns: x1, y1, x2, y2, conf, class
    det[:, :4] = scale_boxes(
        (1, *orig_shape[::-1]), det[:, :4], orig_shape
    )  # scale boxes to original image

    detections = []
    for *xyxy, conf, cls in det:
        cls_id = int(cls)
        detections.append(
            Detection(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                confidence=float(conf),
                class_id=cls_id,
                class_name=str(names.get(cls_id, str(cls_id))),
            )
        )

    result.detections = detections
    result.num_detections = len(detections)
    return result
