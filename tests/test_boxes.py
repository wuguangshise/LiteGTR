"""Coordinate restoration -- a silent failure mode for submissions (P1-9)."""
import pytest

np = pytest.importorskip("numpy")

from utils.boxes import iou_matrix, letterbox_params, size_bucket, unletterbox, xyxy_to_xywh


def test_letterbox_roundtrip_recovers_original_coordinates():
    ori_hw, size = (1080, 1920), 640
    r, px, py = letterbox_params(ori_hw, size)
    orig = np.array([[100.0, 200.0, 300.0, 500.0]])
    # forward: exactly what datasets.transforms.LetterBox does
    fwd = orig.copy()
    fwd[:, [0, 2]] = fwd[:, [0, 2]] * r + px
    fwd[:, [1, 3]] = fwd[:, [1, 3]] * r + py
    back = unletterbox(fwd, ori_hw, size)
    assert np.allclose(back, orig, atol=1e-3)


def test_unletterbox_clips_to_image():
    out = unletterbox(np.array([[-50.0, -50.0, 5000.0, 5000.0]]), (480, 640), 640)
    assert out[0, 0] >= 0 and out[0, 1] >= 0
    assert out[0, 2] <= 640 and out[0, 3] <= 480


def test_unletterbox_handles_empty():
    assert unletterbox(np.zeros((0, 4)), (480, 640), 640).shape == (0, 4)


def test_iou_matrix_matches_hand_computation():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 5.0, 15.0, 15.0], [0.0, 0.0, 10.0, 10.0]])
    iou = iou_matrix(a, b)
    assert np.isclose(iou[0, 0], 25.0 / 175.0, atol=1e-5)
    assert np.isclose(iou[0, 1], 1.0, atol=1e-5)


def test_iou_matrix_empty_inputs():
    assert iou_matrix(np.zeros((0, 4)), np.ones((3, 4))).shape == (0, 3)


def test_size_buckets_follow_coco_thresholds():
    assert size_bucket(31 * 31) == "small"
    assert size_bucket(33 * 33) == "medium"
    assert size_bucket(200 * 200) == "large"


def test_xyxy_to_xywh():
    out = xyxy_to_xywh(np.array([[10.0, 20.0, 40.0, 60.0]]))
    assert np.allclose(out, [[10.0, 20.0, 30.0, 40.0]])


def test_letterbox_params_match_the_transform_exactly():
    """Odd padding: LetterBox pads with // -- evaluation must invert the same offset."""
    from datasets.transforms import LetterBox

    for ori_hw in [(765, 1360), (1500, 2000), (481, 640), (640, 427)]:
        box = np.array([[37.0, 21.0, 90.0, 77.0]], dtype=np.float32)
        img = np.zeros((*ori_hw, 3), np.uint8)
        _, fwd = LetterBox(640)(img, box.copy())
        back = unletterbox(fwd, ori_hw, 640)
        assert np.allclose(back, box, atol=1e-3), (ori_hw, back)
