import numpy as np
from PIL import Image
import concurrent.futures
import pathlib


class Confusion:
    """
    Calculate and visualize metrics of predicted images.
    """

    def __init__(self, gt_pth, pd_pth):
        """
        Set up the class.
        """

        # Paths
        self.gt_pth = gt_pth
        self.pd_pth = pd_pth

        self._open_images()

        # Calculate metrics
        self._confusion_elements()
        self._iou_score()
        self._f1_score()
        self._accuracy()
        self._recall()
        self._precision()

    def _open_images(self):
        # Open ground troth and predicted image
        gt_img = np.asarray(Image.open(self.gt_pth))
        pd_img = np.asarray(Image.open(self.pd_pth))

        # Store as array
        self.gt_arr = np.asarray(gt_img)
        self.pd_arr = np.asarray(pd_img)

    def _confusion_elements(self):
        """
        Calculate boolean confusion matrix elements as numpy arrays
        containing
        true positive (tp), false positive (fp),
        false negative(fn), true negative(tn).
        """
        ground_truth = self.gt_arr
        predicted = self.pd_arr
        # Invert ground truth and prediction.
        ground_truth_inverse = np.logical_not(ground_truth)
        predicted_inverse = np.logical_not(predicted)

        # Calculate confusion elements.
        self.true_positive_matrix = np.logical_and(ground_truth, predicted)
        self.true_negative_matrix = np.logical_and(ground_truth_inverse, predicted_inverse)
        self.false_positive_matrix = np.logical_and(ground_truth_inverse, predicted)
        self.false_negative_matrix = np.logical_and(ground_truth, predicted_inverse)

        # Reduce confusion elements
        self.true_positive = np.sum(self.true_positive_matrix)
        self.true_negative = np.sum(self.true_negative_matrix)
        self.false_positive = np.sum(self.false_positive_matrix)
        self.false_negative = np.sum(self.false_negative_matrix)

    def _iou_score(self):
        """
        Get intersection over union (IoU) of confusion.
        """
        fn = self.false_negative
        fp = self.false_positive
        tp = self.true_positive
        # Special case
        if (fp + tp + fn) == 0:
            self.iou = 0
        else:
            self.iou = tp / (fp + tp + fn)

    def _f1_score(self):
        """
        Get f1 score of image confusion.
        """
        fn = self.false_negative
        fp = self.false_positive
        tp = self.true_positive
        # Special case
        if ((2 * tp) + fp + fn) == 0:
            self.f1 = 0
        else:
            self.f1 = (2 * tp) / ((2 * tp) + fp + fn)

    def _accuracy(self):
        """
        Get accuracy of image confusion.
        """
        tp = self.true_positive
        tn = self.true_negative
        fp = self.false_positive
        fn = self.false_negative
        # Special case
        if tp + tn == 0:
            self.accuracy = 0
        else:
            self.accuracy = (tp + tn) / (tp + tn + fp + fn)

    def _recall(self):
        """
        Get recall (false negative rate) of image confusion.
        """
        fn = self.false_negative
        tp = self.true_positive
        # Special case
        if tp + fn == 0:
            self.recall = 0
        else:
            self.recall = tp / (tp + fn)

    def _precision(self):
        """
        Get accuracy (ture positive rate) of image confusion.
        """
        tp = self.true_positive
        fp = self.false_positive
        # Special case
        if tp + fp == 0:
            self.precision = 0
        else:
            self.precision = tp / (tp + fp)

    def confusion_image(self):
        """
        Visualized confusion.
        """
        # Get confusion elements
        confusion = {
            "tp": self.true_positive_matrix,
            "tn": self.true_negative_matrix,
            "fp": self.false_positive_matrix,
            "fn": self.false_negative_matrix,
        }

        # Confusion element colors
        color = {
            "tp": np.array([0, 255, 0], dtype=np.uint8),
            "tn": np.array([0, 0, 255], dtype=np.uint8),
            "fp": np.array([255, 0, 0], dtype=np.uint8),
            "fn": np.array([255, 255, 0], dtype=np.uint8),
        }

        # Replace every confusion element with color
        for key in confusion:
            confusion[key] = confusion[key].astype(np.uint8)
            confusion[key] = np.stack(
                (confusion[key], confusion[key], confusion[key]), axis=2
            )
            confusion[key] = np.where(
                confusion[key] == [1, 1, 1],
                color[key],
                confusion[key],
            )

        # Merge confusion color mask in one array
        confusion_mask = np.sum(
            (confusion["tp"], confusion["tn"], confusion["fp"], confusion["fn"]),
            axis=0,
            dtype=np.uint8,
        )
        confusion_img = Image.fromarray(confusion_mask)

        return confusion_img


def _evaluate_metric(gt_pth, pd_pth):
    """
    Evaluate metric for one picture

    :param gt_pth: Path to ground truth image.
    :param pd_pth: Path to predicted image.
    :return: Dictionary with image pair metrics.
    """
    confusion = Confusion(gt_pth, pd_pth)
    metrics = {
        "iou": confusion.iou,
        "f1": confusion.f1,
        "acc": confusion.accuracy,
        "prc": confusion.precision,
        "rec": confusion.recall,
        "tp": confusion.true_positive,
        "fp": confusion.false_positive,
        "fn": confusion.false_negative,
        "tn": confusion.true_negative,
    }
    return metrics


class BatchMetrics:
    """
    Compute confusion metrics over a batch of images.
    """

    def __init__(self, gt_pth, pd_pth, gt_ext="png", pd_ext="png"):
        """
        Set up class.

        :param gt_pth: Path to directory containing binary ground truth.
        :param pd_pth: Path to directory with binary predicted images.
        :param gt_ext: File extension for ground truth images.
        :param pd_ext: File extension for predicted images.
        :return: None.
        """
        # Paths
        self.gt_pth = gt_pth
        self.pd_pth = pd_pth
        self.gt_ext = gt_ext
        self.pd_ext = pd_ext

        # Metrics
        self.iou_list = []
        self.f1_list = []
        self.acc_list = []
        self.prc_list = []
        self.rec_list = []
        self.tp_list = []
        self.fp_list = []
        self.fn_list = []
        self.tn_list = []

        self._collect_paths()
        self._batch_metrics()

    def _collect_paths(self):
        """
        Collect path to ground truth and prediction images.

        :return: None.
        """
        # Collect list of paths with file extension
        self.gt_pths = list(pathlib.Path(self.gt_pth).glob(f"*.{self.gt_ext}"))
        self.pd_pths = list(pathlib.Path(self.pd_pth).glob(f"*.{self.pd_ext}"))

        # Sort paths to iterate pairwise over prediction and ground truth
        self.gt_pths = sorted(self.gt_pths)
        self.pd_pths = sorted(self.pd_pths)

        # Sanity checks.
        # TODO: Sanity checks with same file names, not only same amount.
        if len(self.gt_pths) != len(self.pd_pths):
            msg = "Expected same count of ground truth images as predicted, "
            msg += f"got {len(self.gt_pths)} ground truth and "
            msg += f"{len(self.pd_pths)} predicted images."
            raise ValueError(msg)

        if len(self.gt_pths) < 1 or len(self.pd_pths) < 1:
            msg = "Expected at least one ground truth image and one "
            msg += f"predicted image, got {len(self.gt_pths)} ground "
            msg += f"truth images and {len(self.pd_pths)} predicted "
            msg += "images."
            raise ValueError(msg)

    def _batch_metrics(self):
        """
        Compute metrics for all images in folder.

        :return: None.
        """

        arguments = (self.gt_pths, self.pd_pths)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for metric in executor.map(_evaluate_metric, *arguments):
                self.iou_list.append(metric["iou"])
                self.f1_list.append(metric["f1"])
                self.acc_list.append(metric["acc"])
                self.prc_list.append(metric["prc"])
                self.rec_list.append(metric["rec"])
                self.tp_list.append(metric["tp"])
                self.fp_list.append(metric["fp"])
                self.fn_list.append(metric["fn"])
                self.tn_list.append(metric["tn"])

        self.worst_iou = min(self.iou_list)
        self.best_iou = max(self.iou_list)
        self.iou = sum(self.iou_list) / len(self.iou_list)

        self.worst_f1 = min(self.f1_list)
        self.best_f1 = max(self.f1_list)
        self.f1 = sum(self.f1_list) / len(self.f1_list)

        self.worst_acc = min(self.acc_list)
        self.best_acc = max(self.acc_list)
        self.acc = sum(self.acc_list) / len(self.acc_list)

        self.worst_prc = min(self.prc_list)
        self.best_prc = max(self.prc_list)
        self.prc = sum(self.prc_list) / len(self.prc_list)

        self.worst_rec = min(self.rec_list)
        self.best_rec = max(self.rec_list)
        self.rec = sum(self.rec_list) / len(self.rec_list)

        self.worst_tp = min(self.tp_list)
        self.best_tp = max(self.tp_list)
        self.tp = sum(self.tp_list) / len(self.tp_list)

        self.worst_fp = min(self.fp_list)
        self.best_fp = max(self.fp_list)
        self.fp = sum(self.fp_list) / len(self.fp_list)

        self.worst_fn = min(self.fn_list)
        self.best_fn = max(self.fn_list)
        self.fn = sum(self.fn_list) / len(self.fn_list)

        self.worst_tn = min(self.tn_list)
        self.best_tn = max(self.tn_list)
        self.tn = sum(self.tn_list) / len(self.tn_list)


def main():
    pass


if __name__ == "__main__":
    main()
