import numpy as np

class SObjScoring:
    def __init__(self, threshold=0.5):
        """
        threshold: 遮罩中前景像素占比超過該閾值，判定為物體存在
        """
        self.threshold = threshold

    def is_object_present(self, mask):
        """
        根據遮罩中前景像素的比例判斷物體是否存在
        """
        foreground_ratio = np.count_nonzero(mask) / mask.size
        return foreground_ratio > self.threshold

    def score(self, mask):
        """
        返回物體存在性分數
        """
        return 1.0 if self.is_object_present(mask) else 0.0
