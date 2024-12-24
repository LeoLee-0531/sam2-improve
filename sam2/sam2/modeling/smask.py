import numpy as np

class SMaskScoring:
    def __init__(self):
        pass

    def calculate_mask_confidence(self, mask):
        """
        根據遮罩的覆蓋範圍和一致性計算置信度
        """
        non_zero_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        
        confidence = non_zero_pixels / total_pixels
        return confidence

    def score(self, mask):
        """
        返回遮罩的置信度分數
        """
        confidence = self.calculate_mask_confidence(mask)
        return confidence
