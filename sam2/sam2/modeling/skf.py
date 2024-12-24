import numpy as np
from filterpy.kalman import KalmanFilter

class SKFScoring:
    def __init__(self):
        """
        初始化卡爾曼濾波器和運動模型參數
        """
        self.kf = KalmanFilter(dim_x=8, dim_z=4)  # 8個狀態維度, 4個觀測維度
        self._initialize_kalman_filter()
    
    def _initialize_kalman_filter(self):
        """
        初始化卡爾曼濾波器的矩陣參數
        """
        # 狀態轉移矩陣 (F)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        
        # 觀測矩陣 (H)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        # 測量噪聲矩陣 (R)
        self.kf.R *= 0.01
        
        # 預測誤差協方差矩陣 (P)
        self.kf.P *= 1000
        
        # 過程噪聲協方差矩陣 (Q)
        self.kf.Q *= 0.01

    def predict(self):
        """
        使用卡爾曼濾波器進行狀態預測
        """
        self.kf.predict()
        predicted_state = self.kf.x
        predicted_bbox = [
            predicted_state[0],  # x
            predicted_state[1],  # y
            predicted_state[2],  # w
            predicted_state[3]   # h
        ]
        return predicted_bbox
    
    def update(self, measurement):
        """
        使用新的測量值更新卡爾曼濾波器狀態
        :param measurement: [x, y, w, h]
        """
        self.kf.update(np.array(measurement))
    
    def calculate_iou(self, bbox1, bbox2):
        """
        計算兩個邊界框的 IoU
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union = area1 + area2 - intersection

        return intersection / union if union != 0 else 0

    def score(self, mask_bbox):
        """
        使用卡爾曼預測的邊界框計算 IoU 分數
        :param mask_bbox: 來自遮罩的邊界框 [x, y, w, h]
        """
        predicted_bbox = self.predict()
        iou_score = self.calculate_iou(predicted_bbox, mask_bbox)
        return iou_score
