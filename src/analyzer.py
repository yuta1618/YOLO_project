import numpy as np

class SpatialAnalyzer:
    """物体の位置関係や物理的な占有率を計算するクラス"""
    
    @staticmethod
    def get_metrics(box, img_shape):
        """バウンディングボックスから中心座標と占有率を算出"""
        h, w = img_shape[:2]
        x1, y1, x2, y2 = box
        
        # 中心点 (Center point)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        # 面積と占有率 (Area and Occupancy)
        area_px = int((x2 - x1) * (y2 - y1))
        occ_ratio = (area_px / (h * w)) * 100
        
        return (cx, cy), area_px, occ_ratio

    @staticmethod
    def compute_distance(p1, p2):
        """2つの物体間のユークリッド距離を計算"""
        return np.linalg.norm(np.array(p1) - np.array(p2))