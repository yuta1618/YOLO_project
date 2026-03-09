import cv2
from ultralytics.utils.plotting import Annotator, colors

class SceneVisualizer:
    """解析結果の可視化を管理するクラス"""
    
    def __init__(self, line_width=2):
        self.line_width = line_width

    def apply_overlay(self, img, results, metadata):
        """セグメンテーションマスクとIDラベルを合成"""
        annotator = Annotator(img.copy(), line_width=self.line_width)
        h, w = img.shape[:2]

        # 1. マスクの描画
        if results.masks:
            mask_layer = img.copy()
            for i, mask in enumerate(results.masks.data):
                cls = int(results.boxes.cls[i])
                m = cv2.resize(mask.cpu().numpy(), (w, h))
                mask_layer[m > 0.5] = colors(cls, True)
            img = cv2.addWeighted(img, 0.7, mask_layer, 0.3, 0)
            annotator = Annotator(img, line_width=self.line_width)

        # 2. ボックスとID、物理情報の描画
        for data in metadata:
            label = f"ID:{data['id']} {data['class']} ({data['occ']:.1f}%)"
            annotator.box_label(data['box'], label, color=colors(data['cls_id'], True))
            
        return annotator.result()