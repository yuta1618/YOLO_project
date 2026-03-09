import sys
import os
import pandas as pd
import cv2
import logging
from ultralytics import YOLO

# 自作モジュールのパスを通す
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyzer import SpatialAnalyzer
from visualizer import SceneVisualizer

# ロギング設定 (プロフェッショナルなデバッグ用)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_pipeline(input_path, output_dir):
    # 1. 初期化
    model = YOLO('yolo11x-seg.pt') # 最先端セグメンテーションモデル
    viz = SceneVisualizer()
    
    # 2. 推論 (Tracking有効)
    logger.info(f"Processing: {input_path}")
    results = model.track(input_path, persist=True, conf=0.25)[0]
    img = cv2.imread(input_path)
    
    if results.boxes is None or results.boxes.id is None:
        logger.warning("No significant objects detected.")
        return

    # 3. 物理情報の抽出
    object_metrics = []
    boxes = results.boxes.xyxy.cpu().numpy()
    ids = results.boxes.id.cpu().numpy().astype(int)
    clss = results.boxes.cls.cpu().numpy().astype(int)

    for box, obj_id, cls_id in zip(boxes, ids, clss):
        center, area, occ = SpatialAnalyzer.get_metrics(box, img.shape)
        object_metrics.append({
            'id': obj_id,
            'class': model.names[cls_id],
            'cls_id': cls_id,
            'box': box,
            'center': center,
            'area_px': area,
            'occ': occ
        })

    # 4. 可視化とデータ保存
    final_img = viz.apply_overlay(img, results, object_metrics)
    
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'analytic_result.jpg'), final_img)
    pd.DataFrame(object_metrics).drop('box', axis=1).to_csv(os.path.join(output_dir, 'spatial_data.csv'), index=False)
    
    logger.info(f"Success! Results saved in {output_dir}")

if __name__ == "__main__":
    # Google Colab上のパスを指定
    execute_pipeline('/content/test_street.jpg', 'YOLO_project/outputs')