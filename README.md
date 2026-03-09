# 🚀 YOLOv11: Advanced Spatial & Physical Analysis Pipeline
## 📌 概要
本プロジェクトは、最新の物体検出モデル **YOLOv11** をフル活用し、静止画および動画から**「高度な物理情報」**を抽出するための統合解析パイプラインです。単なる物体検知にとどまらず、空間内での物体の占有面積、個体識別、および物体間の相互作用（距離関係）を数値化します。



## ✨ 主な機能

### 1. 統合インスタンス解析 (Integrated Instance Analysis)
- **Instance Segmentation**: 物体の境界線をピクセル単位で特定し、形状を抽出。
- **Object Tracking**: 独自のIDを付与し、画像内の全個体を識別管理。
- **Pose Estimation**: 人物の骨格（17ポイント）を検知し、姿勢や向きを解析。

### 2. 物理情報の数値化 (Physical Metrics)
- **占有面積 (Area Occupation)**: 画像全体の解像度に対する各物体のピクセル占有率を算出。
- **座標位置 (Spatial Coordinates)**: 物体の中心点 ($c_x, c_y$) を特定。
- **近接検知 (Proximity Alert)**: 物体間のユークリッド距離を計算し、設定値以下の接近を自動警告。

### 3. 自動レポーティング (Automated Reporting)
- **Visual Report**: セグメンテーション、ID、ポーズ、距離情報を統合した解析画像を生成。
- **Data Export**: 全物体の物理統計（クラス、ID、面積、座標）をCSV形式で出力。



## 🛠 セットアップと使用方法

### Google Colab で実行する場合
1. リポジトリ内の `main.ipynb` を Google Colab で開きます。
2. `Runtime` > `Change runtime type` から **T4 GPU** を選択します。
3. 全てのセルを実行します。

### ローカル環境の場合
```bash
git clone [https://github.com/yuta1618/yolo-advanced-analysis.git](https://github.com/yuta1618/yolo-advanced-analysis.git)
cd yolo-advanced-analysis
pip install -r requirements.txt
python src/main.py --source data/test.jpg
