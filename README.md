# 人工智慧概論 期末報告 ( SAM2 改良 )

## 參考

- [SAM2 GitHub](https://github.com/facebookresearch/sam2)
- [SAMURAI GitHub](https://github.com/yangchris11/samurai)

## 專案架構

- data/：影片位置
- lib/：SAMURAI 核心函式庫與工具
- sam2/：基於 SAM2 修改的模型
- scripts/：推論、展示的腳本。
- checkpoints/：預訓練模型的權重與檢查點

## 使用方式

需要的版本：`python>=3.10`、`torch>=2.3.1`、`torchvision>=0.18.1`

### 安裝 SAM2 依賴

```
cd sam2
pip install -e .
pip install -e ".[notebooks]"
```

### 安裝其他依賴

```
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru
```

### 下載 checkpoint

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

## Demo

在終端機輸入

```
python scripts/demo.py --video_path data/game1/game1.mp4 --txt_path data/game1/game1.txt
```

data/ 裡有三個影片可做 demo，只須更改 video_path 與 txt_path
