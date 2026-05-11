這是一個為「嵌入式人工智慧與自駕車系統設計」課程開發的自動駕駛機器人專案。本專案整合了計算機視覺、深度學習與硬體控制，實現了一個能夠在模擬跑道上自主行駛、辨識號誌並接受遠端監控的系統。

## 👥 開發團隊
- **組員**: 郭峻宇、張采盈

## 🌟 核心功能

### 1. 自動車道跟隨 (Autonomous Lane Following)
- **CV 演算法**: 使用 OpenCV 進行影像預處理（ROI 裁切、閾值過濾、形態學操作）及 Hough 轉換提取車道線。
- **智慧判斷**: 支援雙線模式（中心行駛）與單線模式（根據左/右線推算路徑中心）。
- **平滑控制**: 內建 PID 控制器，根據影像偏差即時修正轉向，確保行駛曲線平滑。

### 2. AI 號誌辨識 (Traffic Light Recognition)
- **模型**: 使用 YOLOv8 進行輕量化物件偵測。
- **訓練**: 透過 Roboflow 標註資料集，並於 Google Colab GPU 環境訓練 50 個 Epochs。
- **決策**: 具備紅綠燈偵測邏輯。若連續偵測到紅燈則強制停止；偵測到綠燈或無號誌時繼續行駛。

### 3. 即時網頁儀表板 (Cyberpunk Dashboard)
- **視覺化串流**: 合併顯示 YOLO 偵測畫面與車道偵測圖層。
- **遠端遙控**: 支援 WebSocket 通訊，可用鍵盤 (WASD/方向鍵) 或網頁虛擬按鈕手動駕駛。
- **動態調參**: 
  - 即時調整 PID (Kp, Ki, Kd) 參數。
  - 調整基礎行駛速度 (Base Speed)。
  - 即時查看車道偏差量 (Pixel Error) 的視覺化統計圖。

## 🛠️ 硬體架構

- **主運算單元**: Raspberry Pi 4B (64-bit)
- **感測器**: Raspberry Pi Camera Module 3
- **馬達驅動**: PCA9685 PWM 模組 (I2C)
- **底盤**: 4 輪直流減速馬達機器人平台
- **控制引腳**: PCA9685 Channels + Raspberry Pi GPIO (gpiozero)

## 📦 軟體需求

### 環境依賴
- Python 3.9+
- picamera2 (樹莓派相機新版驅動)
- Flask, Flask-Sock
- OpenCV
- Ultralytics (YOLOv8)
- smbus2, gpiozero

### 安裝步驟
```bash
# 建議在虛擬環境執行
pip install flask flask-sock opencv-python ultralytics smbus2 gpiozero
```

## 🚀 啟動說明

1. **確認硬體連接**: 檢查 I2C 位址是否為 `0x40` (PCA9685)。
2. **準備權重檔案**: 確保目錄下有 `best.pt`。
3. **執行主程式**:
   ```bash
   python app.py
   ```
4. **存取網頁**: 開啟瀏覽器訪問 `http://<樹莓派IP>:5000`。

## 📂 專案結構

- `app.py`: Flask 後端、WebSocket 控制與馬達線程管理。
- `camera.py`: 影像擷取線程、YOLO 推理與號誌決策中心。
- `lane_detection.py`: 車道偵測演算法實作。
- `drive.py`: PCA9685 與馬達底層驅動封裝。
- `pid_controller.py`: PID 控制演算法類別。
- `templates/index.html`: 霓虹風動態控制介面。
