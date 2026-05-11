import time
import threading
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform
from ultralytics import YOLO
from lane_detection import detect_lane

TARGET_FPS     = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS
SKIP_FRAMES    = 2


class CameraDevice:
    def __init__(self):
        try:
            self.model = YOLO('best.pt')
            print("YOLO 模型載入成功")
        except Exception as e:
            print(f"無法載入 YOLO 模型: {e}")
            self.model = None

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (320, 240)},
            transform=Transform(vflip=True, hflip=True)
        )
        self.picam2.configure(config)
        self.picam2.start()

        self.lock             = threading.Lock()
        self.frame            = None
        self.lane_error       = None
        self.traffic_light    = None   # 'red' | 'green' | None
        self._light_history   = []     # 記錄最近幾幀的燈號，用於時間過濾
        self.running          = True
        self._frame_count     = 0
        self._last_yolo_frame = None

        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        while self.running:
            loop_start = time.time()
            try:
                raw = self.picam2.capture_array()
                bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

                # ── YOLO（跳幀）──
                self._frame_count += 1
                if self.model is not None and self._frame_count % SKIP_FRAMES == 0:
                    results = self.model.predict(
                        source=bgr, imgsz=128,
                        conf=0.25, verbose=False, stream=False
                    )

                    # ── AI 號誌整合：紅綠燈狀態解析 ──────────────
                    detected_light = None
                    if results and results[0].boxes is not None:
                        boxes = results[0].boxes
                        for cls_id, conf in zip(boxes.cls, boxes.conf):
                            label = self.model.names[int(cls_id)]
                            if label in ('red', 'green') and float(conf) >= 0.25:
                                detected_light = label
                                break   # 取第一個高信心的燈號就夠

                    # ── 時間序列濾波 (Debouncing) 抑制閃爍 ──
                    self._light_history.append(detected_light)
                    if len(self._light_history) > 3:
                        self._light_history.pop(0)

                    # 連續 3 幀才改變狀態 (如：連續紅燈才停，或連續綠燈才走)
                    valid_light = None
                    if self._light_history.count('red') == 3:
                        valid_light = 'red'
                    elif self._light_history.count('green') == 3:
                        valid_light = 'green'
                    elif self.traffic_light is not None:
                        # 否則維持前一次狀態，除非連續 3 幀無辨識結果
                        if self._light_history.count(None) == 3:
                            valid_light = None
                        else:
                            valid_light = self.traffic_light

                    with self.lock:
                        self.traffic_light = valid_light
                    # ────────────────────────────────────────────

                    yolo_raw   = results[0].plot() if results else bgr.copy()
                    yolo_frame = cv2.cvtColor(yolo_raw, cv2.COLOR_RGB2BGR)
                    self._last_yolo_frame = yolo_frame
                    if results:
                        del results
                else:
                    yolo_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # ── 車道線偵測 ──
                lane_frame, error = detect_lane(bgr)

                # ── 合併左右兩張圖 ──
                combined = np.hstack([yolo_frame, lane_frame])

                ret, buf = cv2.imencode(
                    '.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 55])
                if ret:
                    with self.lock:
                        self.frame      = buf.tobytes()
                        self.lane_error = error

            except Exception as e:
                print(f"[Camera] {e}")

            elapsed = time.time() - loop_start
            sleep_t = FRAME_INTERVAL - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def get_frame(self):
        """回傳 (combined_jpeg, lane_error)"""
        with self.lock:
            return self.frame, self.lane_error

    def get_lane_frame(self):
        """app.py PID 用，只需要 error"""
        with self.lock:
            return self.frame, self.lane_error

    def get_traffic_light(self):
        """回傳最新紅綠燈狀態：'red' | 'green' | None"""
        with self.lock:
            return self.traffic_light

    def __del__(self):
        self.running = False
        try:
            self.picam2.stop()
        except Exception:
            pass


video_camera = CameraDevice()
