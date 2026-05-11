import cv2
import numpy as np

LAST_ERROR   = 0
LANE_WIDTH_PX = 180  # 放寬預設車道寬度（原本80太窄會讓車子貼著單邊線走，請依實際跑道寬度微調）

def detect_lane(frame) -> tuple:
    global LAST_ERROR

    height, width = frame.shape[:2]
    img_center = width // 2

    # 1. 灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. ROI (將畫面上半部裁切掉更多，避開窗框與遠處反光，只保留底部真實地板)
    roi_top = int(height * 0.55)
    roi_vertices = np.array([[
        (0,     height),
        (0,     roi_top),
        (width, roi_top),
        (width, height),
    ]], dtype=np.int32)
    roi_mask = np.zeros_like(gray)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    gray_roi = cv2.bitwise_and(gray, roi_mask)

    # 3. 捨棄頂帽變換，改回直接閾值法 (Thresholding)
    #    近距離的車道線太粗，頂帽會將其核心掏空導致無法偵測。
    #    改用 190 作為高亮閾值，專注抓取極亮的實體車道線。
    _, white_mask = cv2.threshold(gray_roi, 190, 255, cv2.THRESH_BINARY)

    # 4. 白光雜訊抑制 (形態學效能優化)
    #    - 注意：樹莓派 CPU 無法負荷大型 kernel，用 25x25 會導致極大延遲與當機！
    #    - 改用輕量的 3x3 與 5x5 修整邊緣，比較大的破洞直接交給 HoughLinesP 來跨越接合。
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    # 5. Hough 轉換 (使用 maxLineGap 接合破洞，直接取代耗效能的大型形態學運算)
    lines = cv2.HoughLinesP(
        cleaned, 1, np.pi / 180,
        threshold=25,
        minLineLength=20,
        maxLineGap=120
    )

    result = frame.copy()
    ref_y = int(height * 0.8)
    left_xs, right_xs = [], []

    if lines is not None:
        valid_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # 放寬斜率限制，允許接近垂直與平緩彎道
            if abs(slope) < 0.05:
                continue

            # 計算線段延伸至底部基準線 (ref_y) 的 x 座標
            x_ref = int(x1 + (ref_y - y1) / slope)
            valid_lines.append((x_ref, slope, x1, y1, x2, y2))

        if valid_lines:
            valid_lines.sort(key=lambda item: item[0])
            min_x_ref = valid_lines[0][0]
            max_x_ref = valid_lines[-1][0]

            # 判斷是否「同時拍到雙線」(左右測群集差距大於車寬 40%)
            if (max_x_ref - min_x_ref) > (LANE_WIDTH_PX * 0.4):
                mid_point = (min_x_ref + max_x_ref) / 2
                for (x_ref, slope, x1, y1, x2, y2) in valid_lines:
                    if x_ref < mid_point:
                        left_xs.append(x_ref)
                        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        right_xs.append(x_ref)
                        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                # 單線模式：除了參考斜率外，強迫加入「座標位置」輔助判斷
                # 解決 S 彎中，內彎線因為斜率而被誤認為反方向線的致命問題
                avg_slope = np.mean([item[1] for item in valid_lines])
                avg_x     = np.mean([item[0] for item in valid_lines])
                
                is_left = False
                if avg_x < img_center * 0.7:   # 明顯偏左，強判為左線
                    is_left = True
                elif avg_x > img_center * 1.3: # 明顯偏右，強判為右線
                    is_left = False
                else:                          # 處於中間地帶，回歸斜率判斷
                    is_left = (avg_slope < 0)

                for (x_ref, slope, x1, y1, x2, y2) in valid_lines:
                    if is_left:
                        left_xs.append(x_ref)
                        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        right_xs.append(x_ref)
                        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    error = None

    # ── 雙線：走中間 ────────────────────────────────
    if left_xs and right_xs:
        x_left      = int(np.mean(left_xs))
        x_right     = int(np.mean(right_xs))
        lane_center = (x_left + x_right) // 2
        error       = lane_center - img_center

        cv2.circle(result, (lane_center, ref_y), 6, (255, 255, 0), -1)
        cv2.arrowedLine(result,
            (img_center, ref_y), (lane_center, ref_y),
            (255, 0, 255), 2, tipLength=0.3)
        cv2.putText(result, f"BOTH  err:{error:+d}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ── 單左線：車道中心在左線右邊半個車道寬 ──────────
    elif left_xs:
        x_left      = int(np.mean(left_xs))
        lane_center = x_left + LANE_WIDTH_PX // 2
        error       = lane_center - img_center

        cv2.circle(result, (lane_center, ref_y), 6, (0, 255, 0), -1)
        cv2.arrowedLine(result,
            (img_center, ref_y), (lane_center, ref_y),
            (255, 0, 255), 2, tipLength=0.3)
        cv2.putText(result, f"LEFT  err:{error:+d}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ── 單右線：車道中心在右線左邊半個車道寬 ──────────
    elif right_xs:
        x_right     = int(np.mean(right_xs))
        lane_center = x_right - LANE_WIDTH_PX // 2
        error       = lane_center - img_center

        cv2.circle(result, (lane_center, ref_y), 6, (0, 0, 255), -1)
        cv2.arrowedLine(result,
            (img_center, ref_y), (lane_center, ref_y),
            (255, 0, 255), 2, tipLength=0.3)
        cv2.putText(result, f"RIGHT err:{error:+d}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ── 無線：停車 ──────────────────────────────────
    else:
        cv2.putText(result, "NO LANE - STOP",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 6. 平滑濾波
    if error is not None:
        alpha  = 0.6
        error  = int(alpha * error + (1 - alpha) * LAST_ERROR)
        LAST_ERROR = error

    # UI
    cv2.polylines(result, roi_vertices, True, (80, 80, 80), 1)
    cv2.line(result, (img_center, height), (img_center, roi_top),
             (200, 200, 200), 1)

    return result, error