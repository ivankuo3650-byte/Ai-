import time
import threading
from flask import Flask, render_template, Response, jsonify
from flask_sock import Sock
from camera import video_camera
from drive import RobotControl
from pid_controller import PIDController

app  = Flask(__name__)
sock = Sock(app)
car  = RobotControl()
pid  = PIDController(kp=0.16, ki=0.0, kd=0.1)

_cmd_lock      = threading.Lock()
_current_dir   = "stop"
_last_cmd_time = time.time()
_cmd_seq       = 0
_auto_mode     = False
_base_speed    = 30         # 可由前端調整

COMMAND_TIMEOUT  = 1.0
AUTO_START_DELAY = 1.5      # 自動模式啟動後等待影像穩定的秒數
MAX_CORRECTION   = 25       # PID 最大修正量（限幅）
NO_LANE_TIMEOUT  = 0.8      # 超過此秒數沒偵測到車道 → 停車


def _motor_loop():
    prev_dir      = None
    prev_auto     = False
    last_pid_time = time.time()
    auto_start_t  = None     # 自動模式啟動時間
    last_lane_t   = time.time()  # 最後一次偵測到車道的時間

    manual_actions = {
        "forward":        lambda: car.forward(_base_speed),
        "backward":       lambda: car.backward(_base_speed),
        "left":           lambda: car.turn_left(_base_speed - 10),
        "right":          lambda: car.turn_right(_base_speed - 10),
        "forward_left":   lambda: car.forward_left(_base_speed),
        "forward_right":  lambda: car.forward_right(_base_speed),
        "backward_left":  lambda: car.backward_left(_base_speed),
        "backward_right": lambda: car.backward_right(_base_speed),
        "stop":           lambda: car.stop(),
    }

    while True:
        with _cmd_lock:
            auto    = _auto_mode
            dir_now = _current_dir
            last_t  = _last_cmd_time
            speed   = _base_speed

        if auto:
            # ── 自動模式剛啟動：等影像穩定 ──
            if not prev_auto:
                pid.reset()
                auto_start_t = time.time()
                last_lane_t  = time.time()
                prev_auto    = True
                car.stop()
                time.sleep(0.05)
                continue

            # 啟動延遲：等待 AUTO_START_DELAY 秒
            if time.time() - auto_start_t < AUTO_START_DELAY:
                time.sleep(0.05)
                continue

            now = time.time()
            dt  = max(now - last_pid_time, 0.01)
            last_pid_time = now

            _, error      = video_camera.get_lane_frame()
            traffic_light = video_camera.get_traffic_light()

            # ── 紅燈：強制停車，跳過 PID ──────────────────
            if traffic_light == 'red':
                car.stop()
                last_lane_t = time.time()   # 重置，避免觸發 NO_LANE_TIMEOUT 停車
                time.sleep(0.05)
                continue
            # ─────────────────────────────────────────────

            # 綠燈或無偵測 → 正常跑 PID
            if error is not None:
                last_lane_t = time.time()
                correction  = pid.compute(error, dt)
                # 限幅：最大修正量
                correction  = max(-MAX_CORRECTION, min(MAX_CORRECTION, correction))

                left_speed  = max(0, min(100, int(speed + correction)))
                right_speed = max(0, min(100, int(speed - correction)))

                try:
                    car._set_left_motors(left_speed, 'fwd')
                    car._set_right_motors(right_speed, 'fwd')
                except Exception as e:
                    print(f"[PID Motor] {e}")
            else:
                # 沒偵測到車道超過 NO_LANE_TIMEOUT → 停車保護
                if time.time() - last_lane_t > NO_LANE_TIMEOUT:
                    car.stop()

            time.sleep(0.05)

        else:
            # ── 手動模式 ──
            prev_auto = False

            if (time.time() - last_t) > COMMAND_TIMEOUT:
                dir_now = "stop"

            if dir_now != prev_dir:
                try:
                    if dir_now in manual_actions:
                        manual_actions[dir_now]()
                except Exception as e:
                    print(f"[Manual Motor] {e}")
                prev_dir = dir_now

            time.sleep(0.02)


threading.Thread(target=_motor_loop, daemon=True).start()


@app.route('/')
def index():
    return render_template('index.html')


def _gen_stream(get_fn):
    while True:
        frame, _ = get_fn()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            time.sleep(0.01)


@app.route('/video_feed')
def video_feed():
    return Response(
        _gen_stream(video_camera.get_frame),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    _, error = video_camera.get_lane_frame()
    with _cmd_lock:
        auto  = _auto_mode
        speed = _base_speed
    return jsonify({
        "auto":          auto,
        "error":         error,
        "dir":           _current_dir,
        "speed":         speed,
        "traffic_light": video_camera.get_traffic_light(),  # 新增燈號狀態
    })


@sock.route('/ws')
def ws_control(ws):
    global _current_dir, _last_cmd_time, _cmd_seq, _auto_mode, _base_speed
    print("[WS] 客戶端連線")
    try:
        while True:
            data = ws.receive()
            if not data:
                continue
            data = data.strip()

            if data == "auto_on":
                with _cmd_lock:
                    _auto_mode = True
                print("[WS] 自動駕駛開啟")

            elif data == "auto_off":
                with _cmd_lock:
                    _auto_mode = False
                    _current_dir = "stop"
                car.stop()
                print("[WS] 自動駕駛關閉")

            elif data.startswith("pid:"):
                try:
                    parts  = data.split(":")
                    pid.kp = float(parts[1])
                    pid.ki = float(parts[2])
                    pid.kd = float(parts[3])
                    pid.reset()
                    print(f"[PID] kp={pid.kp} ki={pid.ki} kd={pid.kd}")
                except Exception as e:
                    print(f"[PID] 參數解析錯誤: {e}")

            elif data.startswith("speed:"):
                try:
                    spd = int(data.split(":")[1])
                    with _cmd_lock:
                        _base_speed = max(30, min(100, spd))
                    print(f"[Speed] 速度更新: {_base_speed}")
                except Exception as e:
                    print(f"[Speed] 解析錯誤: {e}")

            else:
                if ':' in data:
                    direction, seq_str = data.split(':', 1)
                    try:
                        seq = int(seq_str)
                    except ValueError:
                        seq = 0
                else:
                    direction, seq = data, 0

                with _cmd_lock:
                    if seq >= _cmd_seq:
                        _cmd_seq       = seq
                        _current_dir   = direction
                        _last_cmd_time = time.time()

    except Exception:
        pass
    finally:
        with _cmd_lock:
            _current_dir = "stop"
        car.stop()
        print("[WS] 客戶端斷線，停車")


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    finally:
        car.stop()
