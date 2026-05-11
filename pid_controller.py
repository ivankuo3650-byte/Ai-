class PIDController:
    """
    PID 控制器
    輸入：偏差量 e（車道中心 - 影像中心）
    輸出：轉向修正量 u（正值右轉，負值左轉）
    """
    def __init__(self, kp=0.4, ki=0.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self._prev_error  = 0.0
        self._integral    = 0.0
        self._integral_max = 100.0  # 防止積分飽和

    def reset(self):
        self._prev_error = 0.0
        self._integral   = 0.0

    def compute(self, error: float, dt: float = 0.1) -> float:
        """
        計算 PID 輸出
        error: 偏差量（像素）
        dt:    時間間隔（秒）
        回傳:  轉向修正量
        """
        self._integral += error * dt
        # 積分限幅，避免長時間累積造成失控
        self._integral = max(-self._integral_max,
                             min(self._integral_max, self._integral))

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return output