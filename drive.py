import time
import math
import threading
import smbus2 as smbus
from gpiozero import LED


class PCA9685:
    __MODE1      = 0x00
    __PRESCALE   = 0xFE
    __LED0_ON_L  = 0x06
    __LED0_ON_H  = 0x07
    __LED0_OFF_L = 0x08
    __LED0_OFF_H = 0x09

    def __init__(self, address=0x40):
        self._lock = threading.Lock()  # ← I2C 保護鎖，防止多執行緒同時搶匯流排
        self.bus = smbus.SMBus(1)
        self.address = address
        self._raw_write(self.__MODE1, 0x00)

    def _raw_write(self, reg, value):
        """不加鎖的底層寫入，給內部已持鎖的方法用"""
        self.bus.write_byte_data(self.address, reg, value)

    def _raw_read(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def write(self, reg, value):
        with self._lock:
            self._raw_write(reg, value)

    def read(self, reg):
        with self._lock:
            return self._raw_read(reg)

    def setPWMFreq(self, freq):
        prescaleval = 25000000.0 / 4096.0 / float(freq) - 1.0
        prescale = math.floor(prescaleval + 0.5)
        with self._lock:
            oldmode = self._raw_read(self.__MODE1)
            self._raw_write(self.__MODE1, (oldmode & 0x7F) | 0x10)
            self._raw_write(self.__PRESCALE, int(prescale))
            self._raw_write(self.__MODE1, oldmode)
        time.sleep(0.005)
        with self._lock:
            self._raw_write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel, on, off):
        """4 個暫存器用同一把鎖一次寫完，避免中途被插隊"""
        base = self.__LED0_ON_L + 4 * channel
        with self._lock:
            self._raw_write(base,     on  & 0xFF)
            self._raw_write(base + 1, on  >> 8)
            self._raw_write(base + 2, off & 0xFF)
            self._raw_write(base + 3, off >> 8)

    def setDutycycle(self, channel, pulse):
        self.setPWM(channel, 0, int(pulse * (4095 / 100)))

    def setLevel(self, channel, value):
        self.setPWM(channel, 0, 4095 if value == 1 else 0)


class RobotControl:
    def __init__(self):
        self.pwm = PCA9685(0x40)
        self.pwm.setPWMFreq(50)
        self.CHANNELS = {
            'A_PWM': 0, 'A_IN1': 2, 'A_IN2': 1,
            'B_PWM': 5, 'B_IN1': 3, 'B_IN2': 4,
            'C_PWM': 6, 'C_IN1': 8, 'C_IN2': 7,
            'D_PWM': 11
        }
        self.motorD1 = LED(25)
        self.motorD2 = LED(24)

    def _set_left_motors(self, speed, direction='fwd'):
        is_fwd = (direction == 'fwd')
        self.pwm.setDutycycle(self.CHANNELS['A_PWM'], speed)
        time.sleep(0.001)  # 錯開啟動，避免瞬時電流尖峰
        self.pwm.setLevel(self.CHANNELS['A_IN1'], 0 if is_fwd else 1)
        self.pwm.setLevel(self.CHANNELS['A_IN2'], 1 if is_fwd else 0)
        time.sleep(0.001)
        self.pwm.setDutycycle(self.CHANNELS['C_PWM'], speed)
        self.pwm.setLevel(self.CHANNELS['C_IN1'], 1 if is_fwd else 0)
        self.pwm.setLevel(self.CHANNELS['C_IN2'], 0 if is_fwd else 1)

    def _set_right_motors(self, speed, direction='fwd'):
        is_fwd = (direction == 'fwd')
        time.sleep(0.002)  # 左馬達先啟動後，右馬達再啟動
        self.pwm.setDutycycle(self.CHANNELS['B_PWM'], speed)
        self.pwm.setLevel(self.CHANNELS['B_IN1'], 1 if is_fwd else 0)
        self.pwm.setLevel(self.CHANNELS['B_IN2'], 0 if is_fwd else 1)
        time.sleep(0.001)
        self.pwm.setDutycycle(self.CHANNELS['D_PWM'], speed)
        if is_fwd:
            self.motorD1.off()
            self.motorD2.on()
        else:
            self.motorD1.on()
            self.motorD2.off()

    def stop(self):
        for ch in [0, 5, 6, 11]:
            self.pwm.setDutycycle(ch, 0)
        self.motorD1.off()
        self.motorD2.off()

    def forward(self, s=60):
        self._set_left_motors(s, 'fwd');  self._set_right_motors(s, 'fwd')

    def backward(self, s=60):
        self._set_left_motors(s, 'rev');  self._set_right_motors(s, 'rev')

    def turn_left(self, s=50):
        self._set_left_motors(s, 'rev');  self._set_right_motors(s, 'fwd')

    def turn_right(self, s=50):
        self._set_left_motors(s, 'fwd');  self._set_right_motors(s, 'rev')

    def forward_left(self, s=60):
        self._set_left_motors(s * 0.3, 'fwd');  self._set_right_motors(s, 'fwd')

    def forward_right(self, s=60):
        self._set_left_motors(s, 'fwd');  self._set_right_motors(s * 0.3, 'fwd')

    def backward_left(self, s=60):
        self._set_left_motors(s * 0.3, 'rev');  self._set_right_motors(s, 'rev')

    def backward_right(self, s=60):
        self._set_left_motors(s, 'rev');  self._set_right_motors(s * 0.3, 'rev')