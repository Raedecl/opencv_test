from machine import Pin, I2C
import time

#配置IIC
class Gimbal:
    def __init__(self,scl_pin,sda_pin,freq_value = 100000):
        self.i2c=I2C(0, scl=Pin(scl_pin),sda=Pin(sda_pin), freq=freq_value)
        #设置IIC地址
        self.Servo_ADD = 0x2D
    #IIC控制舵机函数
    def IICServo(self,servonum, angle):
        self.i2c.writeto(self.Servo_ADD,bytearray([servonum,angle]))
        time.sleep(0.1)

if __name__ == '__main__':
    gimbal = Gimbal(33,25)
    gimbal.IICServo(1,0) 
    time.sleep(2)
    gimbal.IICServo(1,160)
    time.sleep(2)
