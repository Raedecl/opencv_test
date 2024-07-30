from machine import SoftI2C,Pin
import time
import struct
import _thread
import socket

class Car:
    def __init__(self,sda_pin,scl_pin,freq_value = 100000):
        self.i2c = SoftI2C(sda=Pin(sda_pin),scl=Pin(scl_pin),freq = freq_value)
        #初始化方法，init在创建类时自动调用一次，以完成实例参数初始化
        self.devices = self.i2c.scan()#扫描本地设备

    def go_straight_ahead(self,direction = 'forward'):#前后移动
        if direction == 'backward':
            for device in self.devices:
                self.i2c.writeto(device,b'\x5A')
        else:
            for device in self.devices:
                self.i2c.writeto(device,b'\xA5')

    def go_straight_ahead_vertically(self,direction = 'left'):#左右螃蟹式垂直移动
        if direction == 'right':
            for device in self.devices:
                self.i2c.writeto(device,b'\x69')
        else:
            for device in self.devices:
                self.i2c.writeto(device,b'\x96')

    def stop(self,delay = 0):#停止移动（可选带延时）
        time.sleep(delay)
        for device in self.devices:
                self.i2c.writeto(device,b'\x00')
    
    
    
if __name__ == '__main__':    
    car = Car(14,15)
    car.go_straight_ahead(direction = 'forward')
    car.stop(1.5)
    car.go_straight_ahead(direction = 'backward')
    car.stop(0.5)
    car.go_straight_ahead_vertically(direction = 'left')
    car.stop(1.5)
    car.go_straight_ahead_vertically(direction = 'right')
    car.stop(1.5)