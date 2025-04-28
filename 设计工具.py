#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author : brother_yan

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

############################################################
#                         Analysis                         #
############################################################
def Yeq(C, R, CC): # 根据元件值求等效导纳。C[i], R[i]是第i级的电容与电阻，CC是最后一级的电容
    a = np.array([CC]) # Yeq == a[0] * s + a[1] * s^2 + ...
    for i in range(len(C) - 1, -1, -1):
        a = np.append([C[i][0] + C[i][1], C[i][0] * C[i][1] * (R[i][0] + R[i][1])], R[i][0] * R[i][1] * C[i][0] * C[i][1] * a)
    return a

def response(Rin, C, R, CC): # 根据元件值求传递函数。Rin是上电阻，C[i], R[i]是第i级的电容与电阻，CC是最后一级的电容
    a = Yeq(C, R, CC)
    b = [1]
    a = np.append([1], Rin * a)[::-1]
    return b / a[0], a / a[0]

############################################################
#                          Design                          #
############################################################
def all_pole_filter_design1(lpf_cutoff, lpf_order, Rin): # 全极点贝塞尔滤波器设计(让电容只有1种值)
    b, a = signal.bessel(lpf_order, lpf_cutoff, 'lowpass', analog = True) # bessel filter
    
    if len(b) != 1:
        raise Exception('不是全极点滤波器')
    b = b[::-1] # b[0] + b[1] * s + ...
    a = a[::-1] # a[0] + a[1] * s + ...
    b /= a[0] # 用a[0]归一化
    a /= a[0] # 用a[0]归一化
    Yeq = a[1:] / Rin # Yeq[0] * s + Yeq[1] * s^2 + ...
    
    C_list, R_list, CC = [], [], None
    while len(Yeq) >= 3:
        C = Yeq[0] / 2
        R1_and_R2 = Yeq[1] / (C**2)
        if len(Yeq) > 3:
            R1_mul_R2 = Yeq[2] / (2 * C**3)
        else:
            R1_mul_R2 = Yeq[2] / (C**3) # 最下面是单个电容
        R1 = R1_and_R2 / 2 + np.sqrt(R1_and_R2 ** 2 - 4 * R1_mul_R2) / 2
        R2 = R1_and_R2 / 2 - np.sqrt(R1_and_R2 ** 2 - 4 * R1_mul_R2) / 2
        Yeq = Yeq[2:] / (R1 * R2 * C**2)
        
        C_list.append([C, C])
        R_list.append([R1, R2])
    if len(Yeq) == 1:
        CC = Yeq[0]
    
    return C_list, R_list, CC

def all_pole_filter_design2(lpf_cutoff, lpf_order, Rin): # 全极点贝塞尔滤波器设计(让每个单元的电容只有1种值，电阻也只有1种值)
    b, a = signal.bessel(lpf_order, lpf_cutoff, 'lowpass', analog = True) # bessel filter
    
    if len(b) != 1:
        raise Exception('不是全极点滤波器')
    b = b[::-1] # b[0] + b[1] * s + ...
    a = a[::-1] # a[0] + a[1] * s + ...
    b /= a[0] # 用a[0]归一化
    a /= a[0] # 用a[0]归一化
    Yeq = a[1:] / Rin # Yeq[0] * s + Yeq[1] * s^2 + ...
    
    C_list, R_list, CC = [], [], None
    while len(Yeq) >= 3:
        C = Yeq[0] / 2
        R = Yeq[1] / (2 * C**2)
        Yeq = Yeq[2:] / (R**2 * C**2)
        
        C_list.append([C, C])
        R_list.append([R, R])
    if len(Yeq) == 1:
        CC = Yeq[0]
    
    return C_list, R_list, CC




############################################################
#                           Main                           #
############################################################
if __name__ == '__main__':
    # Datron4910的主滤波器元件值
    Rin = 78.7e3
    C = [[1e-6, 1e-6], [1e-6, 1e-6], [1e-6, 1e-6]]
    R = [[118e3, 26.7e3], [11.8e3, 45.3e3], [9.53e3, 14.0e3]]
    CC = 1e-6
    
    # 设计滤波器
    lpf_cutoff = 35
    lpf_order = 7
    Rin = 78.7e3
    
    C, R, CC = all_pole_filter_design1(lpf_cutoff, lpf_order, Rin)
    
    # 打印元件值
    print('|    C1    |    C2    |     R1    |     R2    |')
    for i in range(len(C)):
        print('| %5.2f uF | %5.2f uF | %6.2f kΩ | %6.2f kΩ |' % (C[i][0] * 1e6, C[i][1] * 1e6, R[i][0] / 1e3, R[i][1] / 1e3))
    print('Z = %.2f uF' % (CC * 1e6))
    
    # 画频率响应图
    b, a = response(Rin, C, R, CC)
    w, h = signal.freqs(b, a)
    f = w / (2 * np.pi)
    plt.semilogx(f, 20 * np.log10(abs(h)), 'r')
    plt.xlabel('f(Hz)')
    plt.ylabel('A(dB)')
    plt.grid(which = 'both')
    plt.show()
