import random
import matplotlib.pyplot as plt
import numpy as np
import psutil
import time
import test2


def funct1():

    print(str(psutil.cpu_percent()))
    time.sleep(1)
    funct2()
    time.sleep(1)
    funct3()
    time.sleep(1)
    test2.funct_test2()

def funct2():
    
    print(str(psutil.cpu_percent()))

    return

def funct3():
    
    print(str(psutil.cpu_percent()))

    return

funct1()