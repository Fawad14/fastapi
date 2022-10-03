# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 02:53:12 2022

@author: fawad
"""

from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class Maintenance(BaseModel):
    TEMPRATURE: float 
    VOLTAGE: float 
    current: float 
    POWER: float