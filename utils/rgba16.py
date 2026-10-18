"""显式声明的16位RGBA输入适配；保持原RGB/Alpha浮点精度。"""
import cv2
import numpy as np
import torch


class FloatRGBA:
    def __init__(self,path,white_background=False,no_bg=True):
        raw = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.ndim != 3 or raw.shape[-1] != 4:
            raise ValueError('rgba16输入必须为真正的uint16四通道PNG：'+str(path))
        self.array = raw[...,[2,1,0,3]].astype(np.float32)/65535.
        self.saturated = (raw[...,[2,1,0]] == 65535) & (raw[...,3:4] > 0)
        if no_bg:
            alpha = self.array[...,3:4]
            self.array[...,:3] = self.array[...,:3]*alpha+float(white_background)*(1-alpha)
        self.size = (raw.shape[1],raw.shape[0])

    def to_tensor(self,resolution):
        # AREA非负权重使缩小后的满码观测仍可解释为下界；不经过8位量化。
        resized = cv2.resize(self.array,resolution,interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2,0,1).clamp(0,1)

    def saturation_tensor(self,resolution):
        fraction = cv2.resize(self.saturated.astype(np.float32),resolution,interpolation=cv2.INTER_AREA)
        return torch.from_numpy(fraction>0).permute(2,0,1)
