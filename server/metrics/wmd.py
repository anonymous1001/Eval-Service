#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import scipy.spatial.distance as distance


def wms(lst_a, lst_b):
    wms = 0
    for x in lst_a:
        max_sim = sys.float_info.min
        for y in lst_b:
            d = 1 - distance.cosine(x, y)
            if d > max_sim:
                max_sim = d
        wms += max_sim
    return float(wms) / (len(lst_a) + len(lst_b))
