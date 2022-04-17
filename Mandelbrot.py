import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial

def isCard(x, y):
    p = (x - 0.25)**2 + y**2
    teta = np.arctan2(y, x-0.25)
    pc = 0.5 - 0.5*np.cos(teta)
    if pc**2 > p:
        return True
    else:
        return False

def isMandelbrot(x0, y0, checks):
    if isCard(x0, y0):
        return True
    x = x0
    y = y0
    
    x_old = x0
    y_old = y0
    period = 0
    
    for i in range(checks):
        if x**2 + y**2 > 4:
            return False
        x_new = x**2 - y**2 + x0
        y_new = 2*x*y + y0
        x = x_new
        y = y_new
        
        if x == x_old and y == y_old:
            return True
        
        period += 1
        if period == 20:
            period = 0
            x_old = x
            y_old = y
        
    if x**2 + y**2 > 4:
        return False
    return True

def find_members(side, center, dens, checks):
    y_min = center[1]-side/2
    y_max = center[1]+side/2
    x_min = center[0]-side/2
    x_max = center[0]+side/2
    x_members = []
    y_members = []
    for y in tqdm(np.linspace(y_min, y_max, dens)):
        for x in np.linspace(x_min, x_max, dens):
            if isMandelbrot(x, y, checks):
                x_members.append(x)
                y_members.append(y)
    return x_members, y_members

def find_members_in_row(y, x_min, x_max, dens, checks):
    x_members = []
    y_members = []
    for x in np.linspace(x_min, x_max, dens):
        if isMandelbrot(x, y, checks):
            x_members.append(x)
            y_members.append(y)
    return x_members, y_members

def find_members_parallel(side, center, dens, checks, workers):
    y_min = center[1]-side/2
    y_max = center[1]+side/2
    x_min = center[0]-side/2
    x_max = center[0]+side/2
    x_members = []
    y_members = []
    with Pool(workers) as p:
        partial_find_members_in_row = partial(find_members_in_row, x_min=x_min, x_max=x_max, dens=dens, checks=checks)
        res = list(tqdm(p.imap(partial_find_members_in_row, np.linspace(y_min, y_max, dens)), total=dens))
        for members in res:
            x_members += members[0]
            y_members += members[1]
    return x_members, y_members

def plot_mandelbrot(side, center, x_members, y_members):
    y_min = center[1]-side/2
    y_max = center[1]+side/2
    x_min = center[0]-side/2
    x_max = center[0]+side/2
    
    plt.rcParams["figure.figsize"] = (20,20)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.scatter(x_members, y_members, s=1, c = 'black', cmap="binary", marker='s')
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()