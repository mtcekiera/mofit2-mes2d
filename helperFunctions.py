import numpy as np
import pandas as pd

hbar = 1
m = 0.067

au_to_nm = 0.05292
nm_to_au = 1/au_to_nm
au_to_meV = 27211.6
meV_to_au = 1/au_to_meV

w12 = (18+np.sqrt(30))/36
w34 = (18-np.sqrt(30))/36
w = [w12, w12, w34, w34]

p = [
    -np.sqrt(3/7 - 2/7 * np.sqrt(1.2)),
    np.sqrt(3/7 - 2/7 * np.sqrt(1.2)),
    np.sqrt(3/7 + 2/7 * np.sqrt(1.2)),
    -np.sqrt(3/7 + 2/7 * np.sqrt(1.2))
]

def read_nlg_table(fname):
    data = pd.read_csv(fname, header=None)
    k_size = np.max(data[0])
    nlg = np.zeros([k_size, 9], dtype=int)
    nlg[data[0]-1, data[1]-1] = data[2]
    return nlg, k_size

def read_coor_nlg(fname):
    data = pd.read_csv(fname, header=None)
    x = np.array(data[1])
    y = np.array(data[2])
    return x,y

def create_coor_ki(nlg_vs_ki_fname:str, coor_vs_nlg_fname:str):
    nlg, k_size = read_nlg_table(nlg_vs_ki_fname)
    x, y = read_coor_nlg(coor_vs_nlg_fname)
    x_ki, y_ki = x[nlg-1], y[nlg-1]
    return x_ki, y_ki, k_size, nlg



def f1(xi:float):
    return 0.5*(1-xi)
def f2(xi:float):
    return 0.5*(1+xi)

def g(no: int, xi1:float, xi2:float):
    if no==1:
        return f1(xi1) * f1(xi2)
    if no==2:
        return f2(xi1) * f1(xi2)
    if no==3:
        return f1(xi1) * f2(xi2)
    if no==4:
        return f2(xi1) * f2(xi2)
    raise ValueError("Wrong function number")

def q1(xi:float):
    return xi*(xi-1)*0.5
def q2(xi:float):
    return (1-xi)*(1+xi)
def q3(xi:float):
    return xi*(xi+1)*0.5

def h(no:int, xi1, xi2):
    if no==1:
        return q1(xi1)*q1(xi2)
    if no==2:
        return q3(xi1)*q1(xi2)
    if no==3:
        return q1(xi1)*q3(xi2)
    if no==4:
        return q3(xi1)*q3(xi2)
    if no==5:
        return q2(xi1)*q1(xi2)
    if no==6:
        return q3(xi1)*q2(xi2)
    if no==7:
        return q1(xi1)*q2(xi2)
    if no==8:
        return q2(xi1)*q3(xi2)
    if no==9:
        return q2(xi1)*q2(xi2)
    raise ValueError("Wrong function number")


def dh_over_dxi1(i:int, l:int, n:int, delta:float = 0.001):
    return 0.5*(h(i, p[l]+delta, p[n]) - h(i, p[l]-delta, p[n]))/delta
def dh_over_dxi2(i:int, l:int, n:int, delta:float = 0.001):
    return 0.5*(h(i, p[l], p[n]+delta) - h(i, p[l], p[n]-delta))/delta

def edges_like(arr:np.ndarray):
    is_edge = np.zeros_like(arr)
    is_edge[0,:] = 1
    is_edge[:,0] = 1
    is_edge[-1,:] = 1
    is_edge[:,-1] = 1
    return is_edge

def s(j:int, i:int, *, a:float):
    sum = 0
    for l in range(4):
        for n in range(4):
            sum += w[l]*w[n] * h(j, p[l], p[n]) * h(i, p[l], p[n])
    # sum = sum*100000
    sum = 0.25*a**2*sum
    # return sum*1e34
    return sum


def t(j, i):
    sum = 0
    for l in range(4):
        for n in range(4):
            sum += w[l]*w[n]*(dh_over_dxi1(j, l, n) * dh_over_dxi1(i, l, n) + dh_over_dxi2(j, l, n)*dh_over_dxi2(i, l, n))
    return 0.5*sum*hbar/m
    

def gen_s_loc(i_size, *, a:float):
    s_loc = np.zeros([i_size, i_size])
    for i1 in range(1, i_size+1):
        for i2 in range(1, i_size+1):
            s_loc[i1-1,i2-1] = s(i1, i2, a=a)
    return s_loc

def gen_t_loc(i_size):
    t_loc = np.zeros([i_size, i_size])
    for i1 in range(1, i_size+1):
        for i2 in range(1, i_size+1):
            t_loc[i1-1,i2-1] = t(i1, i2)
    return t_loc
           

