import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helperFunctions as hf
from scipy.linalg import eigh


class Mes2d:
    def __init__(self, *, N:int, L:float, omega:float, m:float = hf.m, in_nm:bool=True,
        nlg_fname:str   = "data/nlg_N_2_L_100.dat",
        node_fname:str  = "data/wezly_N_2_L_100.dat"):

        self.i_size:int = 9
        self.L:float    = L
        self.N:int      = N
        self.a:float    = 0.5*L/N
        self.m:float    = m

        self.in_nm = in_nm

        self.nlg_vs_ki_fname = nlg_fname
        self.coor_vs_nlg_fname = node_fname
        self.omega = omega
        self.x_nlg, self.y_nlg, self.k_size, self.nlg = hf.create_coor_ki(self.nlg_vs_ki_fname, self.coor_vs_nlg_fname)
        self.nlg_size = np.max(self.nlg)
        self.is_edge = hf.edges_like(self.nlg)
        
        self.s_loc = hf.gen_s_loc(self.i_size, a = self.a)
        self.t_loc = hf.gen_t_loc(self.i_size)
        self.S = self.H = np.zeros([self.nlg_size, self.nlg_size])


        self.gen_S_H()
        

    def get_X_nlg(self, k:int, i:int):
        """Gets the value of x_nlg(k,i) in au; indexed from 1"""
        v = self.x_nlg[k-1, i-1]
        if self.in_nm:
            v *= hf.nm_to_au
        return v 
    def get_Y_nlg(self, k:int, i:int):
        """Gets the value of y_nlg(k,i) in au; indexed from 1"""
        v = self.y_nlg[k-1, i-1]
        if self.in_nm:
            v *= hf.nm_to_au
        return v
    def get_nlg(self, k:int, i:int):
        """Gets the value of nlg(k,i); indexed from 1"""
        return self.nlg[k-1, i-1]
    
    
    def x_real(self, k:int, xi1:float, xi2:float):
        x = 0
        for i in range(1,5):
            x += self.get_X_nlg(k,i) * hf.g(i, xi1, xi2)
        return x
    
    def y_real(self, k:int, xi1:float, xi2:float):
        y = 0
        for i in range(1,5):
            y += self.get_Y_nlg(k,i) * hf.g(i, xi1, xi2)
        return y
    

    ### s_ji matrix
    def get_s_at(self, i1:int, i2:int):
        return self.s_loc[i1-1, i2-1]
    
    def get_s(self):
        return self.s_loc
    

    ### t_ji matrix
    def get_t_at(self, i1:int, i2:int):
        return self.t_loc[i1-1, i2-1]
    
    def get_t(self):
        return self.t_loc
    
    
    def v(self, k:int, j:int, i:int):
        sum = 0
        for l in range(4):
            for n in range(4):
                sum += hf.w[l]*hf.w[n] * (self.x_real(k, hf.p[l], hf.p[n])**2 + self.y_real(k, hf.p[l], hf.p[n])**2) * hf.h(j, hf.p[l], hf.p[n]) * hf.h(i, hf.p[l], hf.p[n])
        sum = sum * 0.125*self.a**2 * self.m * self.omega**2
        return sum
    
    def get_v_at(self, k:int):
        v_loc = np.zeros([self.i_size, self.i_size])
        for i1 in range(1, self.i_size+1):
            for i2 in range(1, self.i_size+1):
                v_loc[i1-1, i2-1] = self.v(k, i1, i2)
        return v_loc
    
    def gen_S_H(self):
        self.H =self.S = np.zeros([self.nlg_size, self.nlg_size])
        for k in range(self.k_size):
            for i1 in range(1,10):
                if self.is_edge[k-1, i1-1]: continue
                for i2 in range(1,10):
                    if self.is_edge[k-1, i2-1]: continue
                    self.S[self.get_nlg(k, i1)-1, self.get_nlg(k,i2)-1] += self.get_s_at(i1, i2)
                    self.H[self.get_nlg(k, i1)-1, self.get_nlg(k,i2)-1] += self.get_t_at(i1, i2) + self.v(k,i1,i2)
        # for i in range(self.nlg_size):
            # self.S[i, i] = 1
            # self.H[i, i] = -1410
        
        # eigvals, P = np.linalg.eig(self.S)
        # self.S = np.diag(eigvals)
        # eigvals, P = np.linalg.eig(self.H)
        # self.H = np.diag(eigvals)
        # Eigenvalues of S
        # w = np.linalg.eigvalsh(self.S)
        # print("min eig(S) =", w.min())
        # print("max eig(S) =", w.max())
    
    def get_S_H(self):
        """Returns S and H matrices"""
        return self.S, self.H
    

    def get_psi(self, i:int=0):
        E, c = eigh(-self.H, -self.S)
        return E[i], c[:,i]
