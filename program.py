from mes import Mes2d
import matplotlib.pyplot as plt
import numpy as np
import helperFunctions as hf

L_nm = 100
omega_meV = 10
L_au = L_nm * hf.nm_to_au
omega_au = omega_meV * hf.meV_to_au
a = Mes2d(N=2, L=L_au, omega=omega_au)

E, psi = a.get_psi(0)
psi_2d = np.zeros([9,9])
x = y = np.zeros([81])
for k in range(1, 16):
    for i in range(1,9):
        nlg =a.get_nlg(k, i)-1
        xi = nlg%9
        yi = nlg//9
        psi_2d[xi,yi]
        # print(xi, yi," - ",nlg," - ",a.x_real(k, hf.p[0], hf.p[0]),a.y_real(k, hf.p[0], hf.p[0]))
plt.imshow(psi_2d)
plt.colorbar()
plt.savefig("plots/psi.png")
plt.clf()
S, H = a.get_S_H()


def read_file(fname):
    data = np.genfromtxt(fname, delimiter=",")
    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    v = data[:, 2]

    nx = x.max()
    ny = y.max()

    arr = np.empty((ny, nx))
    arr[x-1, y-1] = v
    return arr


############ s
s_loc_theory = read_file("data/S_loc_N2_L100.dat")
s_loc = a.get_s()

plt.imshow(s_loc, cmap="inferno")
plt.colorbar()
plt.savefig('plots/s_loc.png')
plt.clf()
plt.imshow(s_loc_theory, cmap="inferno")
plt.colorbar()
plt.savefig("plots/s_loc_theoretical.png")
plt.clf()

ds_loc = (s_loc_theory-s_loc)/s_loc_theory*100
cbar = plt.imshow(ds_loc, cmap="inferno")
plt.title("(s_theory - s)/s_theory [%]")
plt.colorbar()
plt.savefig("plots/ds_loc.png")
plt.clf()



########### t
t_loc_theory = read_file("data/T_loc_N2_L100.dat")
t_loc = a.get_t()

plt.imshow(t_loc, cmap="inferno")
plt.colorbar()
plt.savefig('plots/t_loc.png')
plt.clf()

plt.imshow(t_loc_theory, cmap="inferno")
plt.colorbar()
plt.savefig("plots/t_loc_theoretical.png")
plt.clf()

dt_loc = (t_loc_theory-t_loc)/t_loc_theory

plt.imshow(dt_loc, cmap="inferno")
plt.title("(t_theory - t)/t_theory")
plt.colorbar()
plt.savefig("plots/dt_loc.png")
plt.clf()

t_ratio = t_loc_theory/t_loc


########### v
v_loc_theory = read_file("data/V_loc_N2_L100_element_11.dat")
v_loc = a.get_v_at(11)

plt.imshow(v_loc_theory, cmap="inferno")
plt.colorbar()
plt.savefig("plots/v_loc_11_theoretical.png")
plt.clf()

plt.imshow(v_loc, cmap="inferno")
plt.colorbar()
plt.savefig('plots/v_loc_11.png')
plt.clf()

dv_loc = (v_loc_theory - v_loc)/v_loc_theory

plt.imshow(dv_loc, cmap="inferno")
plt.colorbar()
plt.title('(v_theory - v)/v_theory')
plt.savefig('plots/dv_loc_11.png')
plt.clf()

# v_theory_over_v = v_loc_theory/v_loc

# plt.imshow(S, cmap="inferno")
# plt.colorbar()
# plt.savefig('plots/S.png')
# plt.clf()

# plt.imshow(H, cmap="inferno")
# plt.colorbar()
# plt.savefig('plots/H.png')
# plt.clf()

