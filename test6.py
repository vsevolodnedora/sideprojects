from __future__ import division
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate


def find_nearest_index(array, value):
    ''' Finds index of the value in the array that is the closest to the provided one '''
    idx = (np.abs(array - value)).argmin()
    return idx

class Constants:

    light_v = np.float( 2.99792458 * (10 ** 10) )      # cm/s
    solar_m = np.float ( 1.99 * (10 ** 33)  )          # g
    solar_l = np.float ( 3.9 * (10 ** 33)  )           # erg s^-1
    solar_r = np.float ( 6.96 * (10 ** 10) )           #cm
    grav_const = np.float ( 6.67259 * (10 ** (-8) )  ) # cm3 g^-1 s^-2
    k_b     =  np.float ( 1.380658 * (10 ** (-16) ) )  # erg k^-1
    m_H     =  np.float ( 1.6733 * (10 ** (-24) ) )    # g
    m_He    =  np.float ( 6.6464764 * (10 ** (-24) ) ) # g
    c_k_edd =  np.float ( 4 * light_v * np.pi * grav_const * ( solar_m / solar_l ) )# k = c_k_edd*(M/L) (if M and L in solar units)

    yr      = np.float( 31557600. )
    smperyear = np.float(solar_m / yr)

    steph_boltz = np.float(5.6704*10**(-5))

#
light_v = np.float( 2.99792458 * (10 ** 10) )      # cm/s
solar_m = np.float ( 1.99 * (10 ** 33)  )          # g
solar_l = np.float ( 3.9 * (10 ** 33)  )           # erg s^-1
solar_r = np.float ( 6.96 * (10 ** 10) )           #cm
grav_const = np.float ( 6.67259 * (10 ** (-8) )  ) # cm3 g^-1 s^-2
constant_c_k_edd = np.float ( 4 * light_v * np.pi * grav_const * ( solar_m / solar_l ) )
glob_table_name = "./data/OPAL/table8.data"
n= 70
ngrid_rho = n
ngrid_temp = n
ngrid_kappa = n
#
def get_rho(r_arr, t_arr):
    cols = len(r_arr)  # 28
    raws = len(t_arr)  # 76

    rho = np.zeros((raws, cols))

    for i in range(raws):
        for j in range(cols):
            rho[i, j] = r_arr[j] + 3 * t_arr[i] - 18

    return rho

def read_opal_table(fname):
    f = open(fname, 'r').readlines()
    len1d = f.__len__()
    len2d = f[0].split().__len__()
    table = np.zeros((len1d, len2d))
    for i in range(len1d):
        table[i, :] = np.array(f[i].split(), dtype=float)

    r = table[0, 1:]
    t = table[1:, 0]
    rho = get_rho(r, t)
    k = table[1:, 1:]
    #
    temp = np.array(t)
    for i in range(len(rho[0,:])-1):
        temp = np.vstack((temp, t))
    temp = temp.T

    print(r.shape)
    print(t.shape)
    print(temp.shape)
    print(rho.shape)
    print(k.shape)
    return r, temp, rho, k
_, temp, rho, kappa = read_opal_table(glob_table_name)
#print(kappa)

''' --- getting 'kappa' interpoalted --- '''
grid_temp = np.mgrid[temp.min():temp.max():1j*ngrid_temp]
grid_rho = np.mgrid[rho.min():rho.max():1j*ngrid_rho]
grid_temp, grid_rho = np.meshgrid(grid_temp, grid_rho)
int_kappa = interpolate.griddata((temp.flatten(), rho.flatten()), kappa.flatten(),
                                 (grid_temp, grid_rho), method="cubic")
res = np.reshape(int_kappa, (len(grid_temp),len(grid_temp)))
print(int_kappa)
#print(grid_temp[0,:]); exit(1)
''' --- plotting 'kapopa' interpolated --- '''
fig, ax = plt.subplots(nrows=1, ncols=1)
levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
ax.contourf(grid_temp, grid_rho, int_kappa, levels = levels)
ax.set_title('Intepolated kappa')
ax.set_xlim(4.05, 5.6)
ax.set_ylim(-8.5, -4.9)
plt.tight_layout()
plt.show()

''' --- getting 'rho' intepolated --- '''
grid_temp = np.mgrid[temp.min():temp.max():1j*ngrid_temp]
grid_kappa = np.mgrid[rho.min():rho.max():1j*ngrid_kappa]
grid_temp, grid_kappa = np.meshgrid(grid_temp, grid_kappa)
res = interpolate.griddata((temp.flatten(), kappa.flatten()), rho.flatten(),
                           (grid_temp, grid_kappa), method="cubic")
int_rho = np.reshape(res, (len(grid_temp), len(grid_temp)))
print(int_rho)

''' --- plotting 'rho' intepolated --- '''
fig, ax = plt.subplots(nrows=1, ncols=2)
levels = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5]
ax[0].contourf(grid_temp, grid_rho, int_rho, levels = levels)
ax[0].set_title('Intepolated rho')
ax[0].set_xlim(4.05, 5.6)
ax[0].set_ylim(-0.347, 1.146)
#
ax[1].contourf(grid_temp, grid_rho, int_rho, levels = levels)
ax[1].set_title('Intepolated rho')
#ax[1].set_xlim(4.05, 5.6)
#ax[1].set_ylim(-0.347, 1.146)
plt.tight_layout()
plt.show()

''' --- converting k -> lm --- '''

def logk_loglm(logk, coeff=1.0):
    return np.log10(1 / (10 ** logk)) + np.log10(coeff * constant_c_k_edd)

lm_interp = logk_loglm(int_kappa)
grid_lm = logk_loglm(grid_kappa)

''' --- plotting 'kapopa' interpolated --- '''
fig, ax = plt.subplots(nrows=1, ncols=1)
levels = [4.0, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.50, 4.55]
ax.contourf(grid_temp, grid_rho, lm_interp, levels = levels)
ax.set_title('Intepolated L/M')
ax.set_xlim(4.05, 5.6)
ax.set_ylim(-10.5, -7.)
plt.tight_layout()
plt.show()

''' --- plotting 'rho' interpolated --- '''
fig, ax = plt.subplots(nrows=1, ncols=1)
levels = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5]
ax.contourf(grid_temp, grid_lm, int_rho, levels = levels)
ax.set_title('Intepolated rho ')
ax.set_xlim(4.05, 5.6)
ax.set_ylim(3.5, 4.60)
plt.tight_layout()
plt.show()

''' --- convert rho -> mdot --- '''

def sound_speed(t, mu, array=False):
    '''

    :param t_arr: log(t) array
    :param mu: mean molecular weight, by default 1.34 for pure He ionised matter
    :return: array of c_s (sonic vel) in cgs

    Test: print(Physics.sound_speed(5.2) / 100000) should be around 31
    '''

    if array:
        if len(mu) != len(t):
            raise ValueError('\t__Error. Mu and t must be arrays of the same size: (mu: {}; t: {})'.format(mu, t))
        res = np.zeros(len(t))
        for i in range(len(t)):
            res[i] = (np.sqrt(Constants.k_b * (10 ** t[i]) / (mu[i] * Constants.m_H))) / 100000
        return res
    else:
        return (np.sqrt(Constants.k_b * (10 ** t) / (mu * Constants.m_H))) / 100000
def rho_mdot(t, rho, dimensions=1, r_s=1., mu=1.34):
    '''
    NOTE! Rho2d should be .T as in all outouts it is not .T in Table Analyze
    :param t: log10(t[:])
    :param rho: log10(rho[:,:])
    :param r_s:
    :param mu:
    :return:
    '''

    # c = np.log10(4*3.14*((r_s * Constants.solar_r)**2) / Constants.smperyear)
    c = np.log10(4 * 3.14 * ((Constants.solar_r) ** 2) / Constants.smperyear) + np.log10(r_s ** 2)

    if int(dimensions) == 0:
        return (rho + c + np.log10(sound_speed(t, mu, False) * 100000))

    if int(dimensions) == 1:
        m_dot = np.zeros(len(t))
        for i in range(len(t)):
            m_dot[i] = (rho[i] + c + np.log10(sound_speed(t[i], mu, False) * 100000))
        return m_dot

    if int(dimensions) == 2:
        cols = len(rho[0, :])
        rows = len(rho[:, 0])
        m_dot = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                m_dot[i, j] = (rho[i, j] + c + np.log10(sound_speed(t[j], mu, False) * 100000))
        return m_dot
    else:
        raise ValueError('\t__Error. Wrong number of dimensions. Use 0,1,2. Given: {} | m_dot |'.format(dimensions))

int_mdot = rho_mdot(grid_temp[0,:], int_rho, dimensions=2, r_s=1.)

''' --- plotting 'mdot' interpolated --- '''
fig, ax = plt.subplots(nrows=1, ncols=1)
levels = [-5.5, -5.25, -5., -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3.]
ax.contourf(grid_temp, grid_lm, int_mdot, levels = levels)
ax.set_title('Intepolated mdot ')
ax.set_xlim(4.05, 5.6)
ax.set_ylim(3.5, 4.60)
plt.tight_layout()
plt.show()

''' --- getting minumum --- '''
tmin, tmax = 5.15, 5.6
lmmin, lmmax = 3.5, 4.6

tarr = grid_temp[0,:]
lmarr = grid_lm[:,0]

min_mdot, min_mdot_lm, min_mdot_t = [], [], []
for ilm, lm in enumerate(lmarr):
    if lm > lmmin and lm < lmmax:
        mdotarr = int_mdot[ilm, :]
        tmask = (tarr > tmin) & (tarr < tmax)
        mdot_mask = (~np.isnan(mdotarr))
        if len(mdotarr[mdot_mask & tmask]) > 0:
            min_mdot.append(mdotarr[mdot_mask & tmask].min())
            min_mdot_t.append(tarr[find_nearest_index(mdotarr, min_mdot[-1])])
            min_mdot_lm.append(lm)
        else:
            print("no mdot for ilm:{} lm:{}".format(ilm, lm))
min_mdot, min_mdot_lm, min_mdot_t = np.array(min_mdot), np.array(min_mdot_lm), np.array(min_mdot_t)

assert len(min_mdot) > 0
print(min_mdot)
print(min_mdot_t)
print(min_mdot_lm)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(min_mdot, min_mdot_lm, color='black', ls='-', label="Minimum Mdot")
ax.set_title('Minimum mdot')
ax.set_ylim(lmmin, lmmax)
ax.set_xlim(-6.0, -3.)
ax.set_xlabel("Mdot")
ax.set_ylabel("L/M")
# ax.invert_xaxis()
# plt.tight_layout()
plt.show()

''' --- OBSERVATIONS --- '''
table = []
fanme = "./data/gal_wne.data"
assert os.path.isfile(fanme)
with open(fanme, 'r') as f:
    for line in f:
        if '#' not in line.split() and line.strip():  # if line is not empty and does not contain '#'
            table.append(line)
names = table[0].split()
num_stars = len(table) - 1
table.remove(table[0])
num_v_n = ["N", "class", "t", "lRt", "v_inf", "X", "Eb-v", "Law","DM", "Mag","R*","mdot","l","eta","m"]
stars_mdot = []
stars_l = []
stars_m = []
for row in table:
    elemets = row.split()
    stars_mdot.append(float(elemets[num_v_n.index("mdot")]))
    stars_l.append(float(elemets[num_v_n.index("l")]))
    stars_m.append(float(elemets[num_v_n.index("m")]))
stars_mdot, stars_l, stars_m = np.array(stars_mdot), np.array(stars_l), np.array(stars_m)
stars_lm = lm = np.log10(10 ** stars_l / stars_m)
print(stars_m)
print(stars_l)
print(stars_lm)
print(stars_mdot)

''' --- plot stars minmdot --- '''

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(min_mdot, min_mdot_lm, color='black', ls='-', label="Minimum Mdot")
ax.plot(stars_mdot, stars_lm, marker="*", color="blue", linestyle="None")
ax.set_title('Minimum mdot')
ax.set_ylim(lmmin, lmmax)
ax.set_xlim(-6.0, -3.)
ax.set_xlabel("Mdot")
ax.set_ylabel("L/M")
# ax.invert_xaxis()
# plt.tight_layout()
plt.show()


''' interpolating stars ts '''
stars_t = []
for i in range(len(stars_mdot)):
    idx = find_nearest_index(min_mdot, stars_mdot[i])
    nearest_minmdot = min_mdot[idx]
    nearest_t = min_mdot_t[idx]
    nearest_lm = min_mdot_lm[idx]
    #
    i_lm = find_nearest_index(lmarr, stars_lm[i])
    mdot_row = int_mdot[i_lm, :]
    mdot_row = mdot_row[tarr>nearest_t]
    t_row = tarr[tarr>nearest_t]
    stars_t.append(interpolate.interp1d(mdot_row, t_row, kind="linear")(stars_mdot[i]))

''' --- plotting 'mdot' interpolated & stars --- '''
fig, ax = plt.subplots(nrows=1, ncols=1)
levels = [-5.5, -5.25, -5., -4.75, -4.5, -4.25, -4, -3.75, -3.5, -3.25, -3.]
ax.contourf(grid_temp, grid_lm, int_mdot, levels = levels)
ax.plot(stars_t, stars_lm, marker="*", color="blue", linestyle="None")
ax.set_title('Intepolated mdot ')
ax.set_xlim(4.05, 5.6)
ax.set_ylim(3.5, 4.60)
plt.tight_layout()
plt.show()

# fig, ax = plt.subplots(nrows=2, ncols=2)
# # Plot the model function and the randomly selected sample points
# ax[0,0].contourf(X, Y, T)
# ax[0,0].scatter(px, py, c='k', alpha=0.2, marker='.')
# ax[0,0].set_title('Sample points on f(X,Y)')
#
# # Interpolate using three different methods and plot
# for i, method in enumerate(('nearest', 'linear', 'cubic')):
#     Ti = griddata((px, py), f(px,py), (X, Y), method=method)
#     r, c = (i+1) // 2, (i+1) % 2
#     ax[r,c].contourf(X, Y, Ti)
#     ax[r,c].set_title("method = '{}'".format(method))
#
# plt.tight_layout()
# plt.show()