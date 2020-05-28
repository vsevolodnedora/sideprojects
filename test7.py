"""

    Radio Source

"""

import numpy as np

tau_norm = 1.0
nu_norm = 1.0

gam = 3.0
R = 1.0
D = 50
H_0 = 0.1
k_H = 1000.0
k_N = 0.0
m = 2.0
n = 0.0

task ='T'
nu_test = 0.1 # Test frequensy GGz
nu1 = 0.01
nu2 = 2.0
nupoints=100 # for spectrum
rpoints = 1000 # For distribution

points1 = 100  # Number elements in one tau[i]
points2 = 1000 # number of tau[i]
points3 = 1000 # number of intensites

precision = 100.0 # Maximum degree in exp(Mu*tau[i])

r=0.0
d=0.0
eps_0=0.0
mu_0=0.0
N_0=0.0

''' --- '''

def Limits(ro):
    return np.sqrt(r**2 - ro**2)

def Max(arr, size=0):
    """ finds max of arrauy [idx] """
    return np.argmax(arr)

def Min(arr, size=0):
    """ find min opf array [idx] """
    return np.argmin(arr)

def Middle(array, size=0):
    """ find middle of monotonic array [idx] """
    value = 0.5 * Max(array)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def Print(mass_x, mass_y, size, ts):
    max = Max(mass_y)
    max_x = mass_x[max]
    max_y =  mass_y[max]
    #
    fnmae = ""
    if ts == 1: fname = "spectrum_I.txt"
    elif ts == 2: fname = "spectrum_S.txt"
    elif ts == 3: fname = "distribution.txt"
    elif ts == 4: fname = "theta.txt"
    else: raise NameError("ts is not recognized")

    with open(fname, 'w') as the_file:
        if ts == 1:
            line = "# I( nu_max={} ) = {}  [SGSE] \n".format(max_x, max_y)
        elif ts == 2:
            line = "# S( nu_max={} ) = {}  [SGSE] \n".format(max_x, max_y)
        elif ts == 3:
            line = "# I_max = {}  [SGSE] \n".format(max_y)
        elif ts == 4:
            line = "# Theta_max = {} second of angle \n".format(max_y)
        else:
            raise NameError("ts is not recognized")

        the_file.write(line)

        norm_mass = mass_y / max_y
        for i_mass_x, i_mass_y, i_norm in zip(mass_x, mass_y, norm_mass):
            the_file.write("{} {} {} \n".format(i_mass_x, i_mass_y, i_norm))

        the_file.close()

def Integral(func, ro, a, b, nu):
    """
        computes integral of a faction, interval: ro \in [a, b] with
        fun(ro, nu)
    """
    h = (b-a) / points1
    x = a
    sum = 0

    for i in range(points1):
        x = x + h
        sum = sum + func(x, ro, nu)

    l_l = func(a, ro, nu)
    u_l = func(b, ro, nu)

    sum = sum + 0.5 * (l_l + u_l)
    sum = sum * h

    return sum

def H(x, ro):
    return 1/(1+k_H*pow((np.sqrt(x*x+ro*ro)/r),m))

def N(x, ro):
    return 1/(1+k_N*pow((np.sqrt(x*x+ro*ro)/r),n))

def Eps(nu):
    return N_0*eps_0*pow(nu,(1-gam)/2)

def Mu(nu):
    return N_0 * mu_0 * pow(nu, (-gam - 4) / 2)

def Func_Mu(x, ro, nu):
    return (pow(H(x, ro), (gam + 2) / 2)) * N(x, ro)

def Func_Eps(x, ro, nu):
    Ex = np.exp(Mu(nu) * Integral(Func_Mu, ro, x, Limits(ro), nu))
    return Eps(nu) / Ex

def Norm(nu_norm, nu, ro):
    return 1 / ((mu_0 * pow(nu_norm, (-4 - gam) / 2)) * Integral(Func_Mu, ro, -Limits(ro), Limits(ro), nu))

def Tau_Array(tau, x, a, b, ro, nu):

    for i in range(points2):
        x[i] = b - i * (b - a) / points2

    t = np.zeros(points2)
    for i in range(points2):
        t[i] = Integral(Func_Mu, ro, x[i], x[i - 1], nu)

    tau[0] = 0.0
    for i in range(points2):
        tau[i] = tau[i - 1] + t[i]

    return tau, x