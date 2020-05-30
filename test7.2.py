"""

    Radio Source

"""

import numpy as np
from scipy import special
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt

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

task ='i'
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

''' ------------- '''

def Limits(ro):
    return np.sqrt(r**2 - ro**2)

def Integral(func, a, b, args, plot=False):
    if plot:
        xarr = np.mgrid[a:b:1000*1j]
        yarr = func(xarr, args)
        print(yarr)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xarr, yarr)
        plt.show()
        plt.close()

    int = integrate.quad(func, a, b, args=args)
    res, err = int
    return res

def Eps(nu):
    return N_0 * eps_0 * pow(nu,(1.-gam)/2.)

def Mu(nu):
    return N_0 * mu_0 * pow(nu, (-gam - 4.) / 2.)

def H(x, ro):
    return 1 / (1 + k_H * pow((np.sqrt(x**2 + ro**2) / r), m))

def N(x, ro):
    return 1 / (1 + k_N * pow((np.sqrt(x**2 + ro**2) / r), n))

def Func_Mu(x, ro):
    return (pow(H(x, ro),(gam + 2.) / 2.)) * N(x, ro)

def Func_Eps(x, ro):
    return (pow(H(x, ro),(gam + 1.) / 2.)) * N(x, ro)

def Norm():
    ro = 0.0
    return 1. / ((mu_0 * pow(nu_norm,(-4-gam)/2.)) * Integral(Func_Mu, -r, r, args=ro, plot=False))

def Tau_Array(a, b, ro):

    tau = np.zeros(points2+1)
    x = np.zeros(points2+1)

    for i in range(points2):
        x[i]=b-i*(b-a)/float(points2)

    t = np.zeros(points2+1)
    for i in range(1, points2+1):
        t[i]=Integral(Func_Mu, x[i], x[i-1], args=ro, plot=False)

    tau[0] = 0.0
    for i in range(1, points2+1):
        tau[i] = tau[i - 1] + t[i]
        #print(tau[i])

    return tau, x

def Int_Inten(Func_Eps, ro, nu):

    a = -1. * Limits(ro)
    b = Limits(ro)

    mu = Mu(nu)
    eps = Eps(nu)

    tau, x = Tau_Array(a, b, ro)

    sum = 0.0
    for i in range(points2):
        sum += Func_Eps(x[i], ro) * (np.exp(-mu * tau[i]))

    l_l = Func_Eps(x[points2], ro) * np.exp(-mu * tau[points2])
    u_l = Func_Eps(x[0], ro) * np.exp(-mu * tau[0])

    sum += 0.5 * (l_l + u_l)
    sum *= (eps * (b - a) / (1.*points2))

    return sum

def Intensity(ro, nu):
    return Int_Inten(Func_Eps, ro, nu)

def Func_Rad_Flux_Dens(x, nu):
    return x * Intensity(x, nu)

def Integ_Rad_Flux_Density(Func, a, b, nu):
    #return Integral(Func, Intensity(a, nu), Func(b, nu), args=nu)

    # Rectangle method
    h = (b - a) / points3
    x = a
    sum = 0.0

    for i in range(1, points3):
        x += h
        sum += Func(x, nu)

    l_l = Intensity(a, nu)
    u_l = Func(b, nu)

    sum += 0.5 * (l_l + u_l)
    sum *= h

    return sum

def Rad_Flux_Density(nu):
    return 2 * np.pi * (1. / (d * d)) * Integ_Rad_Flux_Density(Func_Rad_Flux_Dens, 0, r, nu)

''' ------------- '''


def constants():
    e = 4.8032e-10
    me = 9.10953e-28
    c = 2.99792458e10
    pars = 3.08567758128e18
    ggz = pow(10, 9)

    r = R * pars
    d = D * pars * pow(10, 6)

    c1 = (3. * e) / (4. * np.pi * pow(me, 3.) * pow(c, 5.))
    a = special.gamma((3 * gam - 1) / 12.) * special.gamma((3 * gam + 7) / 12.)
    c5 = (np.sqrt(3.) * pow(e, 3.) / (16. * np.pi * me * pow(c, 2.))) * ((gam + (7. / 3.)) / (gam + 1.)) * a
    b = special.gamma((3 * gam + 2.) / 12.) * special.gamma((3 * gam + 10.) / 12.)
    c6 = np.sqrt(3.) * np.pi * e * pow(me, 5.) * pow(c, 10.) * (gam + (10. / 3.)) * b / 72.

    eps_0 = c5 * pow(H_0, (gam + 1.) / 2.) * pow(ggz / (2. * c1), (1. - gam) / 2.)
    mu_0 = c6 * pow(H_0, (gam + 2.) / 2.) * pow((ggz / (2. * c1)), (-gam - 4.) / 2.)

    return r, d, eps_0, mu_0

if __name__ == "__main__":

    r, d, eps_0, mu_0 = constants()
    N_0 = Norm()

    print("eps_0: {} [expected {} ]".format(eps_0, 9.42026e-16))
    print("mu_0:  {} [expected {} ]".format(mu_0, 5.54025e-08))

    print("N_0:     {} [expected {} ]".format(N_0, 1.38734e-10))
    print("eps({}): {} [expected {} ]".format(nu_test, Eps(nu_test), 1.30691e-24))
    print("mu({}):  {} [expected {} ]".format(nu_test, Mu(nu_test), 2.43059e-14))

    print("I({}):   {} [expected {} ]".format(nu_test, Intensity(0.0, nu_test), 3.24802e-10))
    print("S({}):   {} [expected {} ]".format(nu_test, Rad_Flux_Density(nu_test), 6.52387e-26))