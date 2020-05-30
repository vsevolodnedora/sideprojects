"""

    Radio Source

"""

import numpy as np
from scipy import special

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


e = 4.8032 * pow(10, -10)
me = 9.10953 * pow(10, -28)
c = 2.99792458 * pow(10, 10)
pars = 3.08567758128 * pow(10, 18)
ggz = pow(10, 9)

r = R * pars
d = D * pars * pow(10, 6)

c1 = (3. * e) / (4. * np.pi * pow(me, 3) * pow(c, 5))
a = special.gamma((3 * gam - 1) / 12.) * special.gamma((3 * gam + 7) / 12.)
c5 = (np.sqrt(3) * pow(e, 3) / (16. * np.pi * me * pow(c, 2))) * ((gam + (7. / 3.)) / (gam + 1.)) * a
b = special.gamma((3 * gam + 2.) / 12.) * special.gamma((3 * gam + 10.) / 12.)
c6 = np.sqrt(3) * np.pi * e * pow(me, 5) * pow(c, 10) * (gam + (10. / 3.)) * b / 72.

eps_0 = c5 * pow(H_0, (gam + 1) / 2.) * pow(ggz / (2. * c1), (1 - gam) / 2.)
mu_0 = c6 * pow(H_0, (gam + 2) / 2.) * pow((ggz / (2. * c1)), (-gam - 4) / 2.)



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

def Frequency(nu, k):

    # geometric progression
    q = pow(k, 1 / (nupoints - 1))
    b1 = (nu2 - nu1) * (1 - q) / (1 - pow(q, nupoints))

    cur = nu1
    nu[0] = nu1
    for i in range(nupoints):
        cur = cur + b1 * pow(q,i-1)
        nu[i] = cur
    return nu

# ------------------------------Rad_Intensity---------------------------------

def Integral(func, ro, a, b, points):
    """
        computes integral of a faction, interval: ro \in [a, b] with
        fun(ro, nu)
    """
    h = (b-a) / float(points)
    x = a
    sum = 0

    for i in range(points):
        x = x + h
        sum = sum + func(x, ro)

    l_l = func(a, ro)
    u_l = func(b, ro)

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

def Func_Mu(x, ro):
    return (pow(H(x,ro),(gam+2)/2))*N(x,ro)

def Func_Eps(x, ro):
    return (pow(H(x,ro),(gam+1)/2))*N(x,ro)

def Norm():
    ro = 0.0
    nu = 0.0
    return 1/((mu_0*pow(nu_norm,(-4-gam)/2))*Integral(Func_Mu,ro,-r,r,10000))

# ---------------------------Intensity-Spectrum---------------------------------

def Tau_Array(tau, x, a, b, ro):

    for i in range(points2):
        x[i]=b-i*(b-a)/float(points2)

    t = np.zeros(points2)
    for i in range(points2):
        t[i]=Integral(Func_Mu,ro,x[i],x[i-1],points1)

    tau[0] = 0.0
    for i in range(points2):
        tau[i] = tau[i - 1] + t[i]

    return tau, x

def Tau_Congestion(tau, x, a, b, ro, mu):

    cong = False

    iter = 100  # Maximum iterations for find precision.
    x_cong = np.zeros(iter +1) #  Array of <x> points, where congestion is begin

    tau, x = Tau_Array(tau, x, a, b, ro)

    print("{} {} {} {}".format(tau[0], tau[10], tau[100], tau[999]))

    for i in range(points2): # found a future congesion in exp(-tau))
        if (tau[i] * mu) > precision:
            cong = True
            break
        else:
            cong = False

    if cong:
        print("Cong")
        # If congestion exists, tring to find where it begins
        # Assume that it begin in the end (false)
        x_cong[0] = a
        for j in range(iter+1):
            # Chang the start point (less congestion)
            tau, x = Tau_Array(tau, x, x_cong[j], b, ro) # Chang the start point (less congestion)

            for i in range(1, points2+1): # Check where now congestion begin
                if ((tau[i]*mu)>precision):
                    x_cong[j + 1] = x[i]
                    print("tau*mu: i:{} {} > {} precision".format(i, tau[i]*mu, precision))
                    break               # Changing starting point and cycle starts again

            # Checking, is the new start point = previos. If so, we find the real start point pf congestion
            if (x_cong[j]==x_cong[j+1]): # in previos here was a <= 0.01 instead ==
                a = x_cong[j + 1]
                break
            # If the programm rich the end of cycle it meens that x_cong[j] still changing
            if (j==iter):
                print("Warning! Precision was not met! Make it low!\n")

        # Last iteration to count tau[i] and x[i] starting from real congestion point
        tau, x = Tau_Array(tau, x, a, b, ro)

    return a

def Int_Inten(Func_Eps, ro, nu):

    a1 = -1. * Limits(ro)
    b = Limits(ro)

    mu = Mu(nu)
    eps = Eps(nu)

    print("mu: {}".format(mu))
    print("eps: {}".format(eps)) # correct

    tau = np.zeros(points2+1)
    x = np.zeros(points2+1)

    # looking for the new start point <a> where there are no congestions
    print("a1:{}".format(a1))
    a = Tau_Congestion(tau, x, a1, b, ro, mu)
    print("a:{}".format(a))

    sum = 0.0
    for i in range(points2):
        sum += Func_Eps(x[i], ro) * (np.exp(-mu * tau[i]))

    l_l = Func_Eps(x[points2], ro) * np.exp(-mu * tau[points2])
    u_l = Func_Eps(x[0], ro) * np.exp(-mu * tau[0])

    sum += 0.5 * (l_l + u_l)
    sum *= (eps * (b - a) / points2)

    return sum

def Intensity(ro, nu):
    return Int_Inten(Func_Eps, ro, nu)

def Spectrum_I():
    I = np.zeros(nupoints)
    nu = np.zeros(nupoints)

    for i in range(nupoints):
        nu[i] = nu1 + i * (nu2 - nu1) / nupoints
        I[i] = Intensity(0.0, nu[i])

    Print(nu, I, nupoints, 1)

# -------------------------------Rad_Flux_Dens--------------------------------

def Func_Rad_Flux_Dens(x, nu):
    return x * Intensity(x, nu)

def Integ_Rad_Flux_Density(Func, a, b, nu):
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

def Spectrum():

    S = np.zeros(nupoints)
    nu = np.zeros(nupoints)

    nu = Frequency(nu, nupoints * 100)

    for i in range(nupoints):
        S[i] = Rad_Flux_Density(nu[i])

    Print(nu, S, nupoints, 2)

# ---------------------------------Theta---------------------------------------

def Dist_Array(I, ro, nu):

    for i in range(rpoints):

        ro[i] = i * r / rpoints
        I[i] = Intensity(ro[i], nu)

    I[rpoints] = 0.0

    return I, ro

def Distribution(nu):

    I = np.zeros(rpoints)
    ro = np.zeros(rpoints)

    I, ro = Dist_Array(I, ro, nu)

    pars = 3.08567758128 * pow(10, 18)

    for i in range(rpoints):
        ro[i] /= pars

    Print(ro, I, rpoints, 3)

def Theta(nu):

    I = np.zeros(rpoints)
    ro = np.zeros(rpoints)

    I, ro = Dist_Array(I, ro, nu)

    theta = 206264.8 * sqrt(M_PI) * ro[Middle(I, rpoints + 1)] / d # in seconds of angle

    return theta

def Theta_Array():

    theta = np.zeros(nupoints)
    nu = np.zeros(nupoints)

    nu = Frequency(nu, nupoints * 100)

    for i in range(nupoints):
        # nu[i]=nu1+i*(nu2-nu1)/nupoints
        theta[i] = Theta(nu[i])

    Print(nu, theta, nupoints, 4)



#if __name__ == "__main__":




print("mu_0: {} | eps_0: {} | R: {}".format(mu_0, eps_0, r))
N_0 = Norm()
print("N_0: {}".format(N_0))
print("Mu({}):  {}".format(nu_test, Mu(nu_test)))
print("Eps({}): {}".format(nu_test, Eps(nu_test)))

print("\n")

if task == " ": print("No task is given")
elif task == "I": Spectrum_I()
elif task == "S": Spectrum()
elif task == "i": print("I({}) : {}".format(nu_test, Intensity(0.0, nu_test)))
elif task == "s": print("S({}) : {}".format(nu_test, Rad_Flux_Density(nu_test)))
elif task == "d": Distribution(nu_test)
elif task == "t": print("Theta({}):  {}".format(nu_test, Theta(nu_test)))
elif task == "T": Theta_Array()
else: raise NameError("Incorrect task")