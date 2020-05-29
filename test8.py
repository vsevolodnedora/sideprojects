"""

    statistics

"""

from __future__ import division

import numpy as np
import scipy.optimize as opt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

import pandas

# --- describe the data --- 

dataset = pandas.read_csv("./data/dataset.csv")

print(dataset)
# for name, model in dataset.iterrows():
#     print("{} {} {} {}".format(name, model.dset_name, model.models, model.Lambda))

dataset = dataset[dataset["Lambda"] <= 1000]
dataset = dataset.sort_values(by="Lambda")
print(dataset["Mdisk3D"].describe(percentiles=[0.8, 0.9, 0.95]))

mdisk = np.array(dataset.Mdisk3D)

stat_names = ["nobs", "minmax", "mean", "variance", "skewness", "kurtosis"]
stat_values = stats.describe(mdisk)

for name, value in zip(stat_names, stat_values):
    print("{} : {}".format(name, value))

# --- prepare to fit data --- 

def get_error(MdiskPP):
    MdiskPP_min = 5e-4
    return 0.5 * MdiskPP + MdiskPP_min

# def get_error(MdiskPP):
#     return np.std(MdiskPP)

dataset["Mdisk3D_err"] = get_error(dataset["Mdisk3D"])

# x, y, yerr

std = dataset["Mdisk3D"].std()
n = len(np.array(dataset["Mdisk3D"]))
y_vals = np.array(dataset["Mdisk3D"], dtype=float)
y_errs = np.array(dataset["Mdisk3D_err"], dtype=float)
mean = np.mean(y_vals)

z = (y_vals - mean) / y_errs
chi2 = np.sum(z ** 2.)

chi2dof = chi2 / float(n - 1) # mean
sigma = np.sqrt(2. / float(n - 1))
nsig = (chi2dof - 1) / sigma

print("-----------------------------------------------")
print("\t num:     {:d} in the sample".format(n))
print("\t std:     {} ".format(std))
print("\t mean:    {} ".format(mean))
print("\t chi2:    {} ".format(chi2))
print("\t cho2dof: {}".format(chi2dof))
#print("\t nsig:    {} sigma's".format(nsig))


def get_chi2( y_vals, y_expets, y_errs):
    assert len(y_vals) == len(y_expets)
    z = (y_vals - y_expets) / y_errs
    chi2 = np.sum(z ** 2.)
    return chi2

#==================================================================

print(" --- fitting 1 parameter 1 ord polynomial regression --- ")

x = np.array(dataset["Lambda"], dtype=float).reshape((-1, 1))
y = np.array(dataset["Mdisk3D"], dtype=float)
yerr = np.array(dataset["Mdisk3D_err"], dtype=float)

transformer = PolynomialFeatures(degree=1, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
model = LinearRegression().fit(x_, y)
r_sq = model.score(x_, y)

y_pred = model.predict(x_)
chi2 = get_chi2(y, y_pred, yerr)
chi2dof = chi2 / (n - (1 + 1))


print('coefficient of determination R2: {}'.format(r_sq))
print('intercept                    b0: {}'.format(model.intercept_))
print('coefficients                 bi: {}'.format(model.coef_))
print('chi2:                            {}'.format(chi2))
print('chi2dof:                         {}'.format(chi2dof))

#==================================================================

print(" --- fitting 1 parameter 2 ord polynomial regression --- ")

x = np.array(dataset["Lambda"], dtype=float).reshape((-1, 1))
y = np.array(dataset["Mdisk3D"], dtype=float)
yerr = np.array(dataset["Mdisk3D_err"], dtype=float)

transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
model = LinearRegression().fit(x_, y)
r_sq = model.score(x_, y)

y_pred = model.predict(x_)
chi2 = get_chi2(y, y_pred, yerr)
chi2dof = chi2 / (n - (1 + 2))

print('coefficient of determination R2: {}'.format(r_sq))
print('intercept                    b0: {}'.format(model.intercept_))
print('coefficients                 bi: {}'.format(model.coef_))
print('chi2:                            {}'.format(chi2))
print('chi2dof:                         {}'.format(chi2dof))

fig = plt.figure()
ax = fig.add_subplot(111)
for dset_name in list(set(list(dataset["dset_name"]))):
    sel = dataset[dataset["dset_name"] == dset_name]
    sel_x = np.array(dataset["Lambda"], dtype=float).reshape((-1, 1))
    sel_y = np.array(dataset["Mdisk3D"], dtype=float)
    sel_yerr = np.array(dataset["Mdisk3D_err"], dtype=float)
    ax.errorbar(sel_x, sel_y, yerr=sel_yerr, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
ax.plot(x, y_pred, color="red", ls="-")
ax.set_xlabel("Lambda")
ax.set_ylabel("Mdisk3D")
ax.minorticks_on()
plt.legend(loc="upper left")
plt.show()
plt.close()

#==================================================================

print(" --- fitting 2 parameter 2 ord polynomial regression --- ")


x1 = np.array(dataset["Lambda"], dtype=float)#.reshape((-1, 1))
x2 = np.array(dataset["q"], dtype=float)
x = np.reshape(np.array([x1, x2]), (2, len(x1))).T # --> 2 columns of data

y = np.array(dataset["Mdisk3D"], dtype=float)
yerr = np.array(dataset["Mdisk3D_err"], dtype=float)

transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
model = LinearRegression().fit(x_, y)
r_sq = model.score(x_, y)

y_pred = model.predict(x_)
dataset["y_pred"] = y_pred
chi2 = get_chi2(y, y_pred, yerr)
chi2dof = chi2 / (n - (1 + 2))

print('coefficient of determination R2: {}'.format(r_sq))
print('intercept                    b0: {}'.format(model.intercept_))
print('coefficients                 bi: {}'.format(model.coef_))
print('chi2:                            {}'.format(chi2))
print('chi2dof:                         {}'.format(chi2dof))

#==================================================================

fig = plt.figure()
ax = fig.add_subplot(111)
for dset_name in list(set(list(dataset["dset_name"]))):
    sel = dataset[dataset["dset_name"] == dset_name]
    sel_x = np.array(dataset["Lambda"], dtype=float).reshape((-1, 1))
    sel_y = np.array(dataset["Mdisk3D"], dtype=float)
    sel_yerr = np.array(dataset["Mdisk3D_err"], dtype=float)
    ax.errorbar(y_pred, sel_y, yerr=sel_yerr, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
ax.plot([0, 0.3], [0, 0.3], color="red", ls="-", label="chi2dof:{}".format(chi2dof))
ax.set_xlabel("Mdisk3Dfit")
ax.set_ylabel("Mdisk3D")
ax.minorticks_on()
plt.legend(loc="upper left")
plt.show()
plt.close()


#==================================================================

def fitting_function(x, v):
    a, b, c, d = x
    return np.maximum(a + b * (np.tanh((v["Lambda"] - c) / d)), 1e-3)

def coefficients():
    a = -0.243087590223
    b = 0.436980750624
    c = 30.4790977667
    d = 332.568017486
    return np.array((a, b, c, d))

def residuals(x, dataset, v_n = "Mdisk3D"):
    xi = fitting_function(x, dataset)
    return (xi - dataset[v_n])

y_vals = np.array(dataset["Mdisk3D"])
y_errs = np.array(dataset["Mdisk3D_err"])
x0 = coefficients()
xi = fitting_function(x0, dataset) #
chi2 = get_chi2(y_vals, xi, y_errs)
print("chi2 original: {}".format(chi2))

res = opt.least_squares(residuals, x0, args=(dataset,))
xi = fitting_function(res.x, dataset)
chi2 = get_chi2(y_vals, xi, y_errs)
chi2dof = chi2 / (len(y_vals) - len(x0))

dataset["Mdisk3D_fit"] = xi

print("chi2    fit: {}".format(chi2))
print("chi2dof fit: {}".format(chi2dof))
print("Fit coefficients:")
for i in range(len(x0)):
    print("  coeff[{}] = {}".format(i, res.x[i]))




fig = plt.figure()
ax = fig.add_subplot(111)
for dset_name in list(set(list(dataset["dset_name"]))):
    sel = dataset[dataset["dset_name"] == dset_name]
    sel_x = np.array(dataset["Lambda"], dtype=float).reshape((-1, 1))
    sel_y = np.array(dataset["Mdisk3D"], dtype=float)
    y_pred = np.array(dataset["Mdisk3D_fit"], dtype=float)
    sel_yerr = np.array(dataset["Mdisk3D_err"], dtype=float)
    ax.errorbar(y_pred, sel_y, yerr=sel_yerr, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
ax.plot([0, 0.3], [0, 0.3], color="red", ls="-", label="chi2dof:{}".format(chi2dof))
ax.set_xlabel("Mdisk3Dfit")
ax.set_ylabel("Mdisk3D")
ax.minorticks_on()
plt.legend(loc="upper left")
plt.show()
plt.close()