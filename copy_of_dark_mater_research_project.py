#!/usr/env/python3
# -*- coding: utf-8 -*-
import sys, os
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import interp1d as i1d
from scipy.interpolate import interp2d as i2d
from scipy import integrate

orig_stdout = sys.stdout

lim = 1100
PLOT = True

def nfw_gamma(r, gamma):
    return r**-gamma * (1 + r)**-(3 - gamma)


def einasto(r, alpha):
    return np.exp(-(2 / alpha) * (r**alpha - 1))

def burkert(r):
    return (1 + r)**-1 * (1 + r**2)**-2

def moore(r):
    return (r**1.4 * (1 + r)**1.4)**-1

rho_dict = {
        'nfw0-6': lambda r: nfw_gamma(r, 0.6),
        'nfw0-7': lambda r: nfw_gamma(r, 0.7),
        'nfw0-8': lambda r: nfw_gamma(r, 0.8),
        'nfw0-9': lambda r: nfw_gamma(r, 0.9),
        'nfw1-0': lambda r: nfw_gamma(r, 1.0),
        'nfw1-1': lambda r: nfw_gamma(r, 1.1),
        'nfw1-2': lambda r: nfw_gamma(r, 1.2),
        'nfw1-27': lambda r: nfw_gamma(r, 1.27),
        'nfw1-3': lambda r: nfw_gamma(r, 1.3),
        'nfw1-4': lambda r: nfw_gamma(r, 1.4),
        'einasto0-13': lambda r: einasto(r, 0.13),
        'einasto0-16': lambda r: einasto(r, 0.16),
        'einasto0-17': lambda r: einasto(r, 0.17),
        'einasto0-20': lambda r: einasto(r, 0.20),
        'einasto0-24': lambda r: einasto(r, 0.24),
        'burkert': lambda r: burkert(r),
        'moore': lambda r: moore(r),
}


for fil in rho_dict.keys():
    os.mkdir(fil)
    os.chdir(fil)
    f = open('output.txt', 'w')
    sys.stdout = f

    # Setting a range of what our r̃ values will be
    r_vals = np.logspace(-3, 2, num=lim)


    # def rho(r):
    #     return 1 / (r * (1 + r)**2)
    rho = rho_dict[fil]

    rho_vals = rho(r_vals)


    def phi_y(x):
        f = lambda y: y**2 * rho(y)
        return integrate.quad(f, 0, x, points=r_vals[::5], limit=lim // 3)[0]


    def phi_x(r):
        f = lambda x: 1 / x**2 * phi_y(x)
        return -integrate.quad(f, r, 1000, points=r_vals[::5], limit=lim // 3)[0]


    print('computing phi vals')

    ###3
    phi_vals = [phi_x(rr) for rr in r_vals]
    np.save('./phivals.npy', phi_vals)
    ###

    phi_vals = np.load('./phivals.npy')
    # first derivative of rho(r)
    first_derv = np.gradient(rho_vals, phi_vals)

    # second derivative of rho(r)
    sec_derv = np.gradient(first_derv, phi_vals)
    sec_derv[sec_derv < 0] = 0

    # making the second derivative a function
    sec_derv_func = i1d(phi_vals, sec_derv, fill_value=0, bounds_error=False)
    integrand = i1d(phi_vals, sec_derv, fill_value=0, bounds_error=False)

    print('computing dm phase space distribution')
    # plot with new phi values


    def f(E):
        return integrate.quad(lambda x: 1 / (np.sqrt(8) * np.pi**2) * sec_derv_func(x) / np.sqrt(x * 0.999999999999 - E), E, 0, points=phi_vals[3:-3:3], limit=lim)[0]

    ####
    fval = [f(Eval) for Eval in phi_vals[3:-3]] # print(fval[:10])
    print(np.sum(np.isnan(fval)), 'nans in fvals')
    fvals = np.nan_to_num(fval)

    np.save('./fvals.npy', fvals)
    ####

    fvals = np.load('./fvals.npy')


    print('computed f vals')

    # fe, _, es = undim('./fe_GC_NFW_nounits.txt')

    # print(len(oldf['v']))
    num = 1000

    # oldes = oldf['v'][::num]**2/2+np.array([phi_x(rr) for rr in oldf['r'][::num]])
    # np.savez('./oldFEs.npz', oldes=oldes, oldf=oldf['fe'][::num])

    # of = np.load('./oldFEs.npz', allow_pickle=True)
    of = np.load('../fe_nounits.npz', allow_pickle=True)
    # oldes = of['oldes']
    oldes = of['energy']
    # oldfs = of['oldf']
    oldfs = of['fe']
    oldrs = of['rs']
    of.close()

    # comparing fes
    # print('max of fe', oldes[oldfs.argmax()], phi_vals[3:-3][fvals.argmax()])
    # print('shift of fes', oldes
    # print('int of fe', np.trapz((4*np.sqrt(2)*np.pi*fvals*np.sqrt(phi_vals[3:-3]-phi_vals[15]))[12:], phi_vals[3:-3][12:]), rho(r_vals[15]))

    phi_vals = np.flip(np.abs(phi_vals))
    fvals = np.flip(np.abs(fvals))
    r_vals = np.flip(r_vals)

    recovered_rho_new = integrate.simps(4*np.sqrt(2)*np.pi*fvals[:, np.newaxis]*np.sqrt((phi_vals[3:-3][:, np.newaxis]-phi_vals[3:-3][np.newaxis, :]).T*np.tri(len(phi_vals[3:-3])).T), phi_vals[3:-3], axis=0)

    # print((phi_vals[3:-3][:, np.newaxis]-phi_vals[3:-3][np.newaxis, :]).T*np.tri(len(phi_vals[3:-3])).T)
    # print(recovered_rho_new)
    # print((oldes[:, np.newaxis]-oldes[np.newaxis, :]).T*np.tri(len(oldes)).T)
    # print(phi_vals[0], phi_vals[-1], oldes[0], oldes[-1])
    recovered_rho_old = integrate.simps(4*np.sqrt(2)*np.pi*oldfs[:, np.newaxis]*np.sqrt((oldes[:, np.newaxis]-oldes[np.newaxis, :]).T*np.tri(len(oldes)).T), oldes, axis=0)

    if PLOT is True:
        fig, (ax, ax1) = plt.subplots(nrows=2)
        ax.plot(r_vals, np.flip(rho_vals), label='rho')
        ax.plot(oldrs, recovered_rho_old, label='rho from kim fe')
        ax.plot(r_vals[3:-3], recovered_rho_new, label='rho from our fe')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('r')
        ax.set_ylim(bottom=1e-2)
        ax.set_ylabel('rho')
        ax.legend()

        ax1.plot(oldrs, recovered_rho_old/rho(oldrs) - 1)
        ax1.set_xscale('log')
        fig.savefig('recovered_rho.pdf')

        fig, ax = plt.subplots()
        ax.plot(oldrs, (recovered_rho_old - rho(oldrs))/rho(oldrs), label='rho from kim fe')
        ax.plot(r_vals[3:-3], (recovered_rho_new - rho(r_vals[3:-3]))/rho(r_vals[3:-3]), label='rho from our fe')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('r')
        ax.set_ylabel('delta rho / rho')
        ax.legend()
        fig.savefig('recovered_rho_residual.pdf')


        fig, ax = plt.subplots()
        rescale = 1 / oldes.max() * phi_vals[3:-3].max()
        yrescale = fvals[30] / oldfs[30]
        # yrescale = phi_vals[3:-3][-1] / oldes[0]
        # print('rescaled by', rescale)
        # print('y rescaled by', yrescale)
        # rescale = 1
        # yrescale = 1
        ax.plot(oldes * rescale, oldfs * yrescale, label='old fe')
        ax.plot(np.abs(phi_vals[3:-3]), fvals, label='new fe')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('E')
        ax.set_ylabel('f(E)')
        ax.legend()
        fig.savefig('fecomparisons.pdf')

    phi_vals = np.flip(-phi_vals)
    fvals = np.flip(np.abs(fvals))
    r_vals = np.flip(r_vals)


    print('changing variables')

    # vvals = np.linspace(0, np.sqrt(-2*phi_vals[0]), num=lim)
    vvals = np.logspace(-6, np.log10(np.sqrt(-2*phi_vals[0])), num=lim)

    #####
    frv = np.zeros((len(r_vals), len(vvals)))
    finterp = i1d(phi_vals[3:-3], fvals, fill_value=0, bounds_error=False)
    # i is the row index (r values)
    for i in range(len(r_vals)):
        # j is the column index (v values)
        for j in range(len(vvals)):
            E = vvals[j]**2 / 2 + phi_vals[i]
            if E < 0:
                frv[i, j] = finterp(E)

        # print(frv)

    f = i2d(r_vals, vvals, frv.T)
    print('computing p2som')

    integrand = frv[:, :, np.newaxis] * frv[:, np.newaxis, :] * vvals[np.newaxis, :, np.newaxis]**2 * vvals[np.newaxis, np.newaxis, :] * np.tri(len(vvals))[np.newaxis, :, :]
    p2_som = integrate.simps(integrate.simps(integrand, vvals, axis=2), vvals, axis=1)
    np.save('p2_sommerfeld.npy', p2_som) # save
    ####

    p2_som = np.load('p2_sommerfeld.npy') # save
    print(np.sum(np.isnan(p2_som)), 'nan vals in p2som')

    if PLOT is True:
        plt.cla()
        plt.plot(r_vals, p2_som)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('r')
        plt.ylabel('p2 som')
        plt.xlim(left=1e-3, right=1e2)
        # plt.ylim(bottom=10)

        # plt.show()
        # p2isnan = np.isnan(p2_som)
        # fit = np.polyfit(np.log10(r_vals[3:500]), np.log10(p2_som[3:500]), deg=1)
        # print('fit is', fit)

        plt.savefig('./p2somplot.pdf')


    rho_r_som_interp = i1d(r_vals, p2_som, fill_value=0)

    theta_vals = np.logspace(-3, 2, num=lim)

    print('computing Jsom')
    def J_somm(theta):
        f = lambda r: 1 / np.sqrt(1-(theta/(r + 1e-40))**2) * (32*np.pi**2) * rho_r_som_interp(r + 1e-40)
        return integrate.quad(f, theta, 100, limit=lim, points=r_vals)[0]

    ####
    J_som = [J_somm(theta) for theta in theta_vals[:-1]]
    print(np.sum(np.isnan(J_som)), 'nans in Jsom')


    np.save('J_som_vals.npy', J_som) # save
    np.save('theta_vals.npy', theta_vals[:-1])
    #####
    theta_vals = np.load('theta_vals.npy')
    J_som = np.load('J_som_vals.npy') # save

    if PLOT is True:
        with np.load("../df_nfw_h_values.txt") as infile:
            hsomn = infile['hsom']
            radius = infile['radius']

        fig, ax = plt.subplots()
        ax.plot(theta_vals, J_som, label='new code')
        ax.plot(radius, hsomn, label='old cod')

        ax.plot(theta_vals, 10**-2.45*1.18*theta_vals**-1.54*32*np.pi**2, label='mathematica')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('theta')
        ax.set_ylabel('J som')
        ax.set_xlim(left=1e-3, right=1)
        ax.set_ylim(bottom=1)
        ax.legend() 
        fig.savefig('./jsom.pdf')

    # print(J_som)
    equation_8_som = np.trapz(np.nan_to_num(J_som) * theta_vals, theta_vals)
    print('total j factor', equation_8_som)   #eq 8 1.02
    equation_10_som = np.trapz(np.nan_to_num(J_som) * theta_vals**2, theta_vals) / equation_8_som
    print('angular spread', equation_10_som)  #eq 10 match the paper 0.27

    # equation_8_som = np.trapz(np.nan_to_num(hsomn) * radius, radius)
    # print('total j factor', equation_8_som)   #eq 8 1.02
    # equation_10_som = np.trapz(np.nan_to_num(hsomn) * radius**2, radius) / equation_8_som
    # print('angular spread', equation_10_som)  #eq 10 match the paper 0.27
    os.chdir('..')

sys.stdout = orig_stdout
