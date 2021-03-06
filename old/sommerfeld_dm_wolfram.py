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

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
session = WolframLanguageSession()

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
        'nfw0-6': ['nfw', 0.6, lambda r: nfw_gamma(r, 0.6), 2.66623**-4 * (-1.3656)**2, 1.69103],
        'nfw0-7': ['nfw', 0.7, lambda r: nfw_gamma(r, 0.7), 2.03277**-4 * (-1.0006)**2, 1.5188],
        'nfw0-8': ['nfw', 0.8, lambda r: nfw_gamma(r, 0.8), 1.56803**-4 * (-0.7275)**2, 1.38725],
        'nfw0-9': ['nfw', 0.9, lambda r: nfw_gamma(r, 0.9), 1.21591**-4 * (-0.5230)**2, 1.28305],
        'nfw1-0': ['nfw', 1.0, lambda r: nfw_gamma(r, 1.0), 0.94281**-4 * (-0.3702)**2, 1.19814],
        'nfw1-1': ['nfw', 1.1, lambda r: nfw_gamma(r, 1.1), 0.72736**-4 * (-0.2569)**2, 1.1274],
        'nfw1-2': ['nfw', 1.2, lambda r: nfw_gamma(r, 1.2), 0.55536**-4 * (-0.1736)**2, 1.06738],
        'nfw1-3': ['nfw', 1.3, lambda r: nfw_gamma(r, 1.3), 0.41702**-4 * (-0.1132)**2, 1.01569],
        'nfw1-4': ['nfw', 1.4, lambda r: nfw_gamma(r, 1.4), 0.30540**-4 * (-0.0704)**2, 0.970611],
        'nfw1-27': ['nfw', 1.27, lambda r: nfw_gamma(r, 1.27), 0.455443**-4 * (-0.1292)**2, 1.03044],
        'nfw1-25': ['nfw', 1.25, lambda r: nfw_gamma(r, 1.25), 0.482469**-4 * (-0.1209)**2, 1.04061],
        'einasto0-13': ['einasto', 0.13, lambda r: einasto(r, 0.13), 0, 0],
        'einasto0-16': ['einasto', 0.16, lambda r: einasto(r, 0.16), 0, 0],
        'einasto0-17': ['einasto', 0.17, lambda r: einasto(r, 0.17), 0, 0],
        'einasto0-20': ['einasto', 0.20, lambda r: einasto(r, 0.20), 0, 0],
        'einasto0-24': ['einasto', 0.24, lambda r: einasto(r, 0.24), 0, 0],
        'burkert': ['burkert', 0, lambda r: burkert(r), 0, 0],
        'moore': ['moore', 0, lambda r: moore(r), 0, 0],
}


for fil in rho_dict.keys():
    try:
        os.mkdir(fil)
    except FileExistsError:
        pass
    os.chdir(fil)
    f = open('output.txt', 'w')
    # sys.stdout = f

    print('\n\n', fil, '\n')
    r_vals = np.logspace(-5, 2, num=lim)

    np.save('./rvals.npy', r_vals)

    label, gamma, rho, A, Ib = rho_dict[fil]
    k = (rho(r_vals.min()) * r_vals.min()**gamma)**(5/2) * ((3 - gamma) * (2 - gamma))**0.5

    rho_vals = rho(r_vals)

    def phi(r):
        return 


    print('computing phi vals')

    ######3
    #phi_vals = np.array([phi_x(rr) for rr in r_vals]) 
    #np.save('./phivals.npy', phi_vals)
    #####

    phi_vals = np.load('./phivals.npy')


    def analytic_phi(r):
        return r**(2 - gamma) / ((3 - gamma) * (2 - gamma))


    if PLOT is True:
        fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
        ax.plot(r_vals, phi_vals, label='our phi')
        if label == 'nfw':
            ax.plot(r_vals, analytic_phi(r_vals), label='analytic', ls='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('r')
        ax.set_ylabel('phi')
        ax.legend()

        if label == 'nfw':
            ax2.plot(r_vals, (phi_vals - analytic_phi(r_vals))/analytic_phi(r_vals))
        ax2.set_ylabel('percent residual')
        ax2.set_xlabel('r')
        ax2.set_ylim(bottom=-.1, top=.1)
        fig.savefig('./phi_comparison.pdf')

    # first derivative of rho(r)
    first_derv = np.gradient(rho_vals, phi_vals)

    # second derivative of rho(r)
    sec_derv = np.gradient(first_derv, phi_vals)

    # making the second derivative a function
    sec_derv_func = i1d(phi_vals, sec_derv, fill_value='extrapolate', bounds_error=False)

    print('computing dm phase space distribution')
    # plot with new phi values

    def f_analytic(E, gamma=gamma):
        return 3 / (16 * np.sqrt(2) * np.pi) * E **((gamma - 6) / (2 * (2 - gamma)))

    def f(E):
        return integrate.quad(lambda x: 1 / (np.sqrt(8) * np.pi**2) * sec_derv_func(x) / np.sqrt(x * 1.00000000001 - E), E, phi_vals.max(), points=phi_vals, limit=lim+1)[0]

    #####
    #fval = [f(Eval) for Eval in phi_vals] # print(fval[:10])
    #print(np.sum(np.isnan(fval)), 'nans in fvals')
    #fvals = np.nan_to_num(fval)

    #np.save('./fvals.npy', fvals)
    #####

    fvals = np.load('./fvals.npy')


    print('computed f vals')

    # fe, _, es = undim('./fe_GC_NFW_nounits.txt')

    # print(len(oldf['v']))

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

    recovered_rho_new = integrate.simps(4*np.sqrt(2)*np.pi*fvals[np.newaxis, :]*np.sqrt(np.abs(phi_vals[:, np.newaxis]-phi_vals[np.newaxis, :])*np.tri(len(phi_vals)).T), phi_vals, axis=-1)

    recovered_rho_old = integrate.simps(4*np.sqrt(2)*np.pi*oldfs[:, np.newaxis]*np.sqrt(np.abs(oldes[:, np.newaxis]-oldes[np.newaxis, :])*np.tri(len(oldes)).T), oldes, axis=0)

    recovered_rho_analytic = integrate.simps(4*np.sqrt(2)*np.pi*f_analytic(phi_vals)[np.newaxis, :]*np.sqrt(np.abs(phi_vals[:, np.newaxis]-phi_vals[np.newaxis, :]).T*np.tri(len(phi_vals)).T), phi_vals, axis=-1)

    if PLOT is True:
        fig, (ax, ax1) = plt.subplots(nrows=2)
        ax.plot(r_vals, rho_vals, label='rho')
        ax.plot(oldrs, recovered_rho_old, label='rho from kim fe')
        # ax.plot(r_vals, recovered_rho_analytic, label='rho analytic', ls='-.')
        ax.plot(r_vals, recovered_rho_new, label='rho from our fe', ls='--')
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
        # ax.plot(oldrs, (recovered_rho_old - rho(oldrs))/rho(oldrs), label='rho from kim fe')
        # ax.plot(r_vals, (recovered_rho_analytic - rho(r_vals))/rho(r_vals), label='rho analytic', ls='-.')
        ax.plot(r_vals, (recovered_rho_new - rho(r_vals))/rho(r_vals), label='rho from our fe', ls='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('r')
        ax.set_ylabel('delta rho / rho')
        ax.legend()
        fig.savefig('recovered_rho_residual.pdf')


        fig, (ax, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
        rescale = 1 / oldes.max() * (phi_vals.max()+1)
        yrescale = fvals[30] / oldfs[30]
        # yrescale = phi_vals[3:-3][-1] / oldes[0]
        # print('rescaled by', rescale)
        # print('y rescaled by', yrescale)
        # rescale = 1
        # yrescale = 1
        # ax.plot(oldes, oldfs, label='kims fe')
        ax.plot(phi_vals, fvals, label='new fe')
        ax.plot(phi_vals, f_analytic(phi_vals), label='analytic', ls='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('E')
        ax.set_ylabel('f(E)')
        ax.legend()

        ax2.plot(phi_vals, np.abs(np.gradient(fvals, phi_vals)), label='f(E) slope')
        # ax2.plot(phi_vals, np.abs(-5/2 * 3/16/np.sqrt(2) / np.pi * phi_vals**(-7/2)), label='Analytic solution')
        ax2.set_xscale('log')
        ax2.set_xlabel('E')
        ax2.set_ylabel('df/dE')
        ax2.set_yscale('log')
        ax2.legend()

        ax3.plot(phi_vals, (fvals - f_analytic(phi_vals))/f_analytic(phi_vals), label='analytic')
        ax3.set_xlabel('E')
        ax3.set_ylim(bottom=-.1, top=.1)
        ax3.set_ylabel('percent residual')

        fig.savefig('fecomparisons.pdf')


    print('changing variables')

    # vvals = np.linspace(0, np.sqrt(-2*phi_vals[0]), num=lim)
    vvals = np.logspace(-8, np.log10(np.sqrt(2)), num=lim+1)

    fvals = fvals
    phi_vals = phi_vals
    r_vals = r_vals

    ######
    #frv = np.zeros((len(r_vals), len(vvals)))
    #finterp = i1d(phi_vals, fvals, fill_value='extrapolate', bounds_error=False)
    ## i is the row index (r values)
    #for i in range(len(r_vals)):
    #    # j is the column index (v values)
    #    for j in range(len(vvals)):
    #        E = vvals[j]**2 / 2 + phi_vals[i]
    #        if E < 1:
    #            frv[i, j] = finterp(E)

    #print('nonzero frv', np.sum(frv.flatten() != 0))

    #f = i2d(r_vals, vvals, frv.T)
    #print('computing p2som')

    #integrand = frv[:, :, np.newaxis] * frv[:, np.newaxis, :] * vvals[np.newaxis, :, np.newaxis]**2 * vvals[np.newaxis, np.newaxis, :] * np.tri(len(vvals)).T[np.newaxis, :, :]
    #p2_som = integrate.simps(integrate.simps(integrand, vvals, axis=2), vvals, axis=1)
    #print(p2_som)
    #np.save('p2_sommerfeld.npy', p2_som) # save
    #####

    p2_som = np.load('p2_sommerfeld.npy') # save
    # p2_som[p2_som < 1e-6] = 0
    print(np.sum(np.isnan(p2_som)), 'nan vals in p2som')


    def p2_som_analytic(r):
        return A * k**2 * r ** (-1 + gamma * -3/2) / (32 * np.pi**2)

    print(np.sqrt(p2_som_analytic(1)))

    if PLOT is True:
        fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
        ax.plot(r_vals, p2_som, label='numerical')

        if label == 'nfw':
            ax.plot(r_vals, p2_som_analytic(r_vals), label='analytic', ls='--')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('r')
        ax.set_ylabel('p2 som')
        ax.legend()
        ax.set_xlim(right=1e2)

        if label == 'nfw':
            print(p2_som[0], p2_som_analytic(r_vals.min()), p2_som[0] / p2_som_analytic(r_vals.min()))
            ax2.plot(r_vals, (p2_som - p2_som_analytic(r_vals))/p2_som_analytic(r_vals), label='analytic')
            ax2.set_ylabel('percent residual')
            ax2.set_ylim(bottom=-.1, top=.1)
            ax2.set_xlabel('r')
        # ax.set_ylim(bottom=1e3)

        # plt.show()
        # p2isnan = np.isnan(p2_som)
        # fit = np.polyfit(np.log10(r_vals[3:500]), np.log10(p2_som[3:500]), deg=1)
        # print('fit is', fit)

        fig.savefig('./p2somplot.pdf')


    # p2_som = p2_som_analytic(r_vals)
    rho_r_som_interp = i1d(r_vals, p2_som, fill_value='extrapolate', kind='slinear', bounds_error=False)

    theta_vals = np.logspace(-3, 2, num=lim // 2)

    print('computing Jsom')
    def J_somm(theta):
        f = lambda r: (1-(theta/r)**2)**-0.5 * (32*np.pi**2) * rho_r_som_interp(r)# if theta/r > 0.2 else (1 + 0.5 *  (theta/r)**2) * (32*np.pi**2) * rho_r_som_interp(r)
        # return integrate.quad(f, theta, np.inf, limit=lim)[0]
        return integrate.quad(f, theta, theta_vals[-1], limit=lim, points=r_vals)[0]

    ###
    J_som = [J_somm(theta) for theta in theta_vals[:-1].astype(np.longdouble)]
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

        fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
        ax.plot(theta_vals, J_som, label='new code')
        ax.plot(radius, hsomn, label='old cod')

        if label == 'nfw':
            b = -1 + gamma * -3 / 2
            jsom_analytic = A * k**2 * Ib * theta_vals**(b + 1)
            ax.plot(theta_vals, jsom_analytic, label='mathematica', ls='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('theta')
        ax.set_ylabel('J som')
        ax.set_xlim(left=1e-3, right=1)
        ax.set_ylim(bottom=10, top=1e6)
        ax.legend() 

        if label == 'nfw':
            ax2.plot(theta_vals, (J_som - jsom_analytic) / jsom_analytic, label='analytic')
            # ax2.plot(theta_vals, 1 / J_som * jsom_analytic)
            ax2.set_xlabel('theta')
            ax2.set_ylabel('percent residual')
            ax2.set_ylim(bottom=-.1, top=.2)
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
