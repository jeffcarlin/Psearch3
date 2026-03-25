import matplotlib.pyplot as plt
import numpy as np
import os
import time as tm
# import sys
import IPython
from astropy.table import Table

# from Psearch3 import psearch_py
import psearch_py

def plot_multiband_phased_lc(expmid, mag, magerr, bands, period, magmin=17, magmax=23,
                             witherrors=True, save_as=None):

    params = {
       'axes.labelsize': 16,
       'font.size': 16,
       'legend.fontsize': 14,
    #   'xtick.labelsize': 16,
       'xtick.major.width': 3,
       'xtick.minor.width': 2,
       'xtick.major.size': 8,
       'xtick.minor.size': 4,
       'xtick.direction': 'in',
       'xtick.top': True,
       'lines.linewidth':2,
       'axes.linewidth':2,
       'axes.labelweight':3,
       'axes.titleweight':3,
       'ytick.major.width':3,
       'ytick.minor.width':2,
       'ytick.major.size': 8,
       'ytick.minor.size': 4,
       'ytick.direction': 'in',
       'ytick.right': True,
    #   'ytick.labelsize': 20,
       'text.usetex': False,
       'text.latex.preamble': r'\boldmath',
       'figure.figsize': [7, 7],
       'figure.facecolor': 'White'
       }

    plt.rcParams.update(params)

    plot_filter_colors_white_background = {"u": "#61A2B3", "g":"#31DE1F", "r": "#B52626",
                                           "i": "#1600EA", "z": "#BA52FF", "y": "#370201"}
    # plot_filter_colors_white_background = {"u": "#0c71ff", "g": "#49be61", "r": "#c61c00",
    #                                        "i": "#ffc200", "z": "#f341a2", "y": "#5d0000"}
    plot_symbols = {'u': 'o', 'g': '^', 'r': 'v', 'i': 's', 'z': '*', 'y': 'p'}
    plot_line_styles = {
        "u": "--",
        "g": (0, (3, 1, 1, 1)),
        "r": "-.",
        "i": "-",
        "z": (0, (3, 1, 1, 1, 1, 1)),
        "y": ":",
    }

    # t0 = np.min(expmid)
    zero_phase_band = 'r'
    t0 = expmid[bands == zero_phase_band][np.argmin(mag[bands == zero_phase_band])]
#     print(t0, mag[np.argmin(mag[bands == 'r'])])

    mjd_norm = (expmid - t0) / period
    phase = np.mod(mjd_norm, 1.0)

    fig = plt.figure(figsize=(11, 5))

    if witherrors == False:
        med_mag_errors = {}

    i = 0

    for band in np.unique(bands):
        inband = (bands == band)
        if witherrors:
            plt.errorbar(phase[inband], mag[inband], yerr=magerr[inband],
                         marker=plot_symbols[band], linestyle='none',
                         color=plot_filter_colors_white_background[band],
                         fillstyle='none', label=band, ms=7)
            plt.errorbar(phase[inband]+1.0, mag[inband], yerr=magerr[inband],
                         marker=plot_symbols[band], linestyle='none',
                         color=plot_filter_colors_white_background[band],
                         fillstyle='none', label='__none__', ms=7)
        else:
            med_mag_errors[band] = np.nanmedian(magerr[inband])
            plt.plot(phase[inband], mag[inband],
                     marker=plot_symbols[band], linestyle='none',
                     color=plot_filter_colors_white_background[band],
                     fillstyle='none', label=band, ms=7)
            plt.plot(phase[inband]+1.0, mag[inband],
                     marker=plot_symbols[band], linestyle='none',
                     color=plot_filter_colors_white_background[band],
                     fillstyle='none', label='__none__', ms=7)
            xerrval = 0.15
            yerrval0 = magmax-(len(np.unique(bands))*0.05)
            yerrval = yerrval0+(i*0.05)
            plt.errorbar(xerrval, yerrval, yerr=med_mag_errors[band],
                         color=plot_filter_colors_white_background[band],
                         capsize=3)

        i += 1

    # plt.plot([xerrval-0.12, xerrval-0.12, xerrval+0.12, xerrval+0.12, xerrval-0.12],
    #          [yerrval0+(i*0.05)+0.05, yerrval0-0.075,  yerrval0-0.075, yerrval0+(i*0.05), yerrval0+(i*0.05)],
    #          color='black', linewidth=1)
    plt.text(xerrval-0.1, yerrval0-0.025, 'median errors', size='xx-small')

    plt.gca().invert_yaxis()
    plt.legend(loc=(0.8, 0.65))
    plt.xlabel('phase')
    plt.ylabel('magnitude (extinction corrected)')
    plt.minorticks_on()
    plt.ylim(magmax, magmin)
    plt.xlim(-0.02, 2.02)
    if save_as:
        plt.savefig(save_as)
    plt.show()


def plot_multiband_unphased_lc(expmid, mag, magerr, bands, period, magmin=None, magmax=None):

    plot_filter_colors_white_background = {"u": "#61A2B3", "g":"#31DE1F", "r": "#B52626",
                                           "i": "#1600EA", "z": "#BA52FF", "y": "#370201"}
#     plot_filter_colors_white_background = {"u": "#0c71ff", "g": "#49be61", "r": "#c61c00",
#                                            "i": "#ffc200", "z": "#f341a2", "y": "#5d0000"}
    plot_symbols = {'u': 'o', 'g': '^', 'r': 'v', 'i': 's', 'z': '*', 'y': 'p'}
    plot_line_styles = {
        "u": "--",
        "g": (0, (3, 1, 1, 1)),
        "r": "-.",
        "i": "-",
        "z": (0, (3, 1, 1, 1, 1, 1)),
        "y": ":",
    }

    fig = plt.figure(figsize=(9, 6))

    for band in np.unique(bands):
        inband = (bands == band)
        plt.errorbar(expmid[inband], mag[inband], yerr=magerr[inband],
                     marker=plot_symbols[band], linestyle='none',
                     color=plot_filter_colors_white_background[band],
                     fillstyle='none', label=band)

    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel('time (MJD)')
    plt.ylabel('mag')
    plt.minorticks_on()
    if magmin:
        plt.ylim(magmax, magmin)
    plt.show()


def get_amplitude(mags):
    median_mag = np.nanmedian(mags)
    # Remove anything that is >1 mag from the median
    # (this is hacky, but should work most of the time)
    good = (np.abs(mags-median_mag) < 1.0)
    minmag = np.min(mags[good])
    maxmag = np.max(mags[good])

    return maxmag-minmag, np.mean(mags[good])


def gmag_comcam_to_ps1(gmag, gi):
    # Relation from Douglas Tucker, sent to me on June 3, 2025 via Slack
    g_ps1 = gmag - 0.017 - 0.036*gi
    return g_ps1


def get_dist(gmag, period):
    # Following Vivas+2019 for DCs:
    # Mg = −2.01 log P + 0.29
    absmag_g = -2.01*np.log10(period) + 0.29
    dmod = gmag-absmag_g
    dist = 10.0**((dmod+5)/5.0)
    return dist


def run_psearch(inp_file, pmin=0.025, dphi=0.02,
                filtnames=['u', 'g', 'r', 'i', 'z'],
                plot_lc=True, lcfilename='tmp.png',
                magmax=19.13, magmin=18.28,
                plotwitherrors=False,
                results_tab='tmp.csv'):
    tab = Table.read(inp_file)

    ok = (tab['magerr'] > 0.0 ) & (tab['magerr'] <= 0.2)
    hjd = tab[ok]['expmid']
    mag = tab[ok]['mag']
    magerr = tab[ok]['magerr']
    filts_names = tab[ok]['band']
    print(len(mag),' good data points found out of', len(tab))

    filtmap = {'u':0.0, 'g':1.0, 'r':2.0, 'i':3.0, 'z':4.0, 'y':5.0}

    # Apply extinction correction
    ext_coeffs = {'u': np.float64(4.757217815396922),
                  'g': np.float64(3.6605664439892625),
                  'r': np.float64(2.70136780871597),
                  'i': np.float64(2.0536599130965882),
                  'z': np.float64(1.5900964472616756),
                  'y': np.float64(1.3077049588254708)}
    ebv = 0.039242 # From DP1 Object table

    for band in ['u', 'g', 'r', 'i', 'z']:
        mag[(filts_names == band)] = mag[(filts_names == band)] - ebv*ext_coeffs[band]
    
    filts = np.zeros(len(filts_names))
    filts[(filts_names == 'u')] = filtmap['u']
    filts[(filts_names == 'g')] = filtmap['g']
    filts[(filts_names == 'r')] = filtmap['r']
    filts[(filts_names == 'i')] = filtmap['i']
    filts[(filts_names == 'z')] = filtmap['z']

    # And away we go!
    time00 = tm.time()
    periods, psi_m, thresh_m = \
        psearch_py.psearch_py(hjd, mag, magerr, filts, filtnames, pmin, dphi)
    time01 = tm.time()
    print('\n\n%8.3f seconds [walltime for psearch_py]\n' % (time01-time00))

    # Period of the strongest peak of the combined Psi distribution
    idx = np.argmax(psi_m.sum(0))
    p_peak = periods[idx]
    print('\nPeriod: %9.8f' % p_peak)

    # Show the top 10 peaks of the combined Psi distribution
    psearch_py.table_psi_kjm_py( xx=periods, yy=psi_m.sum(0), ee=thresh_m.sum(0), n=10 )
    psearch_py.write_table(xx=periods, yy=psi_m.sum(0), ee=thresh_m.sum(0), n=10, filename=results_tab)

    if plot_lc:
        keep = (filts_names == 'g') | (filts_names == 'r') | (filts_names == 'i') | (filts_names == 'z')
        # keep = (filts_names == 'u') | (filts_names == 'g') | (filts_names == 'r') | (filts_names == 'i') | (filts_names == 'z')

        if magmin is None:
            magmin = np.nanmin(mag[keep])-0.15
        if magmax is None:
            magmax = np.nanmax(mag[keep])+0.15

        plot_multiband_phased_lc(hjd[keep], mag[keep], magerr[keep], filts_names[keep], p_peak,
                                 magmax=magmax, magmin=magmin, witherrors=plotwitherrors,
                                 save_as=lcfilename)

    meanmags = []
    amplitudes = []
    bands = []
    periods = []
    for band in filtnames:
        keep = (filts_names == band)
        amplitude, meanmag = get_amplitude(mag[keep])
        # print(f"Amplitude, mean mag in {band} band: {amplitude:.3}, {meanmag:6.4}")
        bands.append(band)
        periods.append(p_peak)
        meanmags.append(meanmag)
        amplitudes.append(amplitude)

    tab = Table([bands, meanmags, amplitudes, periods], names=['band', 'meanmag', 'amp', 'per'])
    
    return tab


if __name__ == "__main__":
    # Input CSV file of forced-source measurements:
    # ifile = '/Users/jcarlin/Dropbox/comcam_variables/2430400084454684323.csv' # "real variable" from ComCam
    ifile = '/Users/jcarlin/Dropbox/comcam_variables/sx_phe_SV95-25/cands_lcdata/614435753623027782.csv'

    # Run the search
    results = run_psearch(ifile, pmin=0.025, dphi=0.02, filtnames=['u', 'g', 'r', 'i', 'z'],
                          plot_lc=True, lcfilename='sxphe_comcam_phased_lightcurve2.png',
                          magmax=18.98, magmin=18.13, plotwitherrors=False)

    gmean = results[(results['band'] == 'g')]['meanmag']
    imean = results[(results['band'] == 'i')]['meanmag']
    gi = gmean-imean
    
    print(results)

    gps1 = gmag_comcam_to_ps1(gmean, gi)
    gperiod = results[(results['band'] == 'g')]['per']
    print(f"Distance: {get_dist(gps1, gperiod)} pc")

