import numpy as np
from scipy.interpolate import LinearNDInterpolator
import h5py
from astropy.table import Table
import pyneb as pn

line_wavelengths = {
    'O2-3727': 3727.092,
    'O2-3729': 3729.875,
    'Ne3-3869': 3870.16,
    'He': 3970.072,
    'S2-4068': 4073.6245, #blend
    'Hd': 4101.734,
    'Hg': 4341.684,
    'O3-4363': 4364.436,
    'He1-4471': 4472.767, #blend
    'Hb': 4862.683,
    'O3-4959': 4960.295,
    'O3-5007': 5008.240,
    'He1-5016': 5017.0765,
    'Ar3-5192': 5193.27,
    'N1-5198': 5200.527, #blend
    'N2-5755': 5756.240,
    'He1-5875': 5877.305, #blend
    'O1-6300': 6302.046,
    'S3-6313': 6313.8,
    'N2-6548': 6549.85,
    'Ha': 6564.61,
    'N2-6584': 6585.28,
    'He1-6678': 6679.99656,
    'S2-6717': 6718.29,
    'S2-6731': 6732.67,
    'Ar3-7136': 7137.8,
    'O2-7318': 7326.845, #blend
    'Ar3-7751': 7753.2,
    'S3-9068': 9071.1,
    'S3-9532': 9533.2,
    }
#line centres in vacuum http://www.pa.uky.edu/~peter/atomic/ v2.04



def load_data(kappa):
    """Loads a data table for a specified kappa

    Parameters
    ----------
    kappa : 10, 20, 50, 'inf'
    
    Returns
    -------
    Z : M array of floats
        metallicity [Z_solar]
    logq : M array of floats
        ionization parameter [cm/s]
    lines : N list of strings
        emission line names
    flux : MxN array of floats
        line fluxes [normalized to Hb=1]

    """

    data_blue = Table.read('raw/Dopita13/table4.dat', format='cds',
                           readme="raw/Dopita13/ReadMe")
    data_red = Table.read('raw/Dopita13/table5.dat', format='cds',
                          readme="raw/Dopita13/ReadMe")

    #check tables correspond 1:1
    x_blue = data_blue[['Z', 'kappa', 'f_kappa', 'logq']]
    x_red = data_blue[['Z', 'kappa', 'f_kappa', 'logq']]
    assert np.all(x_blue == x_red), "Red and Blue table rows do not match"

    if kappa == 'inf':
        mask_kappa = data_blue['f_kappa'] == 'i'
    elif kappa in [10, 20, 50]:
        mask_kappa = data_blue['kappa'] == kappa
    else:
        raise Exception("Invalid value for kappa")

    lines_blue = data_blue.colnames[4:]
    lines_red = data_red.colnames[4:]
    lines = np.array(lines_blue + lines_red)

    Z = data_blue[mask_kappa]['Z'].data
    logq = data_blue[mask_kappa]['logq'].data

    flux_blue = data_blue[mask_kappa][lines_blue].as_array().data
    flux_blue = flux_blue.view((float, len(flux_blue.dtype.names)))
    flux_red = data_red[mask_kappa][lines_red].as_array().data
    flux_red = flux_red.view((float, len(flux_red.dtype.names)))

    flux = np.column_stack([flux_blue, flux_red])

    #sort lines (should already be)
    wave = [line_wavelengths[l] for l in lines]
    sort = np.argsort(wave)
    lines = lines[sort]
    flux = flux[:,sort]

    return Z, logq, lines, flux


def interp_caseb(flux_Hb, flux_Hg, line):

    temp = np.logspace(np.log10(500.), np.log10(30000.), 10000)[1:-1]
    den = 10.

    H1 = pn.RecAtom('H', 1)
    model_Hb = H1.getEmissivity(temp, den, 4, 2)
    model_Hg = H1.getEmissivity(temp, den, 5, 2)

    if line == 'Hd':
        model_line = H1.getEmissivity(temp, den, 6, 2)
    elif line == 'He':
        model_line = H1.getEmissivity(temp, den, 7, 2)
    else:
        raise Exception

    flux_HgHb = flux_Hg / flux_Hb
    model_HgHb = model_Hg / model_Hb
    model_lineHb = model_line / model_Hb
    flux_lineHb = np.interp(flux_HgHb, model_HgHb, model_lineHb)
    flux_line = flux_lineHb * flux_Hb

    return flux_line


def extend_hydrogen(lines, flux):

    for balmer_line in ['Hd', 'He']:
        flux_Hb = flux[:,lines=='Hb']
        flux_Hg = flux[:,lines=='Hg']
        
        flux_line = interp_caseb(flux_Hb, flux_Hg, balmer_line)

        wave = [line_wavelengths[l] for l in lines]
        balmer_wave = line_wavelengths[balmer_line]
        idx = np.searchsorted(wave, balmer_wave)

        lines = np.concatenate([lines[:idx], [balmer_line], lines[idx:]])
        flux = np.column_stack([flux[:,:idx], flux_line, flux[:,idx:]])

    return lines, flux



def rescale_flux(lines, flux):

    SFR = 1. # M_sun / yr
    Ha_expected = SFR / 7.9e-42 # erg/s 1998ARA&A..36..189K

    Ha_obs = flux[:,lines=='Ha']
    norm = Ha_obs / Ha_expected

    flux /= norm
    

def create_grid(logZ, logU, flux):

    x_logZ = np.unique(logZ)
    x_logU = np.unique(logU)

    points = np.column_stack([logZ, logU])
    intp = LinearNDInterpolator(points, flux)


    X_logZ = np.repeat(x_logZ, len(x_logU))
    X_logU = np.tile(x_logU, len(x_logZ))
    X = np.column_stack([X_logZ, X_logU])
    grid = intp(X).reshape([x_logZ.size, x_logU.size, -1])
    
    return x_logZ, x_logU, grid


def create_file(filename, logZ, logU, lines, flux):

    wave = [line_wavelengths[l] for l in lines]
    
    fh = h5py.File(filename, 'w')

    fh['logZ'] = logZ
    fh['logU'] = logU
    fh['line_name'] = lines
    fh['line_wave'] = wave
    fh['flux'] = flux

    #create dim scales
    
    fh['flux'].dims[0].label = 'logZ'
    fh['flux'].dims[1].label = 'logU'
    fh['flux'].dims[2].label = 'line'

    fh['flux'].dims.create_scale(fh['logZ'], 'logZ')
    fh['flux'].dims.create_scale(fh['logU'], 'logU')
    fh['flux'].dims.create_scale(fh['line_name'], 'name')
    fh['flux'].dims.create_scale(fh['line_wave'], 'wave')

    fh['flux'].dims[0].attach_scale(fh['logZ'])
    fh['flux'].dims[1].attach_scale(fh['logU'])
    fh['flux'].dims[2].attach_scale(fh['line_name'])
    fh['flux'].dims[2].attach_scale(fh['line_wave'])

    return fh

if __name__ == '__main__':
    
    for kappa in [10, 20, 50, 'inf']:
        Z, logq, lines, flux = load_data(kappa)
        #add Hd and He
        lines, flux = extend_hydrogen(lines, flux)
        

        logZ = np.log10(Z)

        cspeed = 29979245800. #speed of light in cm
        logU = logq - np.log10(cspeed)
        
        rescale_flux(lines, flux)
        
        logZ, logU, flux = create_grid(logZ, logU, flux)


        filename = 'grid_Dopita13_kappa='+str(kappa)+'.h5'

        fh = create_file(filename, logZ, logU, lines, flux)
        fh['OH_solar'] = 8.69 # 2013ApJS..208...10D

        fh['logZ'].attrs['units'] = 'Z_solar'
        fh['line_wave'].attrs['units'] = 'A'
        fh['OH_solar'].attrs['units'] = '12 + log10(O/H)'
        fh['flux'].attrs['units'] = 'erg/s'

        fh.flush()
        fh.close()
