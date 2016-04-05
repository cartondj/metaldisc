import numpy as np

class BaseDustModel(object):
    def __init__(self):
        """Base class of dust models
        
        """
        pass


    def __call__(self, galaxy, params):
        raise NotImplementedError("Subclasses should provide __call__ method")



class ConstantDust(BaseDustModel):

    def __init__(self):
        """Model with constant dust attenuation throughout the galaxy
        
        """
        pass


    def __call__(self, galaxy, params):
        """Generates dust attenuation tauV for each bin, given model parameters
        
        Produces an axisymmetric linear profile.

        Parameters
        ----------
        galaxy : instance of BaseGalaxy (or subclasses thereof)
            galaxy model object containing coordinate info
        params : dict
            Dictionary containing the following:
            tauV_0 : float
                V-band optical depth at galaxy centre

        Returns
        -------
        tauV : array of floats
            optical depth of each bin
        
        """
        try:
            tauV_0 = params['tauV_0']
        except KeyError, e:
            print "Parameter '{0}' not found".format(e.message)
            raise

        tauV = np.full_like(galaxy.radius, tauV_0)

        tauV = np.clip(tauV, 0., None) # no negative values

        return tauV



class Nelson2016Dust(BaseDustModel):

    def __init__(self, log_mass):
        """Model with radially decreasing dust attenuation

        Adopts model of Nelson+ 2016 (2016ApJ...817L...9N)
        This assumes tauV = tauV_0 + c*log10(r)
        A is fixed by the stellar mass

        c=0 for galaxies with stellar masses < 10^9.136 Msun

        Contrary to Nelson+ 2016 we do not fix tauV_0
        
        To avoid numerical issues at r=0
        for r < 0.1 kpc : log10(r) = log10(0.1)
        
        """

        self.log_mass = log_mass

        #compute A power-law factor

        #get radial dependence
        c = self.radial_dependence(self.log_mass)

        #Nelson model is defined using Calzetti 2000 dust law so we must convert
        #to a Charlot & Fall 200 model

        self.c = self.scale_attenu_to_tauV(c)


    @staticmethod
    def radial_dependence(log_mass):
        """Get radial dependence

        Dependence is flat for masses < 10^9.136 Msun

        Parameters
        ----------
        log_mass : float
            logarithmic stellar mass [log10(Msun)]

        Returns
        -------
        c : floats
            radial dependence

        """

        c = -1.9 - 2.2 * (log_mass-10.)
        if c > 0:
            c = 0.

        return c


    @staticmethod
    def reddening_Calzetti2000(wave):
        """Returns Calzetti 2000 reddening
        
        Parameters
        ----------
        wave : array of floats
            wavelength [Angstrom]

        Returns
        -------
        k : array of floats
            reddening factor
        
        """
        w = wave / 10000.

        Rv = 4.05
    
        if (w >= 0.63) and (w <= 2.20):
            k = 2.659 * (-1.857 + 1.040/w) + Rv

        elif (w >= 0.12) and (w < 0.63):
            k = 2.659 * (-2.156 + 1.509/w - 0.198/w**2. + 0.011/w**3.) + Rv
        
        else:
            raise Exception("Dust law not defined outside of [1200A,22000A]")

        return k


    def scale_attenu_to_tauV(self, A_Ha):
        """Scales attenuation at Halpha to optical depth value
        
        Parameters
        ----------
        A_Ha : float
            attenuation at A_Ha

        Returns
        -------
        tauV : float
            optical depth
        
        """

        wave_Ha = 6562.8
        wave_Hb = 4861.325
        wave_V = 5500.

        k_Ha = self.reddening_Calzetti2000(wave_Ha)
        k_Hb = self.reddening_Calzetti2000(wave_Hb)

        logR = (A_Ha * (k_Hb - k_Ha)) / (2.5 * k_Ha)
        
        tauV = logR * np.log(10.) * (wave_V**-1.3 /
                                     (wave_Hb**-1.3 - wave_Ha**-1.3))

        return tauV

    def __call__(self, galaxy, params):
        """Generates dust attenuation tauV for each bin, given model parameters
        
        Produces an axisymmetric linear profile.

        Parameters
        ----------
        galaxy : instance of BaseGalaxy (or subclasses thereof)
            galaxy model object containing coordinate info
        params : dict
            Dictionary containing the following:
            tauV_0 : float
                V-band optical depth at galaxy centre

        Returns
        -------
        tauV : array of floats
            optical depth of each bin
        
        """
        try:
            tauV_0 = params['tauV_0']
        except KeyError, e:
            print "Parameter '{0}' not found".format(e.message)
            raise

        #get radii in kpc
        r = galaxy.radius / galaxy.cosmo.arcsec_per_kpc_proper(galaxy.z).value
        r = np.clip(r, 0.1, None) #limit radii smaller than 0.1kpc
        
        A = tauV_0 - self.c * np.log10(0.1)
        tauV = A + self.c * np.log10(r)

        tauV = np.clip(tauV, 0., None) # no negative values

        return tauV
