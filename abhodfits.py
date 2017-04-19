"""
Module to perform power-law fits to wp(rp)
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.special import gamma as gammafn

from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import AssembiasZheng07Cens
from halotools.empirical_models import TrivialPhaseSpace
from halotools.empirical_models import AssembiasZheng07Sats
from halotools.empirical_models import NFWPhaseSpace
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables import wp
from halotools.sim_manager import CachedHaloCatalog

# fast correlation function calculation
from Corrfunc import _countpairs

# default settings of all parameters
from abhodfit_defaults import *

# A class to hold the attributes of a power-law model for wp(rp)


class ABHodFitModel():
    """
    HOD with assembly bias Fit model class for wp(rp)
    """

    ###################################################################
    # Initialize an instance of the ABHodFitModel
    def __init__(self, **kwargs):
        """
        Initialize a ABHodFitModel.
        """

        # first, set up appropriate priors on parameters
        if ('priors' in kwargs.keys()):
            self.set_prior(kwargs['priors'])
        else:
            self.set_prior(default_priors)

        # set up keys for the parameter names for plotting
        self.param_names = ['alpha', 'logM1', 'siglogM',
                            'logM0', 'logMmin', 'Acens', 'Asats']
        self.latex_param_names = [r'$\alpha$', r'$\log(M_1)$',
                                  r'$\sigma_{\log M}$', r'$\log(M_0)$', r'$\log(M_{\rm min})$',
                                  r'$\mathcal{A}_{\rm cens}$', r'$\mathcal{A}_{\rm sats}']

        # set up size parameters for any MCMC
        self.set_nwalkers(ndim=default_ndim, nwalkers=default_nwalkers)

        # if data is specified, load it into memory
        if 'rpcut' in kwargs.keys():
            self.rpcut = kwargs['rpcut']
        else:
            self.rpcut = default_rpcut

        if ('datafile' in kwargs.keys()):
            self.read_datafile(datafile=kwargs['datafile'])
        else:
            self.read_datafile(datafile=default_wp_datafile)

        if ('covarfile' in kwargs.keys()):
            self.read_covarfile(covarfile=kwargs['covarfile'])
        else:
            self.read_covarfile(covarfile=default_wp_covarfile)

        # if binfile is specified, load it into memory
        # these are Manodeep-style bins
        if ('binfile' in kwargs.keys()):
            self.binfile = kwargs['binfile']
        else:
            self.binfile = default_binfile

        # set up a default HOD Model
        if ('cen_occ_model' in kwargs.keys()):
            cen_occ_model = kwargs['cen_occ_model']
        else:
            cen_occ_model = AssembiasZheng07Cens(prim_haloprop_key='halo_mvir',
                                                 sec_haloprop_key='halo_nfw_conc')

        if ('cen_prof_model' in kwargs.keys()):
            cen_prof_model = kwargs['cen_prof_model']
        else:
            cen_prof_model = TrivialPhaseSpace()

        if ('sat_occ_model' in kwargs.keys()):
            sat_occ_model = kwargs['sat_occ_model']
        else:
            sat_occ_model = AssembiasZheng07Sats(prim_haloprop_key='halo_mvir',
                                                 sec_haloprop_key='halo_nfw_conc')

        if ('sat_prof_model' in kwargs.keys()):
            sat_prof_model = kwargs['sat_prof_model']
        else:
            sat_prof_model = NFWPhaseSpace()

        # Default HOD Model is Zheng07 with Heaviside Assembly Bias
        self.hod_model = HodModelFactory(centrals_occupation=cen_occ_model,
                                         centrals_profile=cen_prof_model,
                                         satellites_occupation=sat_occ_model,
                                         satellites_profile=sat_prof_model)

        # set pi_max for wp(rp) calculations
        self.pi_max = default_pi_max

        # set default simulation halocatalog to work with
        self.halocatalog = CachedHaloCatalog(simname=default_simname,
                                             halo_finder=default_halofinder,
                                             redshift=default_simredshift,
                                             version_name=default_version_name)

        return None
        #######################################################################
        #######################################################################

    # Set MCMC dimension and number of walkers
    def set_nwalkers(self, **kwargs):
        """
        Sets the number of MCMC dimensions and sets the number of walkers.

        Parameters
        ----------

        Takes keyword arguments ndim and nwalkers.
        """
        if ('ndim' in kwargs.keys()):
            self.ndim = kwargs['ndim']

        if ('nwalkers' in kwargs.keys()):
            self.nwalkers = kwargs['nwalkers']

        return None

    # Routine to set priors on model parameters
    def set_prior(self, prior_array):
        """
        Sets the model priors.

        Parameters
        -----------

        Takes keyword arguments r0min, r0max, gammamin, gammamax
        """

        self.priors = {}  # priors are stored in a dictionary
        if 'alpha' in prior_array.keys():
            self.priors['alpha'] = prior_array['alpha']

        if 'logM1' in prior_array.keys():
            self.priors['logM1'] = prior_array['logM1']

        if 'sigma_logM' in prior_array.keys():
            self.priors['sigma_logM'] = prior_array['sigma_logM']

        if 'logM0' in prior_array.keys():
            self.priors['logM0'] = prior_array['logM0']

        if 'logMmin' in prior_array.keys():
            self.priors['logMmin'] = prior_array['logMmin']

        if 'mean_occupation_centrals_assembias_param1' in prior_array.keys():
            self.priors['mean_occupation_centrals_assembias_param1'] = prior_array[
                'mean_occupation_centrals_assembias_param1']

        if 'mean_occupation_satellites_assembias_param1' in prior_array.keys():
            self.priors['mean_occupation_satellites_assembias_param1'] = prior_array[
                'mean_occupation_satellites_assembias_param1']

        return None

    # Read in the data
    def read_datafile(self, **kwargs):
        """
        Read in the data. This can assume the input_data_file attribute or it 
        can accept a new data file as a keyword argument, datafile.
        """
        if ('datafile' in kwargs.keys()):
            self.datafile = kwargs['datafile']

        col1, col2, col3 = np.loadtxt(self.datafile, unpack=True)
        self.Number_gals = col1[0]
        self.ngal = col2[0]
        self.ngalerr = col3[0]

        self.rp = col1[1:]
        self.wp = col2[1:]
        self.wperr = col3[1:]

        # Use data only out to the bin at rp=rpcut
        ikeep = np.where(self.rp <= self.rpcut)
        self.rp = self.rp[ikeep]
        self.wp = self.wp[ikeep]
        self.wpT = self.wp.T
        self.wperr = self.wperr[ikeep]

        # set the number of bins
        self.nrpbins = self.rp.size

        return None

    # Read in the data covariances
    def read_covarfile(self, **kwargs):
        """
        Read in the data covariances
        """

        if ('covarfile' in kwargs.keys()):
            self.covarfile = kwargs['covarfile']
        else:
            self.covarfile = default_wp_covarfile

        self.covar = np.loadtxt(self.covarfile, unpack=True)
        self.cov_inv = np.linalg.inv(self.covar)

    # A routine to give wp(rp) computed via a halo model.
    def wp_hod(self, hod_parameters):
        """
        An HOD model for wp(rp) computed by direct simulation 
        population.
        hod_parameters[0] : alpha
        hod_parameters[1] : logM1
        hod_parameters[2] : sigma_logM
        hod_parameters[3] : logM0
        hod_parameters[4] : logMmin
        hod_parameters[5] : Acen
        hod_parameters[6] : Asat
        """

        # The first step is to set the param_dict of the hod_model.
        self.hod_model.param_dict['alpha'] = hod_parameters[0]
        self.hod_model.param_dict['logM1'] = hod_parameters[1]
        self.hod_model.param_dict['sigma_logM'] = hod_parameters[2]
        self.hod_model.param_dict['logM0'] = hod_parameters[3]
        self.hod_model.param_dict['logMmin'] = hod_parameters[4]
        self.hod_model.param_dict[
            'mean_occupation_centrals_assembias_param1'] = hod_parameters[5]
        self.hod_model.param_dict[
            'mean_occupation_satellites_assembias_param1'] = hod_parameters[6]

        # Populate a mock galaxy catalog
        # self.hod_model.populate_mock()
        try:
            self.hod_model.mock.populate()
        except:
            self.hod_model.populate_mock(self.halocatalog)

        # Instruct wp(rp) routine to compute autocorrelation
        autocorr = 1
        # Number of threads
        nthreads = 4

        # use the z-direction as line-of-sight and add RSD
        z_distorted = self.hod_model.mock.galaxy_table[
            'z'] + self.hod_model.mock.galaxy_table['vz'] / 100.0

        # enforce periodicity of the box
        self.hod_model.mock.galaxy_table[
            'zdist'] = z_distorted % self.hod_model.mock.Lbox

        # Return projected correlation function computed using
        # Manodeep Simha's optimized C code.
        cpout = np.array(_countpairs.countpairs_wp(self.hod_model.mock.Lbox,
                                                   self.pi_max,
                                                   nthreads,
                                                   self.binfile,
                                                   self.hod_model.mock.galaxy_table[
                                                       'x'].astype('float32'),
                                                   self.hod_model.mock.galaxy_table[
                                                       'y'].astype('float32'),
                                                   self.hod_model.mock.galaxy_table['zdist'].astype('float32')))

        return cpout[:, 3]

        # A routine to compute wp(rp) in a power-law model xi = (r/r0)^-gamma.
    def wp_powerlaw_model(self, rsep, r0, gamma):
        """
        Power law model for wp(rp) assuming that xi = (r/r0)^-gamma.
        """
        if rsep.any < 0.0:
            return np.inf
        return rsep * (rsep / r0)**(-gamma) * gammafn(0.5) * gammafn((gamma - 1.0) / 2.0) / gammafn(gamma / 2.0)

    # A routine to compute the likelihood of a power-law wp(rp) given data.
    def lnlike(self, theta):
        """
            Log likelihood of a power-law model.

            Parameters
            ------------
            theta : (alpha,logM1,sigma_logM,logM0,logMmin)
            rp : numpy array containing separations
            wp : numpy array containing projected correlation functions
            wperr : numpy array containing errors on measured wp
        """

        # log likelihood from clustering
        wpmodel = self.wp_hod(theta)
        # print 'wpmodel shape = ',np.shape(wpmodel)
        # print 'wperr shape = ',np.shape(self.wperr)
        # print 'wp shape = ',np.shape(self.wp)
        # print 'rp shape = ',np.shape(self.rp)
        wp_dev = (wpmodel - self.wp)
        wplike = -0.5 * np.dot(np.dot(wp_dev, self.cov_inv), wp_dev)

        # log likelihood from number density
        number_gals = len(self.hod_model.mock.galaxy_table)
        ngal = number_gals / (self.hod_model.mock.Lbox**3)
        ng_theory_error = ngal / np.sqrt(number_gals)
        nglike = -0.5 * ((ngal - self.ngal)**2 /
                         (self.ngalerr**2 + ng_theory_error**2))

        return wplike + nglike

        # A prior function on the two parameters in the list theta
    def lnprior(self, theta):
        """
        Prior function

                    Parameters
                    ----------
                    theta : [alpha,logM1,sigma_logM,logM0,logMmin,Acen,Asat]
        Priors are so-called hard priors specified in self.priors.
        """
        alpha, logM1, sigma_logM, logM0, logMmin, Acen, Asat = theta
        if alpha < self.priors['alpha'][0]:
            return -np.inf
        if alpha > self.priors['alpha'][1]:
            return -np.inf

        if logM1 < self.priors['logM1'][0]:
            return -np.inf
        if logM1 > self.priors['logM1'][1]:
            return -np.inf

        if sigma_logM < self.priors['sigma_logM'][0]:
            return -np.inf
        if sigma_logM > self.priors['sigma_logM'][1]:
            return -np.inf

        if logM0 < self.priors['logM0'][0]:
            return -np.inf
        if logM0 > self.priors['logM0'][1]:
            return -np.inf

        if logMmin < self.priors['logMmin'][0]:
            return -np.inf
        if logMmin > self.priors['logMmin'][1]:
            return -np.inf

        if Acen < self.priors['mean_occupation_centrals_assembias_param1'][0]:
            return -np.inf
        if Acen > self.priors['mean_occupation_centrals_assembias_param1'][1]:
            return -np.inf

        if Asat < self.priors['mean_occupation_satellites_assembias_param1'][0]:
            return -np.inf
        if Asat > self.priors['mean_occupation_satellites_assembias_param1'][1]:
            return -np.inf

        return 0.0

    # The probability function including priors and likelihood
    def lnprob(self, theta):
        """
        Probability function to sample in an MCMC.
        """
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)

    # Set the starting position of an MCMC.
    def set_start_position(self, theta_start=default_start):
        """
        Set the starting position for the MCMC.
        """

        self.position = np.zeros([self.nwalkers, self.ndim])

        for iparam in range(self.ndim):
            self.position[:, iparam] = theta_start[iparam] + \
                0.05 * np.random.randn(self.nwalkers)

        return None

    # Fit the data with the hod model model
    def mcmcfit(self, theta_start, **kwargs):
        """
        Peform an MCMC fit to the wp data using the power-law model.
        """

        if ('samples' in kwargs.keys()):
            self.nsamples = kwargs['samples']
        else:
            self.nsamples = 100

        if ('nwalkers' in kwargs.keys()):
            self.set_nwalkers(nwalkers=kwargs['nwalkers'])

        self.wpsampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.lnprob)

        self.set_start_position(theta_start)

        if ('burnin' in kwargs.keys()):
            self.nburnin = kwargs['burnin']
            self.position, self.prob, self.state = self.wpsampler.run_mcmc(
                self.position, self.nburnin)
            self.wpsampler.reset()
        else:
            self.nburnin = 0

        self.wpsampler.run_mcmc(self.position, self.nsamples)

        self.mcmcsamples = self.wpsampler.chain[
            :, :, :].reshape((-1, self.ndim))
        self.lnprobability = self.wpsampler.lnprobability.reshape(-1, 1)

        self.compute_parameter_constraints()

        return None

    # given a set of MCMC samples in self.mcmcsamples, compute the 1-D
    # parameter constraints.
    def compute_parameter_constraints(self):
        """
        Computes the 1D marginalized parameter constraints from 
        self.mcmcsamples.
        """
        self.alpha_mcmc, self.logM1_mcmc, self.sigma_logM_mcmc, self.logM0_mcmc, self.logMmin_mcmc, self.Acen_mcmc, self.Asat_mcmc = map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(self.mcmcsamples, [16, 50, 84], axis=0)))

        self.alpha_1side, self.logM1_1side, self.sigma_logM_1side, self.logM0_1side, self.logMmin_1side, self.Acen_1side, self.Asat_1side = map(
            lambda v: (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]), zip(*np.percentile(self.mcmcsamples, [1, 5, 10, 16, 84, 90, 95, 99], axis=0)))

        return None

    # save chains to file
    def save_chains(self, **kwargs):
        """
        Save the chain to an ascii file.
        """

        if ('filename' in kwargs.keys()):
            fname = kwargs['filename']
        else:
            fname = self.datafile
            if fname.endswith('.dat'):
                fname = fname[:-4]
            fname = fname + '_abfit.chain'

        out_data = np.hstack((self.mcmcsamples, self.lnprobability))
        np.savetxt(fname, out_data, delimiter='  ')

        return None

    # load a chain from an existing chain file
    def load_chains(self, chainfile_names):
        """
        Load a pre-existing chain into memory from a file.
        """

        for chainfile in chainfile_names:

            read_data = np.loadtxt(chainfile, unpack=True).T
            samples = read_data[:, 0:self.ndim]
            lnprob = read_data[:, self.ndim]
            samples = samples.reshape(-1, self.ndim)
            lnprob = lnprob.reshape(-1, 1)

            if (hasattr(self, 'mcmcsamples')):
                self.mcmcsamples = np.concatenate(
                    (self.mcmcsamples, samples), axis=0)
                self.lnprobability = np.concatenate(
                    (self.lnprobability, lnprob), axis=0)
            else:
                self.mcmcsamples = samples
                self.lnprobability = lnprob

#    self.mcmcsamples=self.mcmcsamples.reshape(-1,self.ndim)
#    self.lnprobability=self.lnprobability.reshape(-1,1)

        return None

    # plot the data
    def plot_data(self):
        """
        plot the data only
        """
        fig1 = plt.figure()
        plt.loglog(self.rp, self.wp, 'sk')
        plt.errorbar(self.rp, self.wp, yerr=self.wperr, fmt='sk', ecolor='k')
        plt.xlabel(r'$r_{\rm p}$')
        plt.ylabel(r'$w_{\rm p}(r_{\rm p})$')
        fig1.savefig('wpdata.png')

        return None

    # plot the mcmc run
    def plot_parameter_run(self):
        """
        plot the mcmc samples.
        """
        for idim in range(self.ndim):
            fig = plt.figure()
            plt.plot(range(len(self.mcmcsamples[:, idim])), self.mcmcsamples[
                     :, idim], 'k')
            plt.xlabel('sample number')
            plt.ylabel(self.latex_param_names[idim])
            filename = self.datafile
            if filename.endswith('.dat'):
                filename = filename[:-4]
            plabel = self.param_names[idim].strip('$')
            plabel = plabel.strip('\\')
            filename = filename + '_' + plabel + '_' + 'chain.png'
            fig.savefig(filename)
            del fig

    # plot fitting results
    def plot_chain_samples(self, **kwargs):
        """
        Plots samples from the mcmc chain alongside the data.
        """

        lnp_max = np.max(self.lnprobability)
        if ('deltachi2' in kwargs.keys()):
            lnp_threshold = lnp_max - kwargs['deltachi2'] / 2.0
        else:
            lnp_threshold = lnp_max - 0.5

        if ('samples' in kwargs.keys()):
            numsamples = kwargs['samples']
        else:
            numsamples = 50

        fig = plt.figure()

        # print ' + Beginning sample selection.'
        igood = np.where((self.lnprobability.reshape(-1) > lnp_threshold))
        # print 'numsamples = ',numsamples
        # print 'Length(igood) = ',len(igood[0])
        # print 'Shape(igood) = ',np.shape(igood)

        # reduce the number of samples if there are very few
        if len(igood[0]) < 1:
            print ' > Too few samples that satisfy criterion, n = ', len(igood[0])
            return None
        elif len(igood[0]) < numsamples:
            numsamples = len(igood[0]) - 1

        good_models = self.mcmcsamples[igood]

        # print 'igood is ',igood
        # print 'good_models = ',good_models
        # print 'numsamples = ',numsamples

        irandoms = np.random.randint(len(good_models), size=numsamples)
        # print 'len(irandoms) = ',len(irandoms)
        # print 'irandoms = ',irandoms

        for parameters in good_models[irandoms, :]:
            # print 'Within for loop'
            # print 'Parameters are',parameters
            plt.loglog(self.rp, self.wp_hod(parameters), color='k', alpha=0.07)

        plt.xlim(0.92 * np.min(self.rp), 1.05 * np.max(self.rp))

        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.xlabel(r'$r_{\rm p}$ [$h^{-1}$Mpc]', fontsize=20)
        plt.ylabel(r'$w_{\rm p}(r_{\rm p})$ [$h^{-1}$Mpc]', fontsize=20)

        # plt.loglog(self.rp,self.wp,'sk')
        plt.errorbar(self.rp, self.wp, yerr=self.wperr, fmt='s',
                     color='firebrick', ecolor='firebrick')

        filename = self.datafile
        if filename.endswith('.dat'):
            filename = filename[:-4]
        filename = filename + '_chainsamples.pdf'
        fig.savefig(filename, format='pdf', bbox_inches='tight')
