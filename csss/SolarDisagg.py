import csss as CSSS
import numpy as np

class SolarDisagg_IndvHome(CSSS.CSSS):
    def __init__(self, netloads, solarregressors, loadregressors, names = None):
        ## Inputs
        # netloads:         np.array of net loads at each home, with columns corresponding to entries of "names" if available.
        # solarregressors:  np.array of solar regressors (N_s X T)
        # loadregressors:   np.array of load regressors (N_l x T)

        ## Find aggregate net load, and initialize problem.
        agg_net_load = np.sum(netloads, axis = 1)
        CSSS.CSSS.__init__(self, agg_net_load)

        ## If no names are input, create names based on id in vector.
        self.N, self.M = netloads.shape
        if names is None:
            self.names = [str(i) for i in np.arange(self.M)]
        else:
            self.names = names

        ## Store net loads as a dictionary
        self.netloads = {}
        for i in range(self.M):
            name = self.names[i]
            self.netloads[name] = netloads[:,i]

        ## Store solar and load regressors, solar regressors, and begin true solar dict
        self.solarRegressors = solarregressors
        self.loadRegressors  = loadregressors
        self.trueValues      = {}

        ## Cycle through each net load, and create sources.
        for source_name in self.names:
            self.addSource(regressor=solarregressors, name = source_name, alpha = 1)

            ## Add constraints that solar generation cannot exceed zero or net load.
            self.addConstraint( self.models[source_name]['source'] <= np.array(self.netloads[source_name]) )
            self.addConstraint( self.models[source_name]['source'] <= 0 )

        ## Add the aggregate load source
        self.addSource(regressor=loadregressors, name = 'AggregateLoad', alpha = 1)
        self.addConstraint( self.models['AggregateLoad']['source'] > 0 )

    def addTrueValue(self, trueValue, name):
        ## Function to add true solar for a given model

        ## Check that true value is correct number of dimensions
        trueValue = trueValue.squeeze()
        if not (trueValue.shape == (self.N,)):
            raise Exception('True value of a solar or load signal must be one dimensional and length N = %d' % self.N)

        if name not in (self.names + ['AggregateLoad']):
            raise Exception('Must input a valid household identifier or \"AggregateLoad\"')

        ## Add True Value
        self.trueValues[name] = trueValue
        return(None)

    def performanceMetrics(self, dropzeros = False):
        ## Function to calculate performance metrics
        # Dropping zeros is intended to remove nightime solar.
        self.rmse = {}
        self.cv   = {}

        for name in self.trueValues.keys():
            truth = self.trueValues[name]
            est   = np.array(self.models[name]['source'].value).squeeze()

            ## Calculate metrics.
            self.rmse[name] = np.sqrt(np.mean((truth-est)**2))
            self.cv[name] = self.rmse[name] / np.mean(truth)


        return(None)

    def tuneAlphas(self, tuneSys = None, filter_vec = np.ones(12)/12.0, var_lb_fraction = 0.01):
        ## Function to autotune alphas given some true solar information.
        if tuneSys is None:
            ## If no name for a tuning system is input, use all systems for which
            # a truth is known.
            tuneSys = self.trueValues.keys()
            if 'AggregateLoad' in tuneSys: tuneSys.remove('AggregateLoad')


        ## For each system used for tuning, filter the square residuals.
        filt_sq_resid_norm = np.ones((self.N,len(tuneSys)))
        i=0
        for name in tuneSys:
            print(name)
            truth      = self.trueValues[name].squeeze()
            modelest   = self.models[name]['regressor'] * self.models[name]['theta']
            modelest   = np.array(modelest.value).squeeze()

            ## Create a rough capacity estimate from the theta values
            capest     = np.sum(self.models[name]['theta'].value)
            sq_resid   = ( (truth - modelest) / capest ) ** 2
            filt_sq_resid_norm[:,i] = convolve_cyc( sq_resid , filter_vec )
            i=i+1

        ## Average the filtered squared residuals
        ave_filt_sq_resid_norm = np.mean(filt_sq_resid_norm, axis = 1)
        print(np.min(filt_sq_resid_norm, axis = 0))
        print(np.min(ave_filt_sq_resid_norm, axis = 0))

        ## Create alphas for each other PV system
        total_sol_var   = np.zeros(self.N) ## Instantiate vector for total variance of PV signals,
        total_model_est = np.zeros(self.N) ## Instantiate vector for linear model prediction of net load.

        ## Cycle through each solar model and tune alphas
        for name in self.models.keys():
            ## Model estimated value
            model_est = self.models[name]['regressor'] * self.models[name]['theta'] ## model estimate
            model_est = np.array(model_est.value).squeeze()
            total_model_est = total_model_est + model_est             ## total model estimate

            ## Don't solve for aggregate load yet
            if name.lower() == 'aggregateload':
                continue

            capest      = np.sum(self.models[name]['theta'].value)  ### Rough capacity estimate
            mean_abs_nl = np.mean(np.abs( self.netloads[name] ))    ### Mean absolute net load
            lb_var      = (mean_abs_nl * var_lb_fraction) ** 2                  ### Lower bound on variance
            sol_var     = ave_filt_sq_resid_norm * (capest ** 2)    ### Solar variance (unconstrained)
            sol_var[sol_var < lb_var ] = lb_var                     ### Constrain the solar variance
            total_sol_var = total_sol_var + sol_var                 ### Track the total variance of solar

            alpha = sol_var ** -1         ## alpha
            self.models[name]['alpha'] = alpha
            print((lb_var,np.max(alpha)))

        ## Tune load alphas.
        lb_var            = (np.mean(np.abs(self.aggregateSignal)) * var_lb_fraction) ** 2 ## LOWER BOUND OF VARIANCE, 1%
        total_residual_sq = (self.aggregateSignal.squeeze() - total_model_est.squeeze()) ** 2 ## Square residuals of aggregate signal prediction
        total_var_filt    = convolve_cyc(total_residual_sq, filter_vec)   ## Filter square residuals as variance estimate
        load_var_est      = total_var_filt - total_sol_var                ## Estimate of load variance
        load_var_est[load_var_est < lb_var] = lb_var                      ## Enforce lower bound on variance
        alpha = load_var_est ** -1
        self.models['AggregateLoad']['alpha'] = alpha
        print((lb_var,np.max(alpha)))


        ## Scale all alphas
        self.scaleAlphas()
        self.updateSourceObj('all')
        return(None)

    def scaleAlphas(self, scale_to = 1.0):
        ## Find the maximum value of alpha
        alpha_max = 0
        for name, m in self.models.items():
            if np.max(m['alpha']) > alpha_max:
                alpha_max = np.max(m['alpha'])


        ## Scale other values of alpha
        for name, m in self.models.items():
            m['alpha'] = np.array( m['alpha'] / alpha_max * scale_to ).squeeze()
            self.updateSourceObj(name)



        return(None)



## Function for a cyclic convolution filter, because apparantely there isn't one already in python
def convolve_cyc(x, filt, left = True):
    if (len(filt) % 2) == 1:
        pad_l = np.int((len(filt)-1)/2)
        pad_r = pad_l
    elif left:
        pad_l = np.int(len(filt)/2)
        pad_r = pad_l-1
    else:
        pad_r = np.int(len(filt)/2)
        pad_l = pad_r-1


    x = np.concatenate([x[-pad_l:], x, x[:pad_r]])
    x = np.convolve(x, filt, mode = 'valid')
    return(x)
