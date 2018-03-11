import numpy as np
import cvxpy as cvp

class CSSS:
### Contextually Supervised Source Seperation Class

    def __init__(self, aggregateSignal):
        self.aggregateSignal  = aggregateSignal
        self.modelcounter     = 0   # Number of source signals
        self.models           = {}  # Model for each source signal, this is a list but could be a dict
        self.constraints      = []  # Additional constraints
        self.N                = len(aggregateSignal) # Length of aggregate signal


    def addSource(self, regressor, name = None,
                  costFunction='sse',alpha = 1,      # Cost function for fit to regressors, alpha is a scalar multiplier
                  regularizeTheta='None', beta = 1,  # Cost function for parameter regularization, beta is a scalar multiplier
                  regularizeSource='None', gamma = 1, # Cost function for signal smoothing, gamma is a scalar multiplier
                  lb=None, # Lower bound on source
                  ub=None # Upper bound on source
                 ):
        ### This is a method to add a new source

        ## TO DO
        # Regularize theta
        # Regularize source signal

        self.modelcounter += 1   # Increment model counter

        ## Write model name if it doesn't exist.
        if name is None:
            name = str(self.modelcounter)

        ## Instantiate a dictionary of model terms
        model = {}
        model['name'] = name
        model['alpha'] = alpha
        model['lb']=lb
        model['ub']=ub

        ## Check regressor shape
        regressor = np.array(regressor)
        if regressor.ndim == 0: ## If no regressors are included, set them an empty array
            regressor = np.zeros((self.N,0))
        if regressor.ndim == 1:
            regressor = np.expand_dims(regressor,1)
        if regressor.ndim > 2:
            raise NameError('Regressors cannot have more than 2 dimensions')


        ## Check that regressors have the correct shape (Nobs, Nregressors)
        if regressor.shape[0] != self.N:
            if regressor.shape[1] == self.N:
                regressor = regressor.transpose()
            else:
                raise NameError('Lengths of regressors and aggregate signal must match')

        ## Define model regressors and order
        model['regressor'] = regressor
        model['order']     = regressor.shape[1]

        ## Define decision variables and cost function style
        model['source']    = cvp.Variable(self.N,1)
        model['theta']     = cvp.Variable(model['order'],1)
        model['costFunction'] = costFunction

        ## Define objective function to fit model to regressors
        if costFunction.lower() == 'sse':
            residuals = (model['source'] - model['regressor'] * model['theta'])
            modelObj =  cvp.sum_squares(residuals) * model['alpha']


        ## Define cost function to regularize theta ****************
        # ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *
        # Check that beta is scalar or of length of number of parameters.
        beta = np.array(beta)
        if beta.size not in [1, model['order']]:
            raise NameError('beta must be scalar or vector with one element for each regressor')

        if callable(regularizeTheta):
            ## User can input their own function to regularize theta.
            # Must input a cvxpy variable vector and output a scalar
            # or a vector with one element for each parameter.

            ## TODO: TRY CATCH TO ENSURE regularizeTheta WORKS AND RETURNS SCALAR
            regThetaObj = regularizeTheta(model['theta']) * beta
        elif regularizeTheta.lower() == 'l2':
            ## Sum square errors.
            regThetaObj = cvp.norm(model['theta'] * beta)
        elif regularizeTheta.lower() == 'l1':
            regThetaObj = cvp.norm(model['theta'] * beta, 1)
        else:
            print('Setting theta reg cost to 0')
            regThetaObj = 0
        ### ******* TO DO: ADD MORE MODEL FORMS.

        ## Define cost function to regularize source signal ****************
        # ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *
        # Check that gamma is scalar
        gamma = np.array(gamma)
        if gamma.size != 1:
            raise NameError('gamma must be scalar')

        ## Calculate regularization.
        if callable(regularizeSource):
            ## User can input their own function to regularize the source signal.
            # Must input a cvxpy variable vector and output a scalar.
            regSourceObj = regularizeSource(model['source']) * gamma
        elif regularizeSource.lower() == 'diff1_ss':
            regSourceObj = cvp.sum_squares(cvp.diff(model['source'])) * gamma
        else:
            regSourceObj = 0
        ### ******* TO DO: ADD MORE MODEL FORMS.


        ## Sum total model objective
        model['obj'] = modelObj + regThetaObj + regSourceObj

        ## Append model to models list
        self.models[name]= model
        return None

    def addConstraint(self, constraint):
        ### This is a method to add a new source
        self.constraints.append(constraint)
        return None

    def constructSolve(self):
        ## This method constructs and solves the optimization

        ## Initialize objective function and modeled aggregate signal as 0
        obj = 0
        sum_sources = np.zeros((self.N,1))

        ## Initialize constraints as those custom created
        con = self.constraints

        ## For each model
        #    - Add cost to objective function
        #    - Add source to sum of sources
        #    - (TODO) Add custom constraints for each source
        for name, model in self.models.items():
            obj = obj + model['obj']
            sum_sources = sum_sources + model['source']

        ## Append the constraint that the sum of sources must equal aggergate signal
        con.append(self.aggregateSignal == sum_sources)

        ## Solve problem
        prob = cvp.Problem(cvp.Minimize(obj), con)
        prob.solve()

        return None

    def admmSolve(self,rho, MaxIter=500,ABSTOL= 1e-4,RELTOL=1e-1, verbose=False):
        ### This method constructs and solves the optimization using ADMM

        ### Add the coupling constraint using lagrangian
        ##  Do Gauss-Seidel pass across each source.
        #    - Add costs to objective costFunction.
        #    - Add costs for the equality constraint.
        #    - For each source, set the price and the remaining sources constant,
        #       - Add individual constraints to individual source updates,
        #       - Solve and update the source.
        dual_objective=[]
        norm_resid_equality=[]
        aggregateSignalVector=np.array([[elem] for elem in self.aggregateSignal])
        rel_tolarance=RELTOL*np.mean(aggregateSignalVector)

        if verbose:
            print('Tolarence to aggSignal '+str(rel_tolarance))
        for k in range(0, MaxIter):
            for name, model in self.models.items():
                if k==0:
                    ### Initialize sources and old sources to zeros
                    #model['admmSource']=cvp.Variable(self.N,1)
                    model['admmSource']=np.zeros((self.N,1))
                    model['admmTheta']=np.zeros((model['order'],1))
                    #model['admmTheta']=cvp.Variable(model['order'],1)
                    y=np.zeros((self.N,1))
                    u=(1/rho)*y
                else:
                    ### Update each source by keeping the other sources constant
                    obj=0
                    con=[]
                    sum_sources = np.zeros((self.N,1))
                    for name_sub, model_sub in self.models.items():
                        ### For all the other sources, assume values constant
                        if name_sub!=name:
                            # residuals = (model['admmSource']
                            #     - (model['regressor'].dot( model['admmTheta'])))
                            sum_sources = sum_sources + model['admmSource']
                        else:
                            ### This is the only source
                            ### we are solving for at each update
                            theta_update=cvp.Variable(model['order'],1)
                            source_update=cvp.Variable(self.N,1)

                            residuals = (source_update
                            - (model['regressor'] * theta_update))

                            if model['lb'] is not None:
                                con.append(source_update >= model['lb'])
                            if model['ub'] is not None:
                                con.append(source_update <= model['ub'])
                            sum_sources = sum_sources + source_update

                        modelObj =  cvp.sum_squares(residuals) * model['alpha']
                        obj = obj + modelObj
                    obj=obj+(rho/2)*cvp.sum_squares(sum_sources-aggregateSignalVector+u)
                    prob = cvp.Problem(cvp.Minimize(obj),con)
                    last_obj=prob.solve()
                    ### Update source
                    ## keep diff for tolerance
                    source_update_diff=source_update.value-self.models[name]['admmSource']
                    self.models[name]['admmSource']=source_update.value
                    self.models[name]['admmTheta']=theta_update.value

            if k==0:
                if verbose:
                    print('Initialized all sources')
                    print('iter num','norm_resid','last_dual_objective')
            else:
                dual_objective.append(last_obj)
                u=u+(sum_sources-aggregateSignalVector).value

                norm_resid=cvp.norm(sum_sources-aggregateSignalVector).value
                eps_pri = np.sqrt(self.N)*ABSTOL + RELTOL*max(np.linalg.norm(sum_sources.value),np.linalg.norm(-aggregateSignalVector));
                norm_resid_equality.append(norm_resid)

                s_norm=np.linalg.norm(-rho*source_update_diff)
                eps_dual= np.sqrt(self.N)*ABSTOL + RELTOL*np.linalg.norm(rho*u);


                if (verbose):
                    print(k, norm_resid,last_obj,s_norm,eps_dual,norm_resid,eps_pri)


                if (s_norm<eps_dual) and (norm_resid<eps_pri):
                    break
                #print(np.transpose(y))
                #print(k, dual_objective)
        return dual_objective,norm_resid_equality,y
