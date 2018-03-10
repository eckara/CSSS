# Contextually Supervised Source Separation (CSSS)
This packages deploys contextually supervised source separation (CSSS) techniques in python. CSSS was originally decribed by Wytock and Kolter in [1] and we follow their original notation in this docuemntation.  We also include updates and extensions of the origianl CSSS method as created by the authors of this package in their applied work [2-4]. 	

CSSS is the disaggregation of a time series of source signals from observations of their sum. Equation~\ref{} displays this problem where the vector $\bar y$ is observed and each source signal, $y_i$ is unobserved. 
\begin{equation}
\bar y = \sum_{i=0}^{k} y_i
\end{equation}

Contextual separation is achieved by (A) relating each source signal to exogenous observations, and (B) exploiting known regularity in the source signals. 
Equation~\ref{} displays the general form of this optimization as presented in [1]. $\ell_i()$ is a cost function to fit the source signal, $y_i$ to the exogenous variables, $X_i$, using a linear transformation with parameters $theta_i$; $g_i()$ is a regularization term applied to the source signal which would elicit smoothness or other periodic regularity; and $h_i()$ is a regularization term for the parameters $\theta_i$. 

\begin{align}
\min_{Y,\theta}\hspace{5mm}& \sum_{i=0}^{k} \{ \ell_i(y_i, X_i \theta_i) + g_i(y_i) + h_i(\theta_i)\}\\
\mbox{s.t.,}\hspace{5mm}&  \bar y = \sum_{i=0}^{k} y_i
\end{align}

## Objects
The CSSS package is built upon the CSSS object class, which allows users to fully specifcy and fit a problem. 

A CSSS objest is initailized with only a source signal for disaggregation, in this case the numpy array, Y. 
```python
import CSSSpy
CSSSobject = CSSSpy.CSSS(Y)
```

### Attributes
- `models` is a dictionary of models for each component sources.  Adding sources and their properties are addressed below. 
- `constraints` is a list of additional constraints. Adding constraints is addressed below. 
- `aggregateSignal` is the aggregate signal
- `N` is the number of observations in the aggregate signal. 
- `modelcounter` is the total number of source models included. 

### Adding Sources
The method CSSS.addSource adds a model for a source signal. By default, the model cost function is the sum of square errors, $\left|\left| y_i - X_i \theta_i \right|\right|_2^2$_, and there is no regularization of the source signal or the parameters. Alternate options for this form (i.e. other norms) will be included in future versions of this package. 
```python
CSSSobject.addSource(X1, name = 'y1')  ## Add a model for source signal y1
CSSSobject.addSource(X2, name = 'y2')  ## Add a model for source signal y2
```
The optional parameter `alpha` is a salar that weights the cost of the signal in the objective function. In the following example, costs associated with the errors in the model for `y1` will be weighted twice that of those for `y2`. 

#### Parameter Regularization
The `regularizeTheta` input to `addSource` defines the $h_i()$ term for the source and takes either a string or a function. Strings define standard regularizations and can take "ss" for sum of squares, 'l1' for l1-norms, and 'l2' for l2-norms. `beta` is a parameter for linearly scaling the regularization term in the overall objective function. `beta` may take a scalar or a vector value, if a vector there must be one element for element of $theta_i$.
```python
CSSSobject.addSource(X1, name = 'y1', regularizeTheta='L2', beta = 2)  ## Add a model for source signal y1
```

If inputing a custom function to regularize theta, the function must input a vector of cvxpy variables, and output a scalar that goes into the objective function, and must be convex. The `beta` term can still be used to scale this function. 
```python
import cvxpy

def customReg(x):
	cvxpy.sum_entries(cvxpy.power((x-2),2))

CSSSobject.addSource(X1, name = 'y1', regularizeTheta=customReg)  ## Add a model for source signal y1
```

#### Source Regularization
The `regularizeSource` input to `addSource` defines the $g_i()$ term for the source and takes either a string or a function. Strings define standard regularizations and currently only take "diff1_ss" for sum of the squared differenced source signal. `gamma` is a parameter for linearly scaling the regularization term in the overall objective function. 
```python
CSSSobject.addSource(X1, name = 'y1', regularizeSource='diffss', beta = .1)  ## Add a model for source signal y1
```

If inputing a custom function to regularize the source, the function must input a vector of cvxpy variables and output a scalar that goes into the objective function, and must be convex. The `gamma` term can still be used to scale this function. 
```python
import cvxpy

def customReg(x):
	## Custom regularization function for a source signal with a biased increase
	cvxpy.norm(cvxpy.Diff(x)-.01,1)

CSSSobject.addSource(X1, name = 'y1', regularizeSource=customReg)  ## Add a model for source signal y1
```

#### Anatomy of a source model
TODO.  Inlcude all attributes of a source, how to acceess the cvxpy variables to add constraints etc. 

### Adding Constraints

### Fitting Models

### Distributed Optimizzation
