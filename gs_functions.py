import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution

def calibrate(obj, bounds, n_iter, disp, *args):
    """
    Compute the parameter vector, theta, that minimises a given objective function, obj.

    Parameters:
    -----------
    obj : callable
        Objective function to minimise.
    bounds : array-like
        List of 2-tuples corresponding to the boundaries of the the parameter values.
    n_iter : float
        Number of iterations to perform before returning theta.
    disp : boolean
        If true, displays intermittent progress amounts while calibrating.
    *args :
        Additional arguments to pass to objective function.
        
    Returns:
    --------
    theta : array-like
        Vector of calibrated parameter values.
    """

    # Get progress function for number of iterations if disp is true
    if disp:
        progress_func = progress(n_iter)
    else:
        progress_func = None

    # Minimise the objective function using differential evolution
    result = differential_evolution(obj(*args), bounds, maxiter=n_iter, recombination=1, seed=19, callback=progress_func)

    # Extract and return calibrated parameters
    theta = result.x
    return theta

def import_data(*file_names):
    """
    Import data from text files.

    Parameters:
    -----------
    *file_names : strings
        Names of files from which to import data.

    Returns:
    --------
    data : array-like
        Contains data in a 1-dimensional array sorted by filename input order and then by column order.
    """

    # Initialise array to store data arrays
    data = []

    # Iterate over each file
    for file_name in file_names:
        # Import the data from the file and add this to the data array
        t, y = np.genfromtxt(file_name, delimiter=",", skip_header=True).T
        data.append(t)
        data.append(y)
 
    # Return the imported data
    return data

def model(t, P, *theta):
    """
    Return the pressure model derivative function for given parameters.

    Parameters:
    -----------
    t : array-like
        Time value at which to compute q(t).
    P : array-like
        Mass flow rate data used to compute q(t).
    *theta : other inputs to the pressure ODE which include:
        tqd = time values at which we have mass flow rate data
        qd = mass flow rate data values 
        Model parameters:
            a = mass flow strength parameter (Pa/kg);
            beta = leakage strength parameter (1/(MPa*yr));
            P0 = ambient pressure (MPa); and
            Pc = critical overpressure (MPa).
            
    Returns:
    --------
    dPdt: float
        Value of the model derivative, dP/dt, at time, t, for given parameters.
    """

    # Unpack parameters
    tqd, qd, a, beta, P0, Pc = theta
        
    # Determine b for given parameters
    if (P - P0) >= Pc:  # leakage occurs if overpressure exceeds threshold value
        b = beta * (P - P0)
    else:
        b = 0
        
    # Compute and return dP/dt
    dPdt = (a * np.interp(t, tqd, qd)) - (b * (P - P0))
    return dPdt

def get_q(t,A,B,C,D):
    """
    Return array of mass rate flow q.

    Parameters:
    -----------
    t : array-like
        Time value at which to compute q(t).
    A : float
        parameter A in A+Bsin(Ct+D)
    B : float
        parameter B in A+Bsin(Ct+D)
    C : float
        parameter C in A+Bsin(Ct+D)
    D : float
        parameter D in A+Bsin(Ct+D)
        
    Returns:
    --------
    q : array-like
        mass flow rate values.
    """

    q = A + (B * np.sin((C*t) + D))
    return q

def obj_q(tqd, qd):
    """
    Return the pressure objective function for given pressure data.

    Parameters:
    -----------
    tqd : array-like
        Time data.
    qd : array-like
        Mass flow rate data.
            
    Returns:
    --------
    model_func: callable
        Function that returns the objective, s, for given parameters.
    """

    def obj_q_func(theta):
        """
        Evaluate the objective, the sum of squared residuals.

        Parameters:
        -----------
        theta : array-like 
            Model parameters in order (A, B, C, D), where:
                q = A + (B * np.sin((C*t) + D))
        
        Returns:
        --------
        s : float
            Objective value at theta, the sum of squared residuals.
        """

        # Initialise objective
        s = 0
        # unpack parameter values from theta
        [A,B,C,D] = theta
        # Compute mass flow rates using sine model
        for i in range(len(tqd)):
            q = get_q(tqd[i],A,B,C,D)
            s = s + ((qd[i]-q)**2)
        # Return objective value
        return s

    # Return objective function
    return obj_q_func

def obj_P(tqd, qd, tPd, Pd):
    """
    Return the pressure objective function for given pressure data.

    Parameters:
    -----------
    tqd : array-like
        Time data used to compute derivative.
    qd : array-like
        Mass flow rate data used to compute derivative.
    tPd : array-like
        Time data used to compute pressure model.
    Pd : array-like
        Pressure data used to compute sum of squared residuals.
            
    Returns:
    --------
    model_func: callable
        Function that returns the objective, s, for given parameters.
    """

    # Define objective function
    def obj_P_func(theta):
        """
        Evaluate the objective, the sum of squared residuals.

        Parameters:
        -----------
        theta : array-like 
            Model parameters in order (a, beta, P0, Pc), where:
                a = mass flow strength parameter (Pa/kg);
                beta = leakage strength parameter (1/(MPa*yr));
                P0 = ambient pressure (MPa); and
                Pc = critical overpressure (MPa).
        
        Returns:
        --------
        s : float
            Objective value at theta, the sum of squared residuals.
        """
        t = np.arange(tPd[0],tPd[-1]+0.01,0.01)
        # Solve the pressure model using with given parameters
        P = solve_ode(model, t, 25.16, tqd, qd, *theta)
        # Compute and return sum of squared residuals between the model and pressure data
        indices = np.arange(0,1025,25)  # quarterly pressure data available
        # initialise objective function value
        s = 0
        # Sum squares of residuals
        for i in range(len(indices)):
            s = s + ((Pd[i]-P[indices[i]])**2)
        return s
    # Return objective function
    return obj_P_func

def progress(n_iter):
    """
    Return a function that displays calibration progress for a given number of total iterations.

    Parameters:
    -----------
    n_iter : float
        Total number of iterations of calibration algorithm.

    Returns:
    --------
    progress_func : callable
        Function that displays progress percentages during calibration.
    """

    # Define progress-printing function for given number of iterations
    def progress_func(theta, convergence):
        """
        Parameters:
        -----------
        theta : array-like
            Current parameter values at the end of the iteration (unused).
        convergence : float
            Fractional value of convergence (unused).
        """

        # Get current iteration number
        global it

        # Display percentage every few iterations
        if it % np.ceil(n_iter / 10) == 0:
            percentage = np.floor(it / n_iter * 100)
            print("    Progress: {:.0f}%".format(percentage))

        # Increment iteration count
        it += 1

        # Reset iteration count at end of calibration
        if it == n_iter:
            print("    Progress: 100%")
            it = 0

    # Return progress-printing function
    return progress_func

def solve_ode(f, t, y0, *args):
    """
    Solve an ODE numerically using Improved Euler method.

    Parameters:
    -----------
    f : callable
        Function that returns ode function that returns dydt at given values of t and y.
    t : array-like
        Dependent variable values at which to compute the solution y.
    y0 : float
        Initial value of solution.
    *args : array-like
        Additional arguments to pass to ode function.

    Returns:
    --------
    y : array-like
        Independent variable solution vector.
    """

    y = 0.*t							# array to store solution
    y[0] = y0							# set initial value
    dt = t[1]-t[0]
    for i in range(len(t)-1):
        # find the value predicted by an euler step
        euler_y = y[i] + (dt * f(t[i],y[i],*args))

        # evaluate the average slope of the derivatives at yk and euler_y 
        ave_slope = (1/2) * (f(t[i],y[i],*args)+f(t[i]+dt,euler_y,*args))
        # find the improved euler step
        y1 = y[i] + (dt*ave_slope)
        
        y[i+1]=y1

    return y

def leakage_func(t,L,P,tp,a,beta,P0,Pc):
    """
    This function returns the value of the leakage ode given the specific inputs

        Parameters:
        -----------
        t : float
            Dependent variable value at which to compute the solution y.
        L : array-like
            Value of the independent variable L (required for solve_ode)
        P : array-like
            Array containing pressure model values.
        tp : array-like
            Time array corresponding to pressure model values.
        a : float
            Calibrated mass flow strength parameter.
        beta : float
            Calibrated leakage strength parameter.
        P0 : float
            Calibrated ambient pressure parameter.
        Pc : float
            Calibrated critical overpressure parameter.

        Returns:
        --------
        dLdt : value of leakage ode for given inputs
    """

    # find pressure value at time t
    Pt = np.interp(t, tp, P)
    if Pt-P0<Pc:    # no leakage occurs when P-Pc<Pc
        b=0
    else:
        b = beta*(Pt-P0)
    dLdt = b*(Pt-P0)/a
    return dLdt

def grid_search(theta_P,tqd,qd,tpd,pd):
    """
    This function implements a grid search to compute the posterior over a, beta, P0 and Pc.

        Parameters:
        -----------
        theta_P : array-like
            Array of pressure parameter values: [a,beta,P0,Pc].
        tqd : array-like
            Times for mass flow rate data
        qd : array-like
            Mass flow rate data.
        tpd : array-like
            Times for pressure data.
        pd : float
            Pressure data.

        Returns:
        --------
        a : array-like
            Vector of a parameter values.
        beta : array-like
            Vector of beta parameter values.
        P0 : array-like
            Vector of P0 parameter values.
        Pc : array-like
            Vector of Pc parameter values.
        P : array-like
            Posterior probability distribution.

    """

    # unpack parameter values
    a_best,beta_best,P0_best,Pc_best = theta_P

    # number of values considered for each parameter within a given interval
    N = 7

    # vectors of parameter values
    a = np.linspace(a_best/2,a_best*1.5, N)
    beta = np.linspace(beta_best/2,beta_best*1.5, N)
    P0 = np.linspace(P0_best/2,P0_best*1.5, N)
    Pc = np.linspace(Pc_best/2,Pc_best*1.5, N)

    # grid of parameter values: returns every possible combination of parameters in a and b
    a1, beta1, P01, Pc1 = np.meshgrid(a, beta, P0, Pc, indexing='ij')
   
    # empty 2D matrix for objective function
    S = np.zeros(a1.shape)

    # error variance - 
    v = 0.5
    t = np.arange(2009,2019.01,0.01)
    # grid search algorithm
    for i in range(len(a)):
        for j in range(len(beta)):
            for k in range(len(P0)):
                for l in range(len(Pc)):
                    # 2. compute the sum of squares objective function at each value 
                    pm = solve_ode(model,tpd,25.16,tqd,qd,a[i],beta[j],P0[k],Pc[l])
                    S[i,j,k,l] = np.sum((1/(v**2))*((pd-pm)**2))
                
    # 3. compute the posterior
    P = np.exp((-1*S)/2)
    
    # normalize to a probability density function
    Pint = np.sum(P)*(a[1]-a[0])*(beta[1]-beta[0])*(P0[1]-P0[0])*(Pc[1]-Pc[0])
    P = P/Pint


    return a,beta,P0,Pc,P

def fit_mvn(parspace, dist):
    """Finds the parameters of a multivariate normal distribution that best fits the data

    Parameters:
    -----------
        parspace : array-like
            list of meshgrid arrays spanning parameter space
        dist : array-like 
            PDF over parameter space
    Returns:
    --------
        mean : array-like
            distribution mean
        cov : array-like
            covariance matrix		
    """
    
    # dimensionality of parameter space
    N = len(parspace)
    
    # flatten arrays
    parspace = [p.flatten() for p in parspace]
    dist = dist.flatten()
    
    # compute means
    mean = [np.sum(dist*par)/np.sum(dist) for par in parspace]
    
    # compute covariance matrix
        # empty matrix
    cov = np.zeros((N,N))
        # loop over rows
    for i in range(0,N):
            # loop over upper triangle
        for j in range(i,N):
                # compute covariance
            cov[i,j] = np.sum(dist*(parspace[i] - mean[i])*(parspace[j] - mean[j]))/np.sum(dist)
                # assign to lower triangle
            if i != j: cov[j,i] = cov[i,j]
            
    return np.array(mean), np.array(cov)

def construct_samples(a,beta,P0,Pc,P,N_samples):
    ''' This function constructs samples from a multivariate normal distribution
        fitted to the data.

        Parameters:
        -----------
        a : array-like
            Vector of 'a' parameter values.
        beta : array-like
            Vector of 'beta' parameter values.
        P0 : array-like
            Vector of 'P0' parameter values.
        Pc : array-like
            Vector of 'Pc' parameter values.
        P : array-like
            Posterior probability distribution.
        N_samples : int
            Number of samples to take.

        Returns:
        --------
        samples : array-like
            parameter samples from the multivariate normal
    '''

    # compute properties (fitting) of multivariate normal distribution
    # mean = a vector of parameter means
    # covariance = a matrix of parameter variances and correlations
    a1, beta1, P01, Pc1 = np.meshgrid(a,beta,P0,Pc,indexing='ij')
    mean, covariance = fit_mvn([a1,beta1,P01,Pc1], P)

    # 1. create samples using numpy function multivariate_normal
    samples = np.random.multivariate_normal(mean, covariance, size=N_samples)
    

    return samples

# Define iteration count
it = 0