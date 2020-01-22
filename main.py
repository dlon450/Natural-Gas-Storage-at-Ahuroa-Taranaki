from gs_functions import *
from test_gs import *
import os

def main():

    # get the pressure and mass flow data
    directory = os.getcwd()
    tqd, qd, tPd, Pd = import_data(directory + os.sep + "Data" + os.sep + "gs_mass.txt", directory + os.sep + "Data" + os.sep + "gs_pres.txt")

    # convert q units from kg/month into kT/year
    qd = [(x*12)/1e6 for x in qd]

    # plot pressure and mass flow rate data
    f,ax1 = plt.subplots(1,1,figsize=(12,6))
    ln1 = ax1.plot(tPd,Pd,'b-',label="Pressure")
    ax1.set_xlabel("Time (year)")
    ax1.set_ylabel("Pressure (MPa)")
    ax1.set_title('Pressure and Mass Flow Rate Data')
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mass Flow Rate (kT/year)")
    ln2 = ax2.plot(tqd,qd,'r-',label="Mass Flow Rate")
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]

    plt.legend(lns,labs,loc=0)
    plt.show()
    
    #plot timestep convergence
    plot_timestep_convergence()
    # plot the benchmark
    plot_benchmark()
    t = np.arange(2009,2019.01,0.01)

    # calibrate and solve pressure ode and plot reservoir pressure
    theta_P,pm = pressure_model(tqd,qd,tPd,Pd)
    # calibrate mass flow rate and plot model
    theta_q = q_model(tqd,qd)
    # use calibrated parameters to solve leakage ode and plot leakage for 2009-2019
    Lm = leakage_model(t,pm,theta_P)

    t_predict = np.arange(2019.01,2029.01,0.01)     # prediction interval

    p_2019 = pm[-1]
    # scenario modelling - plot leakage from 2009-2029 for different capacities
    qm1,qm2,qm3,qm4 = prediction(Lm,t_predict,theta_q,theta_P,p_2019)

    # uncertainty - find posterior distribution and plot possible outcomes for 100 samples
    print("Preparing leakage scenario forecasts...")
    a,beta,P0,Pc,posterior = grid_search(theta_P,tqd,qd,tPd,Pd)
    N = 100
    samples = construct_samples(a, beta, P0, Pc, posterior, N)
    model_ensemble(samples,tqd,qd,qm1,qm2,qm3,qm4)

def pressure_model(tqd,qd,tPd,Pd):
    """
    Calibrate pressure data and solve pressure ode using calibrated parameters.

    Parameters:
    -----------
    tqd : array-like
        Time data for mass flow rate data.
    qd : array-like
        Mass flow rate data.
    tPd : array-like
        Time data for pressure data.
    Pd : array-like
        Pressure data.
            
    Returns:
    --------
    theta: array-like
        Array of pressure parameters in order: [a,beta,P0,Pc].
    P : array-like
        Model pressure values from 2009-2019.
    """

    # bounds for a,beta,P0 and Pc
    bounds = ((0, 15), (5, 20), (24, 26), (0, 0.8))
    n_iter = 10
    disp = True
    # obtain calibrated parameter values for pressure model
    print("Calibrating a, beta, P0 and Pc")
    theta = calibrate(obj_P, bounds, n_iter, disp, tqd, qd, tPd, Pd)
    print(theta)

    # obtain pressure model by using calibrated parameters 
    dt = 0.01
    t = np.arange(tPd[0], tPd[-1] + dt, dt)
    P = solve_ode(model, t, 25.16, tqd, qd, *theta)

    # find critical pressure value: P0+Pc
    cP = theta[2] + theta[3]
    # find objective function value
    s = obj_P(tqd, qd, tPd, Pd)(theta)

    # plot pressure model
    f, (ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
    ax1.plot(t, P, label="Model")
    ax1.plot(tPd, Pd, "x", label="Data")
    ax1.axhline(cP, c="g", label="Critical Pressure", lw=0.5)
    ax1.set_title("Pressure Model Objective Value = {}".format(s))
    ax1.set_xlabel("Time (Year)")
    ax1.set_ylabel("Pressure (MPa)")
    ax1.set_ylim((24.25, 26.25))
    ax1.legend(loc=0)

    # find the misfit between data and model 
    indices = np.arange(0,1025,25)  # quarterly pressure data available
    P2 = np.zeros(len(tPd))
    for i in range(len(indices)):
        P2[i] = P[indices[i]]

    misfit = Pd - P2

    # plot misfit between model and data
    ax2.plot(tPd, misfit, "x")
    ax2.axhline(0, c="g", ls="--", lw=0.5)
    ax2.set_title("Pressure Model Misfit")
    ax2.set_xlabel("Time (Year)")
    ax2.set_ylabel("Pressure Misfit (MPa)")
    plt.show()

    return theta,P

def q_model(tqd,qd):
    """
    Calibrate mass flow rate data and obtain calibrated parameters.

    Parameters:
    -----------
    tqd : array-like
        Time data for mass flow rate data.
    qd : array-like
        Mass flow rate data.
            
    Returns:
    --------
    theta_q: array-like
        Array of mass flow rate parameters in order: [A,B,C,D].
    """

    # bounds for parameters
    bounds_q = ((-5, 5), (5, 12), (0, 10), (-10, 15))
    n_iter = 100
    disp = True
    # obtain calibrated parameter values for mass flow rate model
    print("Calibrating A, B, C and D")
    theta_q = calibrate(obj_q, bounds_q, n_iter, disp, tqd, qd)
    print(theta_q)

    # plot mass flow rate model
    f,(ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
    t = np.arange(2009,2019.01,0.01)
    qm = get_q(t,*theta_q)
    ax1.plot(tqd,qd,'ro', label = 'observations')
    ax1.plot(t, qm, 'k-', label='model')
    ax1.set_ylabel("Mass flow rate (kT/year)",size=12)
    ax1.set_xlabel("Time (Year)",size=12)

    # obtain objective function value
    s_q = obj_q(tqd,qd)(theta_q)
    ax1.set_title('Mass Flow Rate Model Objective Function={:2.1e}'.format(s_q),size=12)

    # plot the misfit between fitted model and data
    misfit = np.zeros(len(tqd))
    for i in range(len(tqd)):
        q_model_val = get_q(tqd[i],*theta_q)
        misfit[i] = qd[i] - q_model_val

    # plot the misfit
    ax2.plot(tqd,misfit,'rx')
    ax2.plot(tqd,np.zeros(len(tqd)),'k--')
    ax2.set_ylabel("Mass flow rate misfit [kT/year]",size=12)
    ax2.set_xlabel("Time (Year)",size=12)
    ax2.set_title('Best fit mass flow rate model misfit', size=12)
    plt.show()

    return theta_q

def leakage_model(t,pm,theta_P):
    """
    Use the calibrated parameters to solve for leakage and plot leakage model.

    Parameters:
    -----------
    t : array-like
        Time array used to solve pressure.
    pm : array-like
        Pressure model values at times t.
    theta_P : array-like
        Array of pressure parameters in order: [a,beta,P0,Pc]
            
    Returns:
    --------
    Lm: array-like
        Array of leakage model values at times t.

    """
    # unpack best-fit parameter values
    a, beta, P0, Pc = theta_P

    # solve leakage ode using pressure parameters and model
    Lm = solve_ode(leakage_func,t,0,pm,t,a,beta,P0,Pc)

    # plot leakage
    f,ax = plt.subplots(1,1,figsize=(12,6))
    ax.set_xlabel('Time (Year)')
    ax.plot(t,Lm,'k-')
    ax.set_ylabel("Gas Leakage (kT)",size=14)
    ax.set_title('Leakage Model')
    plt.show()

    return Lm


def prediction(L_current,t_predict,theta_q,theta_p,p_2019):
    """
    Scenario modelling - solve for and plot future leakage for four scenarios

    Parameters:
    -----------
    L_current : array-like
        Array of leakage model values for 2009-2019.
    t_predict : array-like
        Time array at which to perform predictions.
    theta_q : array-like
        Array of mass flow rate parameters.
    theta_P : array-like
        Array of pressure parameters.
    p_2019 : float
        Value of pressure model at 2019 (used as IC for prediction)
            
    Returns:
    --------
    qm_predict1 : array-like
        Predicted model mass flow rate values for current capacity
    qm_predict2 : array-like
        Predicted model mass flow rate values for doubled capacity
    qm_predict3 : array-like
        Predicted model mass flow rate values for halved capacity
    qm_predict4 : array-like
        Predicted model mass flow rate values for zero capacity

    """
    t_current = np.arange(2009,2019.01,0.01)
    # find predicted mass flow rate for the future times
    qm_predict1 = get_q(t_predict,*theta_q)

    # ~~~~~ Scenario 1: q is same ~~~~~
    # find predicted pressure values for future times (using predicted q)
    Pm_predict1 = solve_ode(model, t_predict, p_2019, t_predict, qm_predict1, *theta_p)
    # solve to find model for current and future leakage
    L_predict1 = solve_ode(leakage_func,t_predict,L_current[-1],Pm_predict1,t_predict,*theta_p)
    print('Q same: final leakage = ',L_predict1[-1])

    # ~~~~~ Scenario 2: q is doubled ~~~~~
    qm_predict2 = qm_predict1*2
    Pm_predict2 = solve_ode(model, t_predict, p_2019, t_predict, qm_predict2, *theta_p)
    L_predict2 = solve_ode(leakage_func,t_predict,L_current[-1],Pm_predict2,t_predict,*theta_p)
    print('Q doubled: final leakage = ',L_predict2[-1])

    # ~~~~~ Scenario 3: q is halved ~~~~~
    qm_predict3 = qm_predict1*0.5
    Pm_predict3 = solve_ode(model, t_predict, p_2019, t_predict, qm_predict3, *theta_p)
    L_predict3 = solve_ode(leakage_func,t_predict,L_current[-1],Pm_predict3,t_predict,*theta_p)
    print('Q halved: final leakage = ',L_predict3[-1])

    # ~~~~~ Scenario 4: q is zero ~~~~~
    qm_predict4 = qm_predict1*0
    Pm_predict4 = solve_ode(model, t_predict, p_2019, t_predict, qm_predict4, *theta_p)
    L_predict4 = solve_ode(leakage_func,t_predict,L_current[-1],Pm_predict4,t_predict,*theta_p)
    print('Q zero: final leakage = ',L_predict4[-1])

    # plot future gas leakage for the four scenarios
    f,ax1 = plt.subplots(1,1,figsize=(15,6))
    ax1.plot(t_current,L_current,'k-')
    ax1.plot(t_predict,L_predict1,'r-',label='Same q')
    ax1.plot(t_predict,L_predict2,'g-',label='Double q')
    ax1.plot(t_predict,L_predict3,'b-',label='Halved q')
    ax1.plot(t_predict,L_predict4,'y-',label='Zero q')

    ax1.set_xlabel('Time (Year)')
    ax1.set_ylabel('Gas Leakage (kT)')
    ax1.set_title('Scenario modelling - future leakage')
    ax1.legend()
    plt.show()
    
    return qm_predict1,qm_predict2,qm_predict3,qm_predict4

def model_ensemble(samples,tqd,qd,qm1,qm2,qm3,qm4):
    """
    Plot possible outcomes for future leakage.

    Parameters:
    -----------
    samples : array-like
        parameter samples from the multivariate normal
    tqd : array-like
        Time data for mass flow rate data.
    qd : array-like
        Mass flow rate data.
    qm_1 : array-like
        Predicted model mass flow rate values for current capacity
    qm_2 : array-like
        Predicted model mass flow rate values for doubled capacity
    qm_3 : array-like
        Predicted model mass flow rate values for halved capacity
    qm_4 : array-like
        Predicted model mass flow rate values for zero capacity

    Returns:
    --------
    None

    """

    # get t arrays and initialise final leakage arrays
    t_current = np.arange(2009,2019.01,0.01)
    t_predict = np.arange(2019.01,2029.01,0.01)
    Lm1_final = []
    Lm2_final = []
    Lm3_final = []
    Lm4_final = []

    f,ax = plt.subplots(1,1,figsize=(14,6))
    f,ax1 = plt.subplots(1,1,figsize=(14,6))
    # for each parameter combination in samples
    for a,beta,P0, Pc in samples:
        
        # get pressure model and use this to obtain leakage model (2009-2019)
        pm_current = solve_ode(model,t_current,25.16,tqd,qd,a,beta,P0,Pc)
        Lm_current = solve_ode(leakage_func,t_current,0,pm_current,t_current,a,beta,P0,Pc)

        # find future pressure model for current capacity, solve for future leakage 
        pm_predict1 = solve_ode(model,t_predict,pm_current[-1],t_predict,qm1,a,beta,P0,Pc)
        Lm_predict1 = solve_ode(leakage_func,t_predict,Lm_current[-1],pm_predict1,t_predict,a,beta,P0,Pc)
        Lm1_final.append(Lm_predict1[-1])   # store final leakage value (at 2029)

        # find future pressure model for doubled capacity, solve for future leakage 
        pm_predict2 = solve_ode(model,t_predict,pm_current[-1],t_predict,qm2,a,beta,P0,Pc)
        Lm_predict2 = solve_ode(leakage_func,t_predict,Lm_current[-1],pm_predict2,t_predict,a,beta,P0,Pc)
        Lm2_final.append(Lm_predict2[-1])   # store final leakage value (at 2029)

        # find future pressure model for halved capacity, solve for future leakage 
        pm_predict3 = solve_ode(model,t_predict,pm_current[-1],t_predict,qm3,a,beta,P0,Pc)
        Lm_predict3 = solve_ode(leakage_func,t_predict,Lm_current[-1],pm_predict3,t_predict,a,beta,P0,Pc)
        Lm3_final.append(Lm_predict3[-1])   # store final leakage value (at 2029)

        # find future pressure model for zero capacity, solve for future leakage 
        pm_predict4 = solve_ode(model,t_predict,pm_current[-1],t_predict,qm4,a,beta,P0,Pc)
        Lm_predict4 = solve_ode(leakage_func,t_predict,Lm_current[-1],pm_predict4,t_predict,a,beta,P0,Pc)
        Lm4_final.append(Lm_predict4[-1])   # store final leakage value (at 2029)
        
        # plot the pressure samples
        ax1.plot(t_current,pm_current,lw = 0.6, alpha = 0.4)
        ax1.set_title('Plot of Pressure Samples')
        ax1.set_xlabel('Time (Year)')
        ax1.set_ylabel('Pressure (MPa)')

        # plot each model
        ax.plot(t_current,Lm_current,'k-', lw = 0.4, alpha = 0.3)
        ax.plot(t_predict,Lm_predict1,'r-',lw = 0.4, alpha = 0.3)
        ax.plot(t_predict,Lm_predict2,'g-',lw = 0.4, alpha = 0.3)
        ax.plot(t_predict,Lm_predict3,'b-',lw = 0.4, alpha = 0.3)
        ax.plot(t_predict,Lm_predict4,'y-',lw = 0.4, alpha = 0.3)
    
    # add a line to the legend
    ax.plot([],[],'k-', lw=0.5,alpha=0.6, label='Model ensemble')
    ax.plot([],[],'r-', lw=0.5,alpha=0.6, label='Same q')
    ax.plot([],[],'g-', lw=0.5,alpha=0.6, label='Double q')
    ax.plot([],[],'b-', lw=0.5,alpha=0.6, label='Halved q')
    ax.plot([],[],'y-', lw=0.5,alpha=0.6, label='Zero q')

    # get 95% Confidence Intervals for leakage model under each scenario
    CI_predict1 = np.percentile(Lm1_final, [5,95])
    CI_predict2 = np.percentile(Lm2_final, [5,95])
    CI_predict3 = np.percentile(Lm3_final, [5,95])
    CI_predict4 = np.percentile(Lm4_final, [5,95])
    print("CI for Leakage for Same q Scenario: ", CI_predict1)
    print("CI for Leakage for Double q Scenario: ", CI_predict2)
    print("CI for Leakage for Halved q Scenario: ", CI_predict3)
    print("CI for Leakage for Zero q Scenario: ", CI_predict4)

    # plot the possible outcomes
    ax.set_xlabel('Time (Year)')
    ax.set_ylabel('Gas Leakage (kT)')
    ax.set_title('Future Leakage: Scenario Forecasts with Uncertainty')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()