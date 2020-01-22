import numpy as np
from matplotlib import pyplot as plt
import math
from gs_functions import *
from numpy.linalg import norm
import os

def plot_benchmark():
    """ Compare analytical and numerical solutions by plotting a benchmark problem.

        Parameters:
        -----------
        none

        Returns:
        --------
        none
        
    """
    directory = os.getcwd()
    tqd, qd, tPd, Pd = import_data(directory + os.sep + "Data" + os.sep + "gs_mass.txt", directory + os.sep + "Data" + os.sep + "gs_pres.txt")
    qd = [(x*12)/1e6 for x in qd]

    # choose parameter values
    a = 5
    beta = 19
    P0 = 25
    Pc = 0.5
    qc = 4          # analytical solution for constant q = 4 kT/year

    td = np.arange(2009,2019.01,0.01)
    t = td-2009
    qc_array = np.ones(len(t))*4
    # get numerical pressure solution
    P_numerical = solve_ode(model,t,25.16,t,qc_array,a,beta,P0,Pc)

    P_analytical = np.zeros(len(t))     # initialise analytical solution
    P_analytical[0]=25.16                  # set first value as 25 MPa
    # constant of integration when P-P0>= Pc
    c = np.arctanh(((25.16-P0)*math.sqrt(beta))/(math.sqrt(a)*math.sqrt(qc)))/(math.sqrt(a)*math.sqrt(beta)*math.sqrt(qc))
    
    # find analytical pressure solution
    for i in range(1,len(t)):
        # b = 0 if P - P0 < Pc
        if P_analytical[i-1]<P0+Pc:
            P_analytical[i] = a*qc*t[i] + 25.16
        else:
            # analytical solution when b = beta*(P-P0)
            P_analytical[i] = (math.sqrt(a)*math.sqrt(qc)*np.tanh((math.sqrt(a)*math.sqrt(beta)*math.sqrt(qc))*(c+t[i])))/math.sqrt(beta) + P0
    
    # plot numerical solution against analytical solution
    f,ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(td,P_numerical,'ro', label = 'Numerical solution')
    ax.plot(td, P_analytical, label='Analytical solution')
    ax.set_ylabel("Pressure (MPa)",size=14)
    ax.set_xlabel("Time (year)",size=14)
    ax.set_title('Benchmark: a={:2.1e},beta={:2.1e},P0={:2.1e} Pc={:2.1e} q0={:2.1e}'.format(a,beta,P0,Pc,qc),size=14)
    ax.legend()
    plt.show()

def test_model():
    """ Test the pressure model returns the expected solution for given inputs and situations.

    Parameters:
    -----------
    none

    Returns:
    --------
    none
        
    """
    # 1. Constant q
    t = np.array([0,1])
    q = np.array([4,4])
    theta = [1,2,3,4]
    # a. P-P0<Pc, b=0
    P1 = 6
    model_func1 = model(t,P1,t,q,*theta)
    assert(np.array_equal(model_func1,np.array([4,4])))
    
    # b. P-P0=Pc, b=beta*(P-P0)
    P2 = 7
    model_func2 = model(t,P2,t,q,*theta)
    assert(np.array_equal(model_func2,np.array([-28,-28])))

    # c. P-P0 > Pc, b=beta*(P-P0)
    P3 = 8
    model_func3 = model(t,P3,t,q,*theta)
    assert(np.array_equal(model_func3,np.array([-46,-46])))

    # 2. different q values interpolated
    tq = np.array([0,1,2,3,4])
    q = np.array([1,2,3,2,7])
    t = np.array([2,3,4])
    model_func4 = model(t,P1,tq,q,*theta)
    assert(np.array_equal(model_func4,np.array([3,2,7])))


def plot_timestep_convergence():
    """ Compare analytical and numerical solutions.

        Parameters:
        -----------
        none

        Returns:
        --------
        none
        
    """
    # array for number of improved euler steps to use
    steps=np.arange(30,100,1)
    # initialise array for step size - h
    h_vals = np.zeros(len(steps))
    # use time = 10 (years from 2009) for error convergence test
    time = 10
    # h_vals = np.linspace(0.01,0.1,100)
    p_estimate=np.zeros(len(h_vals))

    # find corresponding step sizes for each number of steps 
    for i in range(len(steps)):
        h_vals[i] = time/steps[i]

    # solve pressure ode for specific theta and constant q 
    for j in range(len(h_vals)):
        t = np.arange(0,10.1+h_vals[j],h_vals[j])
        qc_array = np.ones(len(t))*4
        p_vals = solve_ode(model, t,25,t,qc_array,1,0.3,23,0.5)
        p_estimate[j] = p_vals[-1]

    # plot the estimated pressure value vs 1/step size value
    f1,ax3 = plt.subplots(nrows=1,ncols=1)
    ax3.plot(1./np.array(h_vals),p_estimate,'bs')
    ax3.set_ylabel('Pressure (MPa)')
    ax3.set_xlabel('1/h')
    ax3.set_title('timestep convergence')
    plt.show()

def test_solve_ode():
    """ Test improved euler implementation in solve_ode.

    Parameters:
    -----------
    none

    Returns:
    --------
    none
        
    """

    def dydt(t,y,a,b):
        """ Example ODE function to be solved using improved euler method.

        Parameters:
        -----------
        t: independent variable
        y: dependent variable
        a,b : random parameter values (constant)

        Returns:
        --------
        dydt : return value of the ode function for given t and y.
        
        """
        dydt = -y*a + b*t
        return dydt

    # initialise independent variable array
    x = np.array([0, 0.5])
    # obtain numerical solution
    y = solve_ode(dydt, x, 1, 1, 2)
    y_soln = [1, 0.875] # expected solution
    assert norm(y - y_soln) < 1.e-6

def test_get_q():
    """ Test mass flow rate ode function returns correct output

    Parameters:
    -----------
    none

    Returns:
    --------
    none
    """

    ans = get_q(1,2,3,4,5)
    expected_ans = 2+(3*np.sin((4*1+5)))
    assert abs(ans-expected_ans) < 1.e-6

def test_leakage_func():
    """ Test leakage_func function returns the right output for different inputs/scenarios

    Parameters:
    -----------
    none

    Returns:
    --------
    none
    """

    # 1. At one point in time (one pressure value)
    t = 1
    theta = [1,2,3,4]
    L=1
    # a. P-P0<Pc, b=0
    P1 = np.array([6,6])
    ans1 = leakage_func(t,L,P1,np.array([0,1]),*theta)
    assert abs(ans1 - 0) < 1.e-6
    
    # b. P-P0=Pc, b=beta*(P-P0)
    P2 = np.array([7,7])
    ans2 = leakage_func(t,L,P2,np.array([0,1]),*theta)
    assert abs(ans2 - 32) < 1.e-6

    # c. P-P0 > Pc, b=beta*(P-P0)
    P3 = np.array([8,8])
    ans3 = leakage_func(t,L,P3,np.array([0,1]),*theta)
    assert abs(ans3 - 50) < 1.e-6

    # 2. different P values interpolated
    tp = np.array([0,1,2,3,4])
    p = np.array([5,10,12,15,10])
    t = 3
    ans4 = leakage_func(t,L,p,tp,*theta)
    assert abs(ans4 - 288) < 1.e-6

