"""Solution."""
import numpy as np
import random
from math import log
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

import GPy

SEED = 0
# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

time = 0

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        #pass
        constant_kernel = GPy.kern.Bias(input_dim=1, variance=16) + GPy.kern.Linear(input_dim=1)
        self.f_kernel = GPy.kern.Matern52(input_dim=1, variance=0.5, lengthscale=0.5)
        self.v_kernel = GPy.kern.Matern52(input_dim=1, variance=1.414, lengthscale=0.5) + constant_kernel

        self.xs = np.zeros((0, DOMAIN.shape[0]))
        self.fs = np.zeros((0, 1))
        self.vs = np.zeros((0, 1))

        self.current_f = None
        self.current_v = None
        self.x_best = None
        self.cum_regret = 0.0

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        assert len(self.xs) == len(self.fs) == len(self.vs) and len(self.xs) > 0

        self.f = GPy.models.GPRegression(np.atleast_2d(self.xs), np.atleast_2d(self.fs), self.f_kernel, noise_var=np.square(0.15))

        self.v = GPy.models.GPRegression(np.atleast_2d(self.xs), np.atleast_2d(self.vs), self.v_kernel, noise_var=np.square(0.0001))

        fn_mean = self.f.predict(np.atleast_2d(self.xs))[0]
        xn_best_idx = np.argmax(fn_mean)
        self.x_best = self.xs[xn_best_idx]
        #raise NotImplementedError
        x_opt = self.optimize_acquisition_function()

        return x_opt

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        assert self.f is not None
        assert self.x_best is not None
        global time 
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        #raise NotImplementedError
        #if ACQUISITION_FN == "POI":
        #f_x_best = self.f.predict(np.atleast_2d(self.x_best))[0]
        #f_mean, f_var = self.f.predict(np.atleast_2d(x))
        #f_std = np.sqrt(f_var[0,0])
        #xi = 0.01
        #return norm.cdf((f_mean[0, 0] - f_x_best[0, 0] - xi) / f_std)
        #elif ACQUISITION_FN == "EI":
        #f_x_best = self.f.predict(np.atleast_2d(self.x_best))[0]
        #f_mean, f_var = self.f.predict(np.atleast_2d(x))
        #f_std = np.sqrt(f_var[0,0])
        #xi = 0.01
        #imp = (f_mean[0,0] - f_x_best[0,0] - xi)
        #if(f_std > 0):
        #    Z = imp / f_std
        #    ei = (imp * norm.cdf(Z) + f_std * norm.logpdf(Z))
        #else:
        #    Z = 0
        #    ei = 0
        #return ei
        f_mean, f_var = self.f.predict(np.atleast_2d(x))
        v_mean = self.v.predict(np.atleast_2d(x))[0]
        f_std = np.sqrt(f_var)
        cardinality = len(x)
        time+=1
        delta = random.uniform(0, 1)
        pi_t = np.square(np.pi)*np.square(time)/6*delta
        beta = (2*log(cardinality)*pi_t)
        lamb = 0.25
        if time>20:
            time = 0
        predict = f_mean[0,0] - lamb*(max(v_mean[0,0],0))
        best_upper_bound = (predict + beta*f_std[0,0])
        best_lower_bound = (predict - beta*f_std[0,0])
        if (best_upper_bound >= best_lower_bound):
            return best_upper_bound
        else: 
            return best_lower_bound

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        #raise NotImplementedError
        self.xs = np.vstack((self.xs, x))
        self.fs = np.vstack((self.fs, f))
        self.vs = np.vstack((self.vs, v))

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        #raise NotImplementedError
        fn_mean = self.f.predict(np.atleast_2d(self.xs))[0]
        xn_best_idx = np.argmax(fn_mean)
        self.x_best = self.xs[xn_best_idx]

        return self.x_best

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
