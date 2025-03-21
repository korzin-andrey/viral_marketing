import math
import scipy
import numpy as np

from .model_output import SIRModelOutput


class SIRModel():
    def __init__(self, population: int):
        self.population = population

    def __deriv(self, y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def simulate(self, beta, gamma, init_inf_frac, init_rec_frac, tmax: int):
        I0 = int(self.population*init_inf_frac)
        R0 = int(self.population*init_rec_frac)
        S0 = self.population - I0 - R0
        y0 = S0, I0, R0
        t = np.linspace(0, tmax, tmax)
        self.result = SIRModelOutput(t, *scipy.integrate.odeint(self.__deriv, y0, t,
                                     args=(self.population, beta, gamma)).T)
