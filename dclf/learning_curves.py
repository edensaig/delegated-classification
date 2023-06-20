import numpy as np
import scipy.optimize

__all__ = [
    'Pow3',
]

class Pow3:
    def __init__(self, params):
        self.params = params
        assert 0<=params[0]
        # assert 0<=params[0]<=1
        assert 0<=params[1]
        assert 0<=params[2]
        
    @staticmethod
    def f(x,a,b,c):
        return a-b*x**(-c)
        
    def __call__(self,x):
        return self.f(x,*self.params)
    
    @classmethod
    def from_data(cls,x,y,**curve_fit_params):
        if 'bounds' not in curve_fit_params:
            curve_fit_params['bounds'] = (0,[1,5,5])
        popt, pcov = scipy.optimize.curve_fit(
            f=cls.f,
            xdata=x,
            ydata=y,
            **curve_fit_params,
        )
        return cls(popt)