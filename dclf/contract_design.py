import numpy as np
import pyomo.environ as pe

import itertools

from .common_types import *
from .agent_simulation import *

__all__ = [
    'MinBudgetContract',
    'MinBudgetStatisticalContract',
    'MinBudgetLocalContract',
    'FixedBudgetThresholdContract',
    'FullEnumerationContract',
    'MinPayContract',
]

opt = pe.SolverFactory('glpk')
opt.options ={'tmlim':10}

EPS = 1e-6

def extract_variable(model, var_name):
    return np.array([v.value for v in list(getattr(model,var_name).values())])

class LPContract:
    @classmethod
    def design(cls, design_problem, target_action=-1, **kwargs):
        assert len(design_problem.f_ij)==len(design_problem.cost)
        assert (np.abs(design_problem.f_ij.sum(axis=1)-1)<=1e-6).all()
        selected_ind = target_action+1 if target_action!=-1 else len(design_problem.f_ij)
        f_ij = design_problem.f_ij[:selected_ind]
        cost = design_problem.cost[:selected_ind]
        return cls.solve(f_ij, cost, **kwargs)
        

class MinBudgetContract(LPContract):
    @classmethod
    def solve(cls, f_ij, cost, **kwargs):
        methods = {
            'primal': cls.primal,
            'dual': cls.dual,
        }
        results = {m: f(f_ij,cost) for m,f in methods.items()}
        return {
            f'{m}_{k}': v
            for m,dct in results.items()
            for k,v in dct.items()
        } | {
            't': results['primal']['t']
        }

    @staticmethod
    def primal(f_ij, cost):
        n,m = f_ij.shape
        j_vec = np.arange(m)
        model = pe.ConcreteModel()
        model.t = pe.Var(j_vec, within=pe.NonNegativeReals)
        model.B = pe.Var(within=pe.NonNegativeReals)
        model.ic = pe.Constraint(
            range(n-1),
            rule=lambda model, i: sum((f_ij[n-1,j]-f_ij[i,j])*model.t[j] for j in j_vec) >= cost[n-1]-cost[i],
        )
        model.budget = pe.Constraint(
            j_vec,
            rule=lambda model, j: model.t[j]<=model.B,
        )
        model.obj = pe.Objective(
            expr=model.B,
            sense=pe.minimize,
        )
        result = opt.solve(model)
        return {
            'model': model,
            'result': result,
            't': extract_variable(model, 't'),
            'objective': result['Problem'][0]['Lower bound'],
        }

    @staticmethod
    def dual(f_ij, cost):
        n,m = f_ij.shape
        i_vec = np.arange(n-1)
        j_vec = np.arange(m)
        model = pe.ConcreteModel()
        model.lambd = pe.Var(i_vec, within=pe.NonNegativeReals)
        model.mu = pe.Var(j_vec, within=pe.NonNegativeReals)
        model.f_constraint = pe.Constraint(
            j_vec,
            rule=lambda model, j: sum((f_ij[n-1,j]-f_ij[i,j])*model.lambd[i] for i in i_vec) <= model.mu[j],
        )
        model.mu_sum_constraint = pe.Constraint(
            expr=sum(model.mu[j] for j in j_vec) <= 1,
        )
        model.obj = pe.Objective(
            expr=sum((cost[n-1]-cost[i])*model.lambd[i] for i in i_vec),
            sense=pe.maximize,
        )
        result = opt.solve(model)
        return {
            'model': model,
            'result': result,
            'lambda': extract_variable(model, 'lambd'),
            'mu': extract_variable(model, 'mu'),
            'objective': result['Problem'][0]['Lower bound'],
        }


class MinBudgetStatisticalContract(LPContract):
    @classmethod
    def solve(cls, f_ij, cost, binary_domain, monotone):
        phi_domain = pe.Binary if binary_domain else pe.PercentFraction
        return cls.statistical_primal(f_ij,cost,phi_domain=phi_domain,monotone=monotone)

    @staticmethod
    def statistical_primal(f_ij, cost, phi_domain, monotone):
        n,m = f_ij.shape
        j_vec = np.arange(m)
        model = pe.ConcreteModel()
        model.phi = pe.Var(j_vec, within=phi_domain) # pe.PercentFraction or pe.Binary
        model.beta = pe.Var(within=pe.NonNegativeReals)
        model.ic = pe.Constraint(
            range(n-1),
            rule=lambda model, i: (
                sum(
                    f_ij[n-1,j]*(1-model.phi[j])
                    + f_ij[i,j]*model.phi[j]
                    for j in j_vec
                ) <= 1-(cost[n-1]-cost[i])*model.beta
            ),
        )
        if monotone:
            model.monotonicity = pe.Constraint(
                range(m-1),
                rule=lambda model,j: (
                    model.phi[j] <= model.phi[j+1]
                )
            )
        model.obj = pe.Objective(
            expr=model.beta,
            sense=pe.maximize,
        )
        result = opt.solve(model)
        if model.beta.value<=1e-12:
            raise ZeroDivisionError('Statistical LP is unbounded')
        return {
            'model': model,
            'result': result,
            'phi': extract_variable(model, 'phi'),
            't': extract_variable(model, 'phi')/model.beta.value,
            'objective': result['Problem'][0]['Lower bound'],
            'objective_inv': 1/result['Problem'][0]['Lower bound'],
        }


class MinBudgetLocalContract(LPContract):
    @classmethod
    def solve(cls, f_ij, cost, compare_to_all=True, eps=EPS):
        n = len(f_ij)
        if compare_to_all:
            compare_ind = range(n-1)
        else:
            compare_ind = [n-2]

        min_B = None
        opt_t = None
        binding_action = None
        for i in compare_ind:
            d_cost = cost[-1]-cost[i]
            tv = cls.tv_distance(f_ij[-1], f_ij[i])
            B = d_cost/tv
            t = (f_ij[-1]>=f_ij[i])*B
            ic_constraint = f_ij@t-cost <= f_ij[-1]@t-cost[-1] + eps
            is_feasible = ic_constraint.all()
            if is_feasible and (min_B is None or B<=min_B):
                min_B = B
                opt_t = t
                binding_action = i

        if min_B is None:
            raise RuntimeError('Local min budget contract design failed to find a solution')

        return {
            't': opt_t,
            'budget': min_B,
            'binding_action': binding_action,
        }

    @staticmethod
    def tv_distance(p,q):
        return np.abs(p-q).sum()/2



class FixedBudgetThresholdContract:
    @classmethod
    def design(cls, design_problem, budget):
        sf_ij = design_problem.sf_ij
        cost = design_problem.cost
        selected_action = (sf_ij*budget-np.expand_dims(cost,axis=1)).argmax(axis=0)
        # selected_i = selected_action.argmax()
        selected_j = len(selected_action)-1 - selected_action[::-1].argmax()
        t = (np.arange(sf_ij.shape[1])>=selected_j)*budget
        return {
            't': t,
            'selected_action_alternatives': selected_action,
            'threshold_point': selected_j,
        }


class FullEnumerationContract(LPContract):
    @classmethod
    def solve(cls, f_ij, cost, monotone):
        n,m = f_ij.shape
        dc = cost[-1] - np.array(cost[:-1])
        df = f_ij[-1] - f_ij[:-1]
        best_ratio = None
        best_phi = None
        if monotone:
            it = ((0,)*(j0-1) + (1,)*(m-(j0-1)) for j0 in range(1,m+1))
        else:
            it = itertools.product((0,1), repeat=m)
        for phi_tup in it:
            phi = np.array(phi_tup)
            ratio = (df@phi)/dc
            min_ratio = ratio.min()
            if best_ratio is None or min_ratio>=best_ratio:
                best_ratio = min_ratio
                best_phi = phi
        if best_ratio<=0:
            raise RuntimeError('beta is not positive')
        return {
            't': best_phi/best_ratio,
            'best_phi': best_phi,
            'best_ratio': best_ratio,
        }


class MinPayContract(LPContract):
    @staticmethod
    def solve(f_ij, cost):
        n,m = f_ij.shape
        j_vec = np.arange(m)
        model = pe.ConcreteModel()
        model.t = pe.Var(j_vec, within=pe.NonNegativeReals)
        model.ic = pe.Constraint(
            range(n-1),
            rule=lambda model, i: sum((f_ij[n-1,j]-f_ij[i,j])*model.t[j] for j in j_vec) >= cost[n-1]-cost[i],
        )
        model.obj = pe.Objective(
            expr=sum(f_ij[n-1,j]*model.t[j] for j in j_vec),
            sense=pe.minimize,
        )
        result = opt.solve(model)
        return {
            'model': model,
            'result': result,
            't': extract_variable(model, 't'),
            'objective': result['Problem'][0]['Lower bound'],
        }
