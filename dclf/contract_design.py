import itertools
import time

import numpy as np
import pyomo.environ as pe

from .common_types import *
from .agent_simulation import *

__all__ = [
    'MinBudgetContract',
    'MinBudgetStatisticalContract',
    'MinBudgetSingleBindingActionContract',
    'MinBudgetHybridContract',
    'FixedBudgetThresholdContract',
    'FullEnumerationContract',
    'MinPayContract',
]

opt = pe.SolverFactory('glpk')
opt.options ={'tmlim':10}

EPS = 1e-9

def extract_variable(model, var_name):
    return np.array([v.value for v in list(getattr(model,var_name).values())])

def alternative_actions(f_ij, target):
    mean_acc = f_ij@np.arange(f_ij.shape[1])
    n = len(f_ij)
    return [i for i in range(n) if i!=target and mean_acc[i]<mean_acc[target]]

class LPContract:
    @classmethod
    def design(cls, design_problem, target_action=-1, **kwargs):
        assert len(design_problem.f_ij)==len(design_problem.cost)
        assert (np.abs(design_problem.f_ij.sum(axis=1)-1)<=1e-6).all()
        target_action %= len(design_problem.f_ij)
        # n = len(design_problem.f_ij)
        # selected_ind = target_action+1 if target_action!=-1 else n
        # f_ij = design_problem.f_ij[:selected_ind]
        # cost = design_problem.cost[:selected_ind]
        start_time = time.time()
        # result = cls.solve(f_ij, cost, **kwargs)
        result = cls.solve(
            f_ij=design_problem.f_ij, 
            cost=design_problem.cost, 
            target=target_action,
            **kwargs,
        )
        end_time = time.time()
        result['wall_time'] = end_time-start_time
        return result
        

class MinBudgetContract(LPContract):
    @classmethod
    def solve(cls, f_ij, cost, target):
        methods = {
            'primal': cls.primal,
            'dual': cls.dual,
        }
        results = {m: f(f_ij,cost,target) for m,f in methods.items()}
        return {
            f'{m}_{k}': v
            for m,dct in results.items()
            for k,v in dct.items()
        } | {
            't': results['primal']['t']
        }

    @staticmethod
    def primal(f_ij, cost, target):
        n,m = f_ij.shape
        i_vec = alternative_actions(f_ij, target)
        j_vec = np.arange(m)
        model = pe.ConcreteModel()
        model.t = pe.Var(j_vec, within=pe.NonNegativeReals)
        model.B = pe.Var(within=pe.NonNegativeReals)
        model.ic = pe.Constraint(
            i_vec,
            rule=lambda model, i: sum((f_ij[target,j]-f_ij[i,j])*model.t[j] for j in j_vec) >= cost[target]-cost[i],
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
    def dual(f_ij, cost, target):
        n,m = f_ij.shape
        i_vec = alternative_actions(f_ij, target)
        j_vec = np.arange(m)
        model = pe.ConcreteModel()
        model.lambd = pe.Var(i_vec, within=pe.NonNegativeReals)
        model.mu = pe.Var(j_vec, within=pe.NonNegativeReals)
        model.f_constraint = pe.Constraint(
            j_vec,
            rule=lambda model, j: sum((f_ij[target,j]-f_ij[i,j])*model.lambd[i] for i in i_vec) <= model.mu[j],
        )
        model.mu_sum_constraint = pe.Constraint(
            expr=sum(model.mu[j] for j in j_vec) <= 1,
        )
        model.obj = pe.Objective(
            expr=sum((cost[target]-cost[i])*model.lambd[i] for i in i_vec),
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
    def solve(cls, f_ij, cost, target, binary_domain, monotone):
        phi_domain = pe.Binary if binary_domain else pe.PercentFraction
        return cls.statistical_primal(f_ij, cost, target, phi_domain=phi_domain, monotone=monotone)

    @staticmethod
    def statistical_primal(f_ij, cost, target, phi_domain, monotone):
        n,m = f_ij.shape
        i_vec = alternative_actions(f_ij, target)
        j_vec = np.arange(m)
        model = pe.ConcreteModel()
        model.phi = pe.Var(j_vec, within=phi_domain) # pe.PercentFraction or pe.Binary
        model.beta = pe.Var(within=pe.NonNegativeReals)
        model.ic = pe.Constraint(
            i_vec,
            rule=lambda model, i: (
                sum(
                    f_ij[target,j]*(1-model.phi[j])
                    + f_ij[i,j]*model.phi[j]
                    for j in j_vec
                ) <= 1-(cost[target]-cost[i])*model.beta
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


class MinBudgetSingleBindingActionContract(LPContract):
    @classmethod
    def solve(cls, f_ij, cost, target, compare_to_all=True, eps=EPS):
        n = len(f_ij)
        i_vec = [target-1] + [i for i in range(n) if i not in [target, target-1]]

        for i in i_vec:
            d_cost = cost[target]-cost[i]
            if d_cost<0:
                continue
            tv = cls.tv_distance(f_ij[target], f_ij[i])
            B = d_cost/tv
            assert B>=0
            t = (f_ij[target]>=f_ij[i])*B
            ic_constraint = f_ij@t-cost <= f_ij[target]@t-cost[target] + eps
            is_feasible = ic_constraint.all()
            if is_feasible:
                return {
                    't': t,
                    'budget': B,
                    'binding_action': i,
                }

        raise RuntimeError('Local min budget contract design failed to find a solution')

    @staticmethod
    def tv_distance(p,q):
        return np.abs(p-q).sum()/2


class MinBudgetHybridContract(LPContract):
    @classmethod
    def solve(cls, f_ij, cost, target, local_solver_kwargs={}, lp_solver_kwargs={}):
        try:
            result = MinBudgetSingleBindingActionContract.solve(
                f_ij=f_ij, 
                cost=cost,
                target=target,
                **local_solver_kwargs,
            )
            result['solver'] = 'sba'
        except RuntimeError:
            result = MinBudgetContract.solve(
                f_ij=f_ij, 
                cost=cost,
                target=target,
                **lp_solver_kwargs,
            )
            result['solver'] = 'lp'
        return result


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
    def solve(cls, f_ij, cost, target, monotone):
        n,m = f_ij.shape
        dc = cost[target] - cost
        df = f_ij[target] - f_ij
        pred = alternative_actions(f_ij, target)
        dc = dc[pred]
        df = df[pred]
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
    def solve(f_ij, cost, target):
        n,m = f_ij.shape
        i_vec = [i for i in range(n) if i!=target]
        j_vec = np.arange(m)
        model = pe.ConcreteModel()
        model.t = pe.Var(j_vec, within=pe.NonNegativeReals)
        model.ic = pe.Constraint(
            i_vec,
            rule=lambda model, i: sum((f_ij[target,j]-f_ij[i,j])*model.t[j] for j in j_vec) >= cost[target]-cost[i],
        )
        model.obj = pe.Objective(
            expr=sum(f_ij[target,j]*model.t[j] for j in j_vec),
            sense=pe.minimize,
        )
        result = opt.solve(model)
        return {
            'model': model,
            'result': result,
            't': extract_variable(model, 't'),
            'objective': result['Problem'][0]['Lower bound'],
        }
