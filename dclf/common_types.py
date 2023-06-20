import functools
import typing

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import scipy.stats as stats

__all__ = [
	'Contract',
	'ContractDesignProblem',
	'BootstrappedBinomialMixtureContractDesignProblem',
]

@dataclass
class Contract:
	t: np.array
	metadata: typing.Optional[typing.Any]


@dataclass
class ContractDesignProblem:
	f_ij: np.array  
	cost: np.array

	@property
	def sf_ij(self):
		" Survival function "
		return 1-self.f_ij.cumsum(axis=1)



class BootstrappedBinomialMixtureContractDesignProblem(ContractDesignProblem):
	def __init__(self, acc_p, cost, m, ensure_zero_cost_action=True):
		assert len(cost)==len(acc_p)
		if cost[0] != 0 and ensure_zero_cost_action:
			cost.insert(0, 0)
			acc_p.insert(0,[0.0])
		self.has_zero_cost_action = (cost[0]==0)
		self.acc_p = acc_p
		self.m = m
		f_ij = np.vstack([
			self.binomial_mixture_distribution(v if isinstance(v, Iterable) else [v], m)
			for v in acc_p
		])
		assert len(f_ij)==len(cost)
		self.mean_acc = np.array([np.mean(v) for v in acc_p])
		super().__init__(f_ij=f_ij, cost=cost)

	@staticmethod
	def binomial_mixture_distribution(p_vec, m):
		return np.vstack([
			stats.binom.pmf(np.arange(0,m+1), m, p)
			for p in p_vec
		]).mean(axis=0)

	@classmethod
	def from_series(cls, s, **kwargs):
		return cls(
			acc_p=s.tolist(),
			cost=s.index.tolist(),
			**kwargs,
		)