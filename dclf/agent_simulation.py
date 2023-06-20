from dataclasses import dataclass

import numpy as np

from .common_types import *

__all__ = [
	'get_agent_response'
]


@dataclass
class RationalChoice:
	utility: np.array
	selected_action: int
	selected_action_utility: float
	selected_action_cost: float
	expected_pay: float


def get_agent_response(design_problem, contract, eps=1e-6):
	if type(contract) is Contract:
		t = contract.t
	elif type(contract) is dict:
		t = contract['t']
	else:
		t = contract
	u = design_problem.f_ij@t - design_problem.cost
	i, u_i = selected_action(u, eps)
	return RationalChoice(
		utility=u,
		selected_action=i,
		selected_action_utility=u_i,
		selected_action_cost=design_problem.cost[i],
		expected_pay=u_i+design_problem.cost[i],
	)

def selected_action(u, eps):
    i = np.where(u.max()-u<=eps)[0].max()
    return i,u[i]