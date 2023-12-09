import collections
import functools
import itertools
import os
import time

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
})

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

import dclf

from tqdm.auto import tqdm

tracker = dclf.ParamTracker()
param = tracker.store
