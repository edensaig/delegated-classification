import base64
import io
import json
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import scipy.stats

__all__ = [
    'DownloadableIO',
    'save_fig',
    'download_fig',
    'save_and_download_fig',
    'background_line_style',
    'ParamTracker',
    'plot_curve_with_band',
    'df_plot_curve_with_band',
    'HandlerDashedLines',
    'describe_series_of_samples',
    'is_monotone',
    'is_all_or_nothing',
    'is_threshold',
    'mnist_openmlid',
    'lcdb_learner_names',
]


# Figures

class DownloadableIO(io.BytesIO):
    def __init__(self, filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        
    def _repr_html_(self):
        buf = self.getbuffer()
        buf_enc = base64.b64encode(buf).decode('ascii')
        return f'<a href="data:text/plain;base64,{buf_enc}" download="{self.filename}">Download {self.filename}</a>'

def save_fig(fig, fname, **savefig_kwargs):
    return fig.savefig(
        fname=fname,
        bbox_inches='tight',
        pad_inches=0,
         **savefig_kwargs,
    )

def download_fig(fig, fname, **savefig_kwargs):
    fig_out = DownloadableIO(filename=os.path.basename(fname))
    save_fig(
        fig,
        fig_out,
        format=fname.split('.')[-1],
        **savefig_kwargs,
    )
    display(fig_out)

def save_and_download_fig(fig, fname, **savefig_kwargs):
    save_fig(fig, fname, **savefig_kwargs)
    print(f'Figure saved as {fname}')
    download_fig(fig, fname, **savefig_kwargs)

background_line_style = {
    'color': 'lightgray',
    'linestyle': ':',
    'zorder': -100,
}


# Param tracker

class NpEncoder(json.JSONEncoder):
    # https://stackoverflow.com/questions/50916422
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class ParamTracker:
    def __init__(self):
        self.data = {}

    def store(self, x, param_name):
        if param_name in self.data:
            assert x==self.data[param_name]
        self.data[param_name] = x
        return x

    def get(self, param_name):
        return self.data[param_name]

    def save(self, fname):
        json.dump(self.data, open(fname,'w'), cls=NpEncoder)


# Plots

def plot_curve_with_band(x, mean, lb, ub, ax=None, band_alpha=0.2, **line_kwargs):
    if ax is None:
        fig,ax=plt.subplots()

    l = ax.plot(
        x,
        mean,
        **line_kwargs,
    )
    ax.fill_between(
        x,
        lb,
        ub,
        alpha=band_alpha,
        color=l[0].get_color(),
        zorder=-1,
    )
    return ax
    
def df_plot_curve_with_band(df, mean, lb, ub, ax=None, **kwargs):
    return plot_curve_with_band(
        ax=ax,
        x=df.index.to_numpy(),
        mean=df[mean].to_numpy(),
        lb=df[lb].to_numpy(),
        ub=df[ub].to_numpy(),
        **kwargs
    )

class HandlerDashedLines(matplotlib.legend_handler.HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.
    https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = matplotlib.lines.Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines

# Statistical analysis

CONFIDENCE_LEVEL = 0.95

ztest_confidence_interval = lambda mu, std, count: scipy.stats.norm.interval(
    CONFIDENCE_LEVEL,
    loc=mu,
    scale=std/np.sqrt(count),
)

series_confidence_interval = lambda s: confidence_interval(
    s.mean(),
    s.std(),
    s.count(),
)

def describe_series_of_samples(lst):
    arr = np.array(lst)
    dct = {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'count': len(arr),
    }
    dct['std_lb'] = dct['mean']-2*dct['std']
    dct['std_ub'] = dct['mean']+2*dct['std']
    dct['ztest_lb'], dct['ztest_ub'] = ztest_confidence_interval(
        mu=dct['mean'],
        std=dct['std'],
        count=dct['count'],
    )
    quantiles = [0.01,0.05,0.25,0.5,0.75,0.95,0.99]
    for q in quantiles:
        dct[f'p{q*100:02g}'] = np.quantile(lst,q)
    return pd.Series(dct)


# Learning curve analysis

EPS = 1e-6

is_monotone = lambda t, eps=EPS: (np.diff(t)>=-eps).all()

def is_all_or_nothing(t, eps=EPS):
    close_to_max = t>t.max()-eps
    close_to_min = t<t.min()+eps
    return (close_to_min | close_to_max).all()

def is_threshold(t, eps=EPS):
    close_to_max = t>t.max()-eps
    close_to_min = t<t.min()+eps
    two_distinct_values = (close_to_min | close_to_max).all()
    no_mix = np.arange(len(t))[close_to_max].min() > np.arange(len(t))[close_to_min].max()
    return two_distinct_values and no_mix 


# LCDB

mnist_openmlid = 554 # https://www.openml.org/search?type=data&sort=runs&id=554
lcdb_learner_names = {
    'sklearn.neural_network.MLPClassifier': 'MLP',
    'sklearn.ensemble.GradientBoostingClassifier': 'GBDT',
    'sklearn.linear_model.LogisticRegression': 'Logistic',
    'sklearn.linear_model.Perceptron': 'Perceptron',
    'SVC_linear': 'Linear SVM',
    'SVC_poly': 'Poly SVM',
    'SVC_rbf': 'RBF SVM',
    'sklearn.neighbors.KNeighborsClassifier': 'KNN',
}
