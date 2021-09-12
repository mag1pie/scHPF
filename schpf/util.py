#!/usr/bin/env python

from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.io import mmread
import igraph as ig

import schpf
import scipy.sparse as sp, csr_matrix
from scipy.stats import hypergeom
from scipy.spatial.distance import squareform
#from sklearn.decomposition.nmf import non_negative_factorization
#from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import sparsefuncs
import sklearn

from fastcluster import linkage
from scipy.cluster.hierarchy import leaves_list

import umap
import loompy
from tqdm import tqdm

import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] =42
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

import plotly.express as px
import plotly.io as pio


def mean_cellscore_fraction(cell_scores, ntop_factors=1):
    """ Get number of cells with a percentage of their total scores
    on a small number of factors

    Parameters
    ----------
    cell_scores : ndarray
        (ncells, nfactors) array of cell scores
    ntop_factors : int, optional (Default: 1)
        number of factors that can count towards domance

    Returns
    -------
    mean_cellscore_fraction : float
        The mean fraction of cells' scores that are contained within
        their top `ntop_factors` highest scoring factors

    """
    totals = np.sum(cell_scores, axis=1)
    ntop_scores = np.sort(cell_scores,axis=1)[:, -ntop_factors:]
    domsum = np.sum(ntop_scores, axis=1)
    domfrac = domsum/totals
    return np.mean(domfrac)


def mean_cellscore_fraction_list(cell_scores):
    """ Make a list of the mean dominant fraction at all possible numbers
        of ntop_factors
    """
    return [mean_cellscore_fraction(cell_scores, i+1)
                for i in range(cell_scores.shape[1])]


def max_pairwise(gene_scores, ntop=200, second_greatest=False):
    """ Get the maximum pairwise overlap of top genes

    Parameters
    ----------
    gene_scores : ndarray
        (ngenes, nfactors) array of gene scores
    ntop : int (optional, default 200)
        Number of top genes to consider in each factor
    second_greatest : bool, optional
        Return the second greatest pairwise overlap of top genes

    Returns
    -------
    max_pairwise : int
        The maximum pairwise overlap of the `ntop` highest scoring genes in
        each factors
    p : float
        Hypergeometric p value of max_pairwise, where the number of genes is
        the population size, `ntop` is the number of potential successes and
        the number of draws, and max_pairwise is the number of successes.
    """
    tops = np.argsort(gene_scores, axis=0)[-ntop:]
    max_pairwise, last_max = 0, 0
    for i in range(tops.shape[1]):
        for j in range(tops.shape[1]):
            if i >= j:
                continue
            overlap = len(np.intersect1d(tops[:,i], tops[:,j]))
            if overlap > max_pairwise:
                last_max = max_pairwise
                max_pairwise = overlap
            elif overlap > last_max:
                last_max = overlap

    overlap = last_max if second_greatest else max_pairwise
    p = hypergeom.pmf(k=overlap, M=gene_scores.shape[0],
                N=ntop, n=ntop) \
        + hypergeom.sf(k=overlap, M=gene_scores.shape[0],
                N=ntop, n=ntop)
    Overlap = namedtuple('Overlap', ['overlap', 'p'])
    return Overlap(overlap, p)


def max_pairwise_table(gene_scores, ntop_list=[50,100,150,200,250,300]):
    """ Get the maximum pairwise overlap at

    Parameters
    ----------
    gene_scores : ndarray
        (ngenes, nfactors) array of gene scores
    ntop_list : list, optional
        List of values of ntop to evaluate

    Returns
    -------
    df : DataFrame
    """
    max_overlap, p_max, max2_overlap, p_max2 = [],[],[],[]
    for ntop in ntop_list:
        o = max_pairwise(gene_scores, ntop, False)
        max_overlap.append( o.overlap )
        p_max.append( o.p )

        o2 = max_pairwise(gene_scores, ntop, True)
        max2_overlap.append( o2.overlap )
        p_max2.append( o2.p )
    df = pd.DataFrame({'ntop' : ntop_list, 'max_overlap' : max_overlap,
        'p_max' : p_max, 'max2_overlap' : max2_overlap, 'p_max2' : p_max2})
    return df


def split_coo_rows(X, split_indices):
    """Split a coo matrix into two

    Parameters
    ----------
    X : coo_matrix
        Matrix to split into two by row
    split_indices : ndarray
        Indices to use for the split.

    Returns
    -------
    a : coo_matrix
        rows from X specified in split_indices
    b : coo_matrix
        rows from X *not* specified in split_indices

    """
    a_indices = split_indices
    b_indices = np.setdiff1d(np.arange(X.shape[0]), split_indices)

    X_csr = X.tocsr()
    a = X_csr[a_indices, :].tocoo()
    b = X_csr[b_indices, :].tocoo()
    return a, b


def collapse_coo_rows(coo):
    """Collapse the empty rows of a coo_matrix

    Parameters
    ----------
    coo : coo_matrix
        Input coo_matrix which may have empty rows


    Returns
    -------
    collapsed_coo : coo_matrix
        coo with row indices adjusted to removed empty rows
    collapsed_indices : ndarray
        Indices of the returned rows in the original input matrix
    """
    nz_idx = np.where(coo.getnnz(1) > 0)[0]
    return coo.tocsr()[nz_idx].tocoo(), nz_idx


def insert_coo_rows(a, b, b_indices):
    """Insert rows from b into a at specified row indeces

    Parameters
    ----------
    a : sparse matrix
    b : sparse matrix
    b_indices : ndarray
        Indices in final matrix where b's rows should be. np.max(`b_indices`)
        must be a valid row index in the merged matrix with shape[0] =
        a.shape[0] + b.shape[0].  Must me ordered and unique.

    Returns
    -------
    ab :
        coo_matrix with rows re-indexed to have rows from b
    """
    # check arguments
    if a.shape[1] != b.shape[1]:
        msg = 'a.shape[1] must equal b.shape[1], received a with shape'
        msg += ' {} and b with shape {}'.format(a.shape, b.shape)
        raise ValueError(msg)
    if np.max(b_indices) >= a.shape[0] + b.shape[0]:
        msg = 'Invalid row indices {} for array with '.format(b_indices)
        msg += 'a.shape[0] + b.shape[0] = {} '.format(a.shape[0])
        msg += '+ {} = {}'.format(b.shape[0], a.shape[0]+b.shape[0])
        raise ValueError(msg)
    if not np.all(np.diff(b_indices) > 0):
        msg = '`b_indices` must be ordered without repeats. Received '
        msg += '{}'.format(b_indices)
        raise ValueError(msg)

    out_shape = (a.shape[0] + b.shape[0], a.shape[1])
    a = a.tocsr()
    b = b.tocsr()

    a_row, b_row = 0, 0
    data, indices, indptr = [], [], [0]
    for ab_row in range(out_shape[0]):
        if b_row < len(b_indices) and ab_row == b_indices[b_row]:
            my_row = b[b_row, :]
            b_row += 1
        else:
            my_row = a[a_row, :]
            a_row += 1
        data.append(my_row.data)
        indices.append(my_row.indices)
        indptr.append(indptr[-1] + my_row.indptr[1])

    ab = csr_matrix(
            (np.hstack(data), np.hstack(indices), np.array(indptr)),
            out_shape).tocoo()
    return ab


def minibatch_ix_generator(ncells, batchsize):
    assert ncells >= batchsize # allow equalitiy for testing
    ixs = np.arange(ncells)
    np.random.shuffle(ixs)
    start = 0
    while True:
        stop = start + batchsize
        if stop > ncells:
            stop = stop % ncells
            res = np.hstack([ixs[start:ncells], ixs[0:stop]])
        else:
            res = ixs[start:stop]
        start = stop % ncells # need mod for case where ncells=batchsize
        yield res

        
from sklearn import metrics

def get_param_dfs(model):
    eta_shp = pd.Series(np.ravel(model.eta.vi_shape), name=model.name)
    eta_rte = pd.Series(np.ravel(model.eta.vi_rate), name=model.name)
    beta_shp = pd.DataFrame(model.beta.vi_shape.T)
    beta_shp.index = model.name + ':' + (beta_shp.index + 1).astype(str)
    beta_rte = pd.DataFrame(model.beta.vi_rate.T, index=beta_shp.index)
    return eta_shp, eta_rte, beta_shp, beta_rte


def get_spectra(models):
    eta_shp, eta_rte, beta_shp, beta_rte = zip(*[get_param_dfs(m) for m in models])
    return pd.concat(eta_shp, axis=1).T, pd.concat(eta_rte,axis=1).T, pd.concat(beta_shp), pd.concat(beta_rte)


def get_genescore_spectra(models):
    gene_scores = []
    for m in models:
        gs = pd.DataFrame(m.gene_score().T)
        gs.index = m.name + ':' + (gs.index + 1).astype(str)
        gene_scores.append(gs)
    return pd.concat(gene_scores)


def get_spectra_order(factor_dists, cluster_labels):
    spectra_order = []
    for cl in sorted(set(cluster_labels)):
        cl_filter = cluster_labels==cl
        if cl_filter.sum() > 1:
            cl_dist = squareform(factor_dists[cl_filter, :][:, cl_filter])
            cl_dist[cl_dist < 0] = 0 #Rarely get floating point arithmetic issues
            cl_link = linkage(cl_dist, 'average')
            cl_leaves_order = leaves_list(cl_link)
            spectra_order += list(np.where(cl_filter)[0][cl_leaves_order])
        else:
            ## Corner case where a component only has one element
            spectra_order += list(np.where(cl_filter)[0])
    return spectra_order


# returns the median correlation distance between factor of the row and its n nearest neighbors
def get_local_density(factor_dists, n_neighbors, factor_labels):
    partitioning_order  = np.argpartition(factor_dists, n_neighbors+1)[:, :n_neighbors+1] 
    # finds the n nearest neighbors for every factor across all N_PER*nmodels models (self correlation dist is 0)
    # ndims(partitioning_order) = total n of factors * (n_neighbors+1)
    distance_to_nearest_neighbors = factor_dists[np.arange(factor_dists.shape[0])[:, None], partitioning_order]
    return pd.DataFrame(np.median(distance_to_nearest_neighbors, axis=1),
        #distance_to_nearest_neighbors.sum(1)/(n_neighbors), 
                        columns=['local_density'],
                        index=factor_labels)


def make_consensus_plot(factor_dists, cluster_labels, local_density, density_threshold, dist_cmap='Reds_r'):
    width_ratios = [0.5, 9, 0.5, 4, 1]
    height_ratios = [0.5, 9]
    fig = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
    gs = gridspec.GridSpec(len(height_ratios), len(width_ratios), fig,
                       0.01, 0.01, 0.98, 0.98,
                       height_ratios=height_ratios,
                       width_ratios=width_ratios,
                       wspace=0, hspace=0)

    dist_ax = fig.add_subplot(gs[1,1], xscale='linear', yscale='linear',
                                      xticks=[], yticks=[],xlabel='', ylabel='',
                                      frameon=True)

    spectra_order = get_spectra_order(factor_dists, cluster_labels)
    D = factor_dists[spectra_order, :][:, spectra_order]
    dist_im = dist_ax.imshow(D, interpolation='none', cmap=dist_cmap, aspect='auto',
                                rasterized=True)
    
    left_ax = fig.add_subplot(gs[1,0], xscale='linear', yscale='linear', xticks=[], yticks=[],
                xlabel='', ylabel='', frameon=True)
    left_ax.imshow(cluster_labels.values[spectra_order].reshape(-1, 1),
                            interpolation='none', cmap='Spectral', aspect='auto',
                            rasterized=True)
    top_ax = fig.add_subplot(gs[0,1], xscale='linear', yscale='linear', xticks=[], yticks=[],
                xlabel='', ylabel='', frameon=True)
    top_ax.imshow(cluster_labels.values[spectra_order].reshape(1, -1), 
                              interpolation='none', cmap='Spectral', aspect='auto',
                                rasterized=True)
    
    hist_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 3],
                                   wspace=0, hspace=0)
    hist_ax = fig.add_subplot(hist_gs[0,0], xscale='linear', yscale='linear',
                xlabel='', ylabel='', frameon=True, title='Local density histogram')
    hist_ax.hist(local_density.values, bins=np.linspace(0,1,50)#np.sqrt(2),50)
            )
    hist_ax.yaxis.tick_right()

    xlim = hist_ax.get_xlim()
    ylim = hist_ax.get_ylim()
    #density_threshold = xlim[1] + 1
    if density_threshold < xlim[1]:
        density_filter = local_density.iloc[:, 0] < density_threshold
        hist_ax.axvline(density_threshold, linestyle='--', color='k')
        hist_ax.text(density_threshold  + 0.02, ylim[1] * 0.95, 'filtering\nthreshold\n\n', va='top')
        hist_ax.set_xlim(xlim)
        hist_ax.set_xlabel('Mean distance to k nearest neighbors\n\n%d/%d (%.0f%%) spectra above threshold\nwere removed prior to clustering'%(sum(~density_filter), len(density_filter), 100*(~density_filter).mean()))
        
    return fig, (dist_ax, left_ax, top_ax, hist_ax)


def refit_local_params(X, global_params, nfactors, bp, dp, a=0.3, c=0.3, project_kw={}):
    """
    """
    project_defaults = dict(verbose=True, max_iter=50, check_freq=5)
    eta_shp, eta_rte, beta_shp, beta_rte = global_params
    
    # make a model
    eta = schpf.HPF_Gamma(np.ravel(eta_shp), np.ravel(eta_rte))
    beta = schpf.HPF_Gamma(beta_shp.T.values, beta_rte.T.values)
    model = schpf.scHPF(nfactors, eta=eta, beta=beta, bp=bp, dp=dp, a=a, c=c)           
    
    # setup projection kwarg
    for k,v in project_defaults.items():
        if k not in project_kw.keys():
            project_kw[k] = v
    loss = model.project(X, replace=True, **project_kw)
    model.loss = loss
    
    return model


def get_ranked_genes(model, genes, name_col=1):
    gene_score = model.gene_score()

    # rank the genes by gene_score
    ranks = np.argsort(gene_score, axis=0)[::-1]
    ranked_genes = []
    for i in range(gene_score.shape[1]):
        ranked_genes.append(genes[ranks[:,i], name_col])
    ranked_genes = pd.DataFrame(np.stack(ranked_genes).T)
    return ranked_genes
