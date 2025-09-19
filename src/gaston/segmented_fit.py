from tqdm import trange
from sklearn import linear_model,preprocessing
import numpy as np

from scipy.stats import chi2, linregress
from sklearn.preprocessing import normalize

######################

# INPUTS:
# counts_mat: N x G matrix of counts
# gaston_labels, gaston_isodepth: N x 1 array with labels/isodepth for each spot (labels=0, ..., L-1)
# cell_type_df: N x C dataframe, rows are spots, columns are cell types, entries are cell type proportion in spot
# ct_list: list of cell types to compute piecewise linear fits for

# num_bins: number of bins to bin isodepth into (for visualization)
# pseudocount: pseudocount to add to counts
# umi_threshold: restrict to genes with total UMI count across all spots > umi_threshold
# after filtering, there are G' genes
# t: p-value threshold for LLR test (slope = 0 vs slope != 0)
# isodepth_mult_factor: if the range of isodepth values is too large (causes overflows)
#    then scale isodepth values by isodepth_mult_factor (ie isodepth -> isodepth * mult_factor)
#    to compute Poisson regressoin fits that are more numerically stable
# OUTPUTS:
# pw_fit_dict: dictionary indexed by cell types, as well as 'all_cell_types'
# pw_fit_dict[cell_type] = (slope_mat, intercept_mat, discont_mat, pv_mat)

# slope_mat, intercept_mat: G' x L, entries are slopes/intercepts
# discont_mat: G' x L-1, entries are discontinuity at domain boundaries
# pv_mat: G' x L, entries are p-values from LLR test (slope=0 vs slope != 0)
# def pw_linear_fit(counts_mat, gaston_labels, gaston_isodepth, cell_type_df, ct_list,
#                   umi_threshold=500, idx_kept=None, pc=0, pc_exposure=True, t=0.1,
#                   isodepth_mult_factor=1, reg=0, zero_fit_threshold=0):

#     counts_mat=counts_mat.T # TODO eventually: update code to use N x G matrix instead of G x N matrix...
#     if idx_kept is None:
#         idx_kept=np.where(np.sum(counts_mat,1) > umi_threshold)[0]

#     exposures=np.sum(counts_mat,0)
    
#     cmat=counts_mat[idx_kept,:]
    
#     G,N=cmat.shape
        
#     gaston_isodepth=gaston_isodepth * isodepth_mult_factor
#     L=len(np.unique(gaston_labels))
    
#     pw_fit_dict={}
    
#     # ONE: compute for all cell types
#     print('Poisson regression for ALL cell types')
#     s0_mat,i0_mat,s1_mat,i1_mat,pv_mat=segmented_poisson_regression(cmat,
#                                                    exposures, 
#                                                    gaston_labels, 
#                                                    gaston_isodepth,
#                                                    L, reg=reg)
    
#     slope_mat=np.zeros((len(idx_kept), L))
#     intercept_mat=np.zeros((len(idx_kept), L))
    
#     # use s0 fit for genes with lots of zeros
#     nonzero_per_domain = np.zeros((G,L))
#     for l in range(L):
#         cmat_l=cmat[:,gaston_labels==l]
#         nonzero_per_domain[:,l]=np.count_nonzero(cmat_l,1)

#     inds1= ((pv_mat < t) & (nonzero_per_domain >= zero_fit_threshold))
#     inds0= ((pv_mat >= t) | (nonzero_per_domain < zero_fit_threshold))
    
#     slope_mat[inds1] = s1_mat[inds1]
#     intercept_mat[inds1] = i1_mat[inds1]

#     slope_mat[inds0] = s0_mat[inds0]
#     intercept_mat[inds0] = i0_mat[inds0]
    
#     discont_mat=get_discont_mat(slope_mat, intercept_mat, gaston_labels, gaston_isodepth, L)

#     slope_mat = slope_mat * isodepth_mult_factor

#     pw_fit_dict['all_cell_types']=(slope_mat,intercept_mat,discont_mat, pv_mat)
    
#     # TWO: compute for each cell type in ct_list, if you have cell type info
#     if cell_type_df is None:
#         return pw_fit_dict
    
#     cell_type_mat=cell_type_df.to_numpy()
#     cell_type_names=np.array(cell_type_df.columns)
#     for ct in ct_list:
#         print(f'Poisson regression for cell type: {ct}')
#         ct_ind=np.where(cell_type_names==ct)[0][0]
        
#         ct_spots=np.where(cell_type_mat[:,ct_ind] > 0)[0]
#         ct_spot_proportions=cell_type_mat[ct_spots,ct_ind]
        
#         cmat_ct=cmat[:,ct_spots] * np.tile(ct_spot_proportions,(G,1))
#         exposures_ct=exposures[ct_spots] * ct_spot_proportions
#         gaston_labels_ct=gaston_labels[ct_spots]
#         gaston_isodepth_ct=gaston_isodepth[ct_spots]

#         s0_ct,i0_ct,s1_ct,i1_ct,pv_mat_ct=segmented_poisson_regression(cmat_ct,
#                                                        exposures_ct, 
#                                                        gaston_labels_ct, 
#                                                        gaston_isodepth_ct,
#                                                        L)

#         slope_mat_ct=np.zeros((len(idx_kept), L))
#         intercept_mat_ct=np.zeros((len(idx_kept), L))

#         inds1_ct= (pv_mat_ct < t)
#         inds0_ct= (pv_mat_ct >= t)

#         slope_mat_ct[inds1_ct] = s1_ct[inds1_ct]
#         intercept_mat_ct[inds1_ct] = i1_ct[inds1_ct]

#         slope_mat_ct[inds0_ct] = s0_ct[inds0_ct]
#         intercept_mat_ct[inds0_ct] = i0_ct[inds0_ct]
        
#         discont_mat=get_discont_mat(slope_mat_ct, intercept_mat_ct, gaston_labels_ct, gaston_isodepth_ct, L)

#         slope_mat_ct = slope_mat_ct * isodepth_mult_factor
#         pw_fit_dict[ct]=(slope_mat_ct, intercept_mat_ct, discont_mat, pv_mat_ct)
          
#     return pw_fit_dict
def pw_linear_fit(counts_mat, gaston_labels, gaston_isodepth, cell_type_df, ct_list,
                  umi_threshold=500, idx_kept=None, t=0.1,
                  isodepth_mult_factor=1, reg=0, zero_fit_threshold=0):

    counts_mat = counts_mat.T
    if idx_kept is None:
        idx_kept = np.where(np.sum(counts_mat,1) > umi_threshold)[0]

    exposures = np.sum(counts_mat,0)
    cmat = counts_mat[idx_kept,:]
    
    G, N = cmat.shape
    gaston_isodepth = gaston_isodepth * isodepth_mult_factor
    L = len(np.unique(gaston_labels))
    
    pw_fit_dict = {}
    
    # --- compute for all cell types ---
    print('Poisson regression for ALL cell types')
    s0_mat, i0_mat, s1_mat, i1_mat, pv_mat = segmented_poisson_regression(
        cmat, exposures, gaston_labels, gaston_isodepth, L, reg=reg
    )
    
    slope_mat = np.zeros((len(idx_kept), L))
    intercept_mat = np.zeros((len(idx_kept), L))
    
    nonzero_per_domain = np.zeros((G,L))
    for l in range(L):
        cmat_l = cmat[:, gaston_labels==l]
        nonzero_per_domain[:, l] = np.count_nonzero(cmat_l, 1)

    inds1 = ((pv_mat < t) & (nonzero_per_domain >= zero_fit_threshold))
    inds0 = ((pv_mat >= t) | (nonzero_per_domain < zero_fit_threshold))
    
    slope_mat[inds1] = s1_mat[inds1]
    intercept_mat[inds1] = i1_mat[inds1]

    slope_mat[inds0] = s0_mat[inds0]
    intercept_mat[inds0] = i0_mat[inds0]

    # --- compute goodness-of-fit for all genes/domains ---
    fit_score_mat = np.zeros((len(idx_kept), L))
    for l in range(L):
        idx_spots = np.where(gaston_labels == l)[0]
        for i in range(len(idx_kept)):
            y = cmat[i, idx_spots]
            lam = exposures[idx_spots] * np.exp(slope_mat[i, l] * gaston_isodepth[idx_spots] + intercept_mat[i, l])
            fit_score_mat[i, l] = poisson_deviance_explained(y, lam)

    discont_mat = get_discont_mat(slope_mat, intercept_mat, gaston_labels, gaston_isodepth, L)
    slope_mat = slope_mat * isodepth_mult_factor

    # store everything including deviance-explained
    pw_fit_dict['all_cell_types'] = (slope_mat, intercept_mat, discont_mat, pv_mat, fit_score_mat)
    
    # --- compute for each cell type ---
    if cell_type_df is not None:
        cell_type_mat = cell_type_df.to_numpy()
        cell_type_names = np.array(cell_type_df.columns)
        for ct in ct_list:
            print(f'Poisson regression for cell type: {ct}')
            ct_ind = np.where(cell_type_names == ct)[0][0]
            
            ct_spots = np.where(cell_type_mat[:, ct_ind] > 0)[0]
            ct_spot_proportions = cell_type_mat[ct_spots, ct_ind]
            
            cmat_ct = cmat[:, ct_spots] * np.tile(ct_spot_proportions, (G,1))
            exposures_ct = exposures[ct_spots] * ct_spot_proportions
            gaston_labels_ct = gaston_labels[ct_spots]
            gaston_isodepth_ct = gaston_isodepth[ct_spots]

            s0_ct, i0_ct, s1_ct, i1_ct, pv_mat_ct = segmented_poisson_regression(
                cmat_ct, exposures_ct, gaston_labels_ct, gaston_isodepth_ct, L
            )

            slope_mat_ct = np.zeros((len(idx_kept), L))
            intercept_mat_ct = np.zeros((len(idx_kept), L))
            fit_score_mat_ct = np.zeros((len(idx_kept), L))

            inds1_ct = (pv_mat_ct < t)
            inds0_ct = (pv_mat_ct >= t)

            slope_mat_ct[inds1_ct] = s1_ct[inds1_ct]
            intercept_mat_ct[inds1_ct] = i1_ct[inds1_ct]

            slope_mat_ct[inds0_ct] = s0_ct[inds0_ct]
            intercept_mat_ct[inds0_ct] = i0_ct[inds0_ct]

            for l in range(L):
                idx_spots = np.where(gaston_labels_ct == l)[0]
                for i in range(len(idx_kept)):
                    y = cmat_ct[i, idx_spots]
                    lam = exposures_ct[idx_spots] * np.exp(slope_mat_ct[i, l] * gaston_isodepth_ct[idx_spots] + intercept_mat_ct[i, l])
                    fit_score_mat_ct[i, l] = poisson_deviance_explained(y, lam)

            discont_mat = get_discont_mat(slope_mat_ct, intercept_mat_ct, gaston_labels_ct, gaston_isodepth_ct, L)
            slope_mat_ct = slope_mat_ct * isodepth_mult_factor

            pw_fit_dict[ct] = (slope_mat_ct, intercept_mat_ct, discont_mat, pv_mat_ct, fit_score_mat_ct)
          
    return pw_fit_dict


######################


def poisson_deviance_explained(y, lam):
    """
    Compute deviance explained (pseudo-R²) for Poisson regression.
    Returns a value between 0 and 1.
    """
    y = np.asarray(y, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    
    mask = lam > 0
    y_masked = y[mask]
    lam_masked = lam[mask]
    
    if len(y_masked) == 0:
        return 0.0
    
    # Null model: constant mean
    lam0 = np.mean(y_masked) if np.mean(y_masked) > 0 else 1e-6
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dev_model = 2 * np.sum(np.where(y_masked > 0,
                                        y_masked * np.log(y_masked / lam_masked) - (y_masked - lam_masked),
                                        -lam_masked))
        dev_null  = 2 * np.sum(np.where(y_masked > 0,
                                        y_masked * np.log(y_masked / lam0) - (y_masked - lam0),
                                        -lam0))
    
    r2 = 1 - dev_model / dev_null
    return max(0.0, min(1.0, r2))

################

def llr_poisson(y, xcoords=None, exposure=None, alpha=0):
    s0, i0 = poisson_regression(y, xcoords=0*xcoords, exposure=exposure, alpha=alpha)
    s1, i1 = poisson_regression(y, xcoords=xcoords, exposure=exposure, alpha=alpha)
    
    ll0=poisson_likelihood(s0,i0,y,xcoords=xcoords,exposure=exposure)
    ll1=poisson_likelihood(s1,i1,y,xcoords=xcoords,exposure=exposure)
    
    return s0, i0, s1, i1, chi2.sf(2*(ll1-ll0),1)

def poisson_likelihood(slope, intercept, y, xcoords=None, exposure=None):
    lam=exposure * np.exp(slope * xcoords + intercept)
    return np.sum(y * np.log(lam) - lam)

def poisson_regression(y, xcoords=None, exposure=None, alpha=0):
    # run poisson fit on pooled data and return slope, intercept
    clf = linear_model.PoissonRegressor(fit_intercept=True,alpha=alpha,max_iter=500,tol=1e-10)
    clf.fit(np.reshape(xcoords,(-1,1)),y/exposure, sample_weight=exposure)

    return [clf.coef_[0], clf.intercept_ ]

def segmented_poisson_regression(count, totalumi, dp_labels, isodepth, num_domains,
                                 opt_function=poisson_regression, reg=0):
    """ Fit Poisson regression per gene per domain.
    :param count: UMI count matrix of SRT gene expression, G genes by n spots
    :type count: np.array
    :param totalumi: Total UMI count per spot, a vector of n spots.
    :type totalumi: np.array
    :param dp_labels: domain labels obtained by DP, a vector of n spots.
    :type dp_labels: np.array
    :param isodepth: Inferred domain isodepth, vector of n spots
    :type isodepth: np.array
    :return: A dataframe for the offset and slope of piecewise linear expression function, size of G genes by 2*L domains.
    :rtype: pd.DataFrame
    """

    G, N = count.shape
    unique_domains = np.sort(np.unique(dp_labels))
    # L = len(unique_domains)
    L=num_domains

    slope1_matrix=np.zeros((G,L))
    intercept1_matrix=np.zeros((G,L))
    
    # null setting
    slope0_matrix=np.zeros((G,L))
    intercept0_matrix=np.zeros((G,L))
    
    pval_matrix=np.zeros((G,L))

    for g in trange(G):
        for t in unique_domains:
            pts_t=np.where(dp_labels==t)[0]
            t=int(t)
            
            # need to be enough points in domain
            if len(pts_t) > 10:
                s0, i0, s1, i1, pval = llr_poisson(count[g,pts_t], xcoords=isodepth[pts_t], exposure=totalumi[pts_t], alpha=reg)
            else:
                s0=np.inf
                i0=np.inf
                s1=np.inf
                i1=np.inf
                pval=np.inf
        
            slope0_matrix[g,t]=s0
            intercept0_matrix[g,t]=i0
            
            slope1_matrix[g,t]=s1
            intercept1_matrix[g,t]=i1
            
            pval_matrix[g,t]=pval
            
    return slope0_matrix,intercept0_matrix,slope1_matrix,intercept1_matrix, pval_matrix

def get_discont_mat(s_mat, i_mat, gaston_labels, gaston_isodepth, num_domains):
    G,_=s_mat.shape
    L=num_domains
    discont_mat=np.zeros((G,L-1))
    
    for l in range(L-1):
        pts_l=np.where(gaston_labels==l)[0]
        pts_l1=np.where(gaston_labels==l+1)[0]
        
        if len(pts_l) > 0 and len(pts_l1) > 0:
            x_left=np.max(gaston_isodepth[pts_l])
            y_left=s_mat[:,l]*x_left + i_mat[:,l]

            x_right=np.min(gaston_isodepth[pts_l1])
            y_right=s_mat[:,l+1]*x_right + i_mat[:,l+1]
            discont_mat[:,l]=y_right-y_left
        else:
            discont_mat[:,l]=0
    return discont_mat

