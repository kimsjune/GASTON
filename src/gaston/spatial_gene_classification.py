import numpy as np
from collections import defaultdict

# def get_discont_genes(pw_fit_dict, binning_output, q=0.95):
    
#     _,_,discont_mat,_=pw_fit_dict['all_cell_types']
#     gene_labels_idx=binning_output['gene_labels_idx']
    
#     discont_genes=defaultdict(list) # gene -> [domain boundary p], ie bounary between R_p and R_{p+1}
    
#     discont_q=np.quantile(np.abs(discont_mat), q,0)
#     K=len(discont_q)
#     for i,g in enumerate(gene_labels_idx):
#         for l in range(K):
#             if np.abs(discont_mat[i,l]) > discont_q[l]:
#                 #if g not in discont_genes:
#                 #    discont_genes[g]=[l]
#                 #else:
#                 discont_genes[g].append(l)
    
#     # discont_genes=list( np.where(np.sum(np.abs(discont_mat) > discont_q,1))[0] )    

#     return discont_genes

def get_discont_genes(
    pw_fit_dict,
    binning_output,
    q=0.95,
    min_score=0.3  # <<< NEW: minimum goodness-of-fit threshold
):
    """
    Select genes with significant discontinuities between domains and good fit.

    Returns
    -------
    discont_genes : dict
        Keys = gene names, Values = list of domain boundaries with strong discontinuities
    """

    discont_genes = defaultdict(list)
    gene_labels_idx = binning_output['gene_labels_idx']

    slope_mat_all, intercept_mat_all, discont_mat, _, fit_score_mat = pw_fit_dict['all_cell_types']

    # Compute threshold for discontinuity (absolute value)
    discont_q = np.quantile(np.abs(discont_mat), q, 0)
    Lm1 = discont_mat.shape[1]  # L-1 boundaries

    for i, g in enumerate(gene_labels_idx):
        for l in range(Lm1):
            # <<< NEW: require both discontinuity > quantile AND min goodness-of-fit in adjacent domains
            if np.abs(discont_mat[i, l]) > discont_q[l] and \
               fit_score_mat[i, l] >= min_score and fit_score_mat[i, l+1] >= min_score:
                discont_genes[g].append(l)

    return discont_genes

# def get_cont_genes(pw_fit_dict, binning_output, q=0.95, ct_attributable=False, domain_cts=None, ct_perc=0.6):
    
#     cont_genes=defaultdict(list) # dict of gene -> [list of domains]
#     gene_labels_idx=binning_output['gene_labels_idx']
    

#     slope_mat_all,_,_,_=pw_fit_dict['all_cell_types']
#     slope_q=np.quantile(np.abs(slope_mat_all), q,0)
    
#     L=len(slope_q)
#     for i,g in enumerate(gene_labels_idx):
#         for l in range(L):
#             if np.abs(slope_mat_all[i,l]) > slope_q[l]:
#                 #if g not in cont_genes:
#                 #    cont_genes[g]=[l]
#                 #else:
#                 cont_genes[g].append(l)
    
#     if not ct_attributable:
#         return cont_genes
    
#     cont_genes_domain_ct={g: [] for g in cont_genes} # dict gene -> [(domain,ct)]

#     for g in cont_genes:
#         for l in cont_genes[g]:
#             other=True
#             for ct in domain_cts[l]:
#                 if np.abs( pw_fit_dict[ct][0][gene_labels_idx==g,l] ) / np.abs(pw_fit_dict['all_cell_types'][0][gene_labels_idx==g,l]) > ct_perc:
#                     other=False
#                     cont_genes_domain_ct[g].append( (l,ct) )
                
#             if other:
#                 cont_genes_domain_ct[g].append( (l, 'Other') )
                
#     return cont_genes_domain_ct
def get_cont_genes(
    pw_fit_dict,
    binning_output,
    q=0.95,
    min_score=0.3  # <<< NEW: minimum goodness-of-fit threshold
):
    """
    Select genes with strong slopes and good fit quality.

    Returns
    -------
    cont_genes : dict
        Keys = gene names, Values = list of domains that pass filters
    """

    cont_genes = defaultdict(list)
    gene_labels_idx = binning_output['gene_labels_idx']

    # <<< NEW: unpack fit_score_mat from pw_fit_dict
    slope_mat_all, _, _, _, fit_score_mat = pw_fit_dict['all_cell_types']

    slope_q = np.quantile(np.abs(slope_mat_all), q, 0)
    L = len(slope_q)

    for i, g in enumerate(gene_labels_idx):
        for l in range(L):
            # <<< NEW: require both slope > quantile AND fit_score >= min_score
            if np.abs(slope_mat_all[i, l]) > slope_q[l] and fit_score_mat[i, l] >= min_score:
                cont_genes[g].append(l)

    return cont_genes

######################################################
# Get Type I, II, III gene classification from colorectal tumor analysis (see manuscript)
######################################################

def get_type_123_genes(binning_output, discont_genes, cont_genes):
    gene_labels_idx=binning_output['gene_labels_idx']

    result_dict = {f'{i:03b}': [] for i in range(8)}

    for gene in gene_labels_idx:
        A = '1' if gene in cont_genes and 0 in cont_genes[gene] else '0'
        B = '1' if gene in discont_genes else '0'
        C = '1' if gene in cont_genes and 1 in cont_genes[gene] else '0'
        
        binary_vector = A + B + C
        result_dict[binary_vector].append(gene)
    return result_dict