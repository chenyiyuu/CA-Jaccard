#!/usr/bin/env python2/python3
# -*- coding: utf-8 -*-
"""
Source: https://github.com/zhunzhong07/person-re-ranking
Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22.
- This version accepts distance matrix instead of raw features.
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

__all__ = ['re_ranking']

import numpy as np


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def re_ranking(q_g_dist, q_q_dist, g_g_dist, cids, args, lambda_value=0.3):
    k1, k2 = args.k1, args.k2
    ckrnns, k1_intra, k1_inter = args.ckrnns, args.k1_intra, args.k1_inter
    clqe, k2_intra, k2_inter = args.clqe, args.k2_intra, args.k2_inter
    if ckrnns and clqe:
        mode = f"[CAJaccard (CKRNNS + CLQE)]"
    elif ckrnns and not clqe:
        mode = f"[CAJaccard (CKRNNS + LQE)]"
    elif not ckrnns and clqe:
        mode = f"[CAJaccard (KRNNS + CLQE)]"
    else:
        mode = f"[Jaccard (KRNNS + LQE)]"
    print(mode)

    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    cam_mask = (cids.reshape(-1, 1) == cids.reshape(1, -1))

    inter_rank = np.argpartition(original_dist + 999.0 * cam_mask, range(k1_inter + 2))
    nn_inter = [k_reciprocal_neigh(inter_rank, i, k1_inter) for i in range(all_num)]
    intra_rank = np.argpartition(original_dist + 999.0 * (~cam_mask), range(k1_intra + 2))
    nn_intra = [k_reciprocal_neigh(intra_rank, i, k1_intra) for i in range(all_num)]

    ###################################
    #           KRNNs/CKRNNs          #
    ###################################
    if ckrnns:
        print(f"[CKRNNs] PARAMS: k1_intra: {k1_intra}, k1_inter: {k1_inter}")
    else:
        print(f"[KRNNs] PARAMS: k1: {k1}")

    for i in range(all_num):
        if ckrnns:
            k_reciprocal_index = np.append(nn_intra[i], nn_inter[i])
            k_reciprocal_expansion_index = k_reciprocal_index
        else:
            forward_k_neigh_index = initial_rank[i, :k1 + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
            fi = np.where(backward_k_neigh_index == i)[0]
            k_reciprocal_index = forward_k_neigh_index[fi]
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2.)) + 1]
                candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                   :int(np.around(k1 / 2.)) + 1]
                fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
                candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    ################################
    #            LQE/CLQE          #
    ################################
    V_qe = np.zeros_like(V, dtype=np.float32)
    if clqe:
        print(f"[CLQE] PARAMS: k2_intra: {k2_intra}, k2_inter: {k2_inter}")
    else:
        print(f"[LQE] PARAMS: k2: {k2}")

    for i in range(all_num):
        if clqe:
            k2nn = np.append(intra_rank[i, :k2_intra], inter_rank[i, :k2_inter])
        else:
            k2nn = initial_rank[i, :k2]
        V_qe[i, :] = np.mean(V[k2nn, :], axis=0)
    V = V_qe
    del V_qe
    # del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
