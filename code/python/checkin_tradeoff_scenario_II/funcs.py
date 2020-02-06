import pandas as pd
import numpy as np
import math
import xgboost as xgb
import copy
from scipy.cluster import hierarchy
from scipy.stats import entropy
from numpy.linalg import norm
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import normalize


def update_grid_group(df_train, grid_group_dict):
    df_train_copy = df_train.copy()
    for k, v in grid_group_dict.items():
        df_train_copy.loc[df_train['grid']==k, 'grid_group'] = v
    return df_train_copy


def k_anonymity(df_train, group_age_list):
    return df_train.loc[df_train['grid'].isin(group_age_list)].shape[0]


def l_diversity(df_train, group_grid_list):
    grid_dtr = np.zeros(len(group_grid_list))
    for i in range(len(group_grid_list)):
        grid_dtr[i] = df_train.loc[df_train['grid'] == group_grid_list[i]].shape[0]
    grid_dtr_norm = grid_dtr / norm(grid_dtr, ord=1)
    return entropy(grid_dtr_norm)


def update_rowcol_colrow_dict(area_grid_dict):
    area_grid_rowcol_dict = {}
    area_grid_colrow_dict = {}

    for area in area_grid_dict:
        area_grid_rowcol_dict[area] = {}
        area_grid_colrow_dict[area] = {}

        for grid_id in area_grid_dict[area]:
            row, col = grid_to_rowcol(grid_id)

            if row in area_grid_rowcol_dict[area]:
                area_grid_rowcol_dict[area][row].append(col)
            else:
                area_grid_rowcol_dict[area][row] = [col]

            if col in area_grid_colrow_dict[area]:
                area_grid_colrow_dict[area][col].append(row)
            else:
                area_grid_colrow_dict[area][col] = [row]

    return area_grid_rowcol_dict, area_grid_colrow_dict


def grid_to_rowcol(grid):
    row = int(grid/60)
    col = grid % 60
    return row,col


def get_JSD_PGY(df_cluster, area_grid_dict, JSD_Mat_dict, cluster_num, pgy_dict, JSD_Mat, pgy):
    grid_area_number = len(list(area_grid_dict.keys()))
    group_usersize_dict = {}
    group_grid_size = {}
    group_grid_size[0] = 0
    for gg in range(0, grid_area_number):
        df_cluster_gg = df_cluster.loc[df_cluster['grid_group'] == gg]
        grid_list_gg = area_grid_dict[gg]
        grid_list_gg.sort()
        if gg > 0:
            group_grid_size[gg] = len(area_grid_dict[gg - 1]) + group_grid_size[gg - 1]
        group_usersize_dict[gg] = df_cluster_gg.shape[0]

        JSD_Mat_dict[gg] = cal_JSD_Matrix_withoutGridGroup(df_cluster_gg, cluster_num, 4)
        pgy_dict[gg] = cal_pgy_withoutGridGroup(df_cluster_gg, cluster_num, grid_list_gg)
        pd.DataFrame(JSD_Mat_dict[gg]).to_csv('JSDM_check_in_allObf_seeds1_' + str(gg) + '.csv', index=False,
                                              header=None)
        pd.DataFrame(pgy_dict[gg]).to_csv('pgy_check_in_allObf_seeds1_' + str(gg) + '.csv', index=False, header=None)

    for gg in range(grid_area_number):
        grid_list_gg = area_grid_dict[gg]
        grid_list_gg.sort()
        for grid_index in range(len(grid_list_gg)):
            for col in range(cluster_num):
                if gg == 0:
                    row_num = grid_index
                else:
                    row_num = grid_index + group_grid_size[gg]
                pgy[row_num, gg + col * grid_area_number] = pgy_dict[gg][grid_index, col] * group_usersize_dict[gg] / \
                                                            df_cluster.shape[0]

    for gg in range(grid_area_number):
        for row in range(cluster_num):
            for col in range(cluster_num):
                JSD_Mat[gg + row * grid_area_number, gg + col * grid_area_number] = JSD_Mat_dict[gg][row, col]

    return JSD_Mat, pgy, JSD_Mat_dict, pgy_dict


def cal_JSD_Matrix_withoutGridGroup(df_cluster, cluster_dim, non_item_col):
    df_cluster_dict = {}
    for i in range(1, cluster_dim + 1):
        df_cluster_dict[i] = df_cluster.loc[df_cluster['cluster'] == i]

    default_max_JSD = 1
    JSD_Mat = np.ones((cluster_dim, cluster_dim)) * default_max_JSD

    JSD_cluster = []
    for i in range(1, cluster_dim + 1):
        if df_cluster_dict[i].empty:
            continue
        else:
            items_array_i = df_cluster_dict[i].values[:, :-non_item_col]
            for j in range(i, cluster_dim + 1):
                if df_cluster_dict[j].empty:
                    continue
                else:
                    items_array_j = df_cluster_dict[j].values[:, :-non_item_col]
                    for P in items_array_i:
                        if np.sum(P) > 0:
                            for Q in items_array_j:
                                if np.sum(Q) > 0:
                                    JSD_cluster.append(JSD(P, Q))
                    mean_JSD = np.mean(np.array(JSD_cluster))
                    mean_JSD = mean_JSD if mean_JSD > 0 else 0.000001
                    JSD_Mat[i - 1, j - 1] = mean_JSD
                    JSD_Mat[j - 1, i - 1] = mean_JSD
                    del JSD_cluster[:]

    return JSD_Mat


def cal_pgy_withoutGridGroup(df_cluster, cluster_dim, grid_list):
    user_size = df_cluster.shape[0]

    df_cluster_dict = {}
    for i in range(1, cluster_dim + 1):
        df_cluster_dict[i] = df_cluster.loc[df_cluster['cluster'] == i, ['grid', 'grid_group', 'cluster']]

    pgy_Mat = np.ones((len(grid_list), cluster_dim))*0.0000001

    for i in range(1, cluster_dim + 1):
        if not df_cluster_dict[i].empty:
            group_grid_cnt = df_cluster_dict[i].groupby(['grid']).size().reset_index(name='count')
            j = 0
            for grid in grid_list:
                if grid in list(group_grid_cnt['grid']):
#                     print(grid, group_grid_cnt.loc[group_grid_cnt['grid'] == grid, 'count'])
                    pgy_Mat[j, i - 1] = group_grid_cnt.loc[group_grid_cnt['grid'] == grid, 'count'] / user_size
                j += 1
    return pgy_Mat


def get_obf_X(df_cluster, xpgg, pp):
    user_cluster_dict = {}
    cluster_vec = {}
    user_gridGroup_dict = {}
    gridGroup_vec = {}
    user_size = df_cluster.shape[0]

    for i in range(user_size):
        user_id = df_cluster['uid'][i]
        cluster_id = df_cluster['cluster'][i]
        user_cluster_dict[user_id] = cluster_id
        if cluster_id in cluster_vec:
            cluster_vec[cluster_id].append(user_id)
        else:
            cluster_vec[cluster_id] = [user_id]

        grid_group = df_cluster['grid_group'][i]
        user_gridGroup_dict[user_id] = grid_group
        if grid_group in gridGroup_vec:
            gridGroup_vec[grid_group].append(user_id)
        else:
            gridGroup_vec[grid_group] = [user_id]

    cluster_size = len(cluster_vec)
    gridGroup_size = len(gridGroup_vec)
    print('gridGroup_size in get_obf_X:', gridGroup_size)

    X_ori = {}
    X_obf = {}

    xpgg[xpgg < 0.00001] = 0
    xpgg_norm = normalize(xpgg, axis=0, norm='l1')

    print("obfuscating...")

    for i in range(user_size):
        user_id = df_cluster['uid'][i]
        X_ori[user_id] = df_cluster[df_cluster['uid'] == user_id].values[0, :-1]
        obf_flag = np.random.choice([0, 1], 1, p=[pp, 1 - pp])
        if obf_flag == 1:
            # selecting one cluster to change
            while (True):
                change_index = np.random.choice(range(0, cluster_size * gridGroup_size), 1, p=xpgg_norm[:, int(
                    user_gridGroup_dict[user_id] + (user_cluster_dict[user_id] - 1) * gridGroup_size)])[0]
                change_cluster_index = int(change_index / gridGroup_size) + 1
                change_ageGroup_index = change_index % gridGroup_size
                potential_users = list(
                    set(cluster_vec[change_cluster_index]) & set(gridGroup_vec[change_ageGroup_index]))
                if len(potential_users) > 0:  # potential users may be empty by a slight probability
                    break
                else:
                    print("not find potential users, re-pick")
            #         print("potential users", potential_users)
            uidx = np.random.choice(potential_users, 1)[0]

            X_obf[user_id] = df_cluster[df_cluster['uid'] == uidx].values[0, :-3]
        else:
            X_obf[user_id] = df_cluster[df_cluster['uid'] == user_id].values[0, :-3]
    return X_obf, X_ori


def get_obf_X_withAgeGroup(df_cluster, xpgg, pp):
    # user: cluster
    user_cluster_dict = {}
    # cluster: [users]
    cluster_vec = {}
    # user: gridGroup
    user_gridGroup_dict = {}
    # gridGroup: [users]
    gridGroup_vec = {}
    user_size = df_cluster.shape[0]

    for i in range(user_size):
        user_id = df_cluster['uid'][i]
        cluster_id = df_cluster['cluster'][i]
        user_cluster_dict[user_id] = cluster_id
        if cluster_id in cluster_vec:
            cluster_vec[cluster_id].append(user_id)
        else:
            cluster_vec[cluster_id] = [user_id]

        grid_group = df_cluster['grid_group'][i]
        user_gridGroup_dict[user_id] = grid_group
        if grid_group in gridGroup_vec:
            gridGroup_vec[grid_group].append(user_id)
        else:
            gridGroup_vec[grid_group] = [user_id]

    cluster_size = len(cluster_vec)
    gridGroup_size = 1
    print('gridGroup_size in get_obf_X:', len(gridGroup_vec))

    X_ori = {}
    X_obf = {}

    xpgg[xpgg < 0.00001] = 0
    xpgg_norm = normalize(xpgg, axis=0, norm='l1')

    print("obfuscating...")
    for i in range(user_size):
        user_id = df_cluster['uid'][i]
        user_group = df_cluster['grid_group'][i]
        change_ageGroup_index = user_group
        X_ori[user_id] = df_cluster[df_cluster['uid'] == user_id].values[0, :-1]
        obf_flag = np.random.choice([0, 1], 1, p=[pp, 1-pp])
        if obf_flag == 1:
            # selecting one cluster to change
            while (True):
                change_index = np.random.choice(range(0, cluster_size*gridGroup_size), 1, p=xpgg_norm[:, (user_cluster_dict[user_id] - 1)])[0]
                change_cluster_index = int(change_index/gridGroup_size) + 1
                potential_users = list(set(cluster_vec[change_cluster_index]) & set(gridGroup_vec[change_ageGroup_index]))
                if len(potential_users) > 0: # potential users may be empty by a slight probability
                    break
                else:
                    print("not find potential users, re-pick")
    #         print("potential users", potential_users)
            uidx = np.random.choice(potential_users, 1)[0]

            X_obf[user_id] = df_cluster[df_cluster['uid'] == uidx].values[0, :-3]
        else:
            X_obf[user_id] = df_cluster[df_cluster['uid'] == user_id].values[0, :-3]

    return X_obf, X_ori


def get_random_obf_X_withAgeGroup(df_train, p_rand):
    X_ori = {}
    X_obf = {}

    user_size = df_train.shape[0]
    ageGroup_dict = {}
    for i in range(user_size):
        user_id = df_train['uid'][i]
        age_group = df_train['grid_group'][i]
        if age_group not in ageGroup_dict:
            ageGroup_dict[age_group] = [user_id]
        else:
            ageGroup_dict[age_group].append(user_id)

    for i in range(user_size):
        user_id = df_train['uid'][i]
        age_group = df_train['grid_group'][i]
        # get X_ori
        X_ori[user_id] = df_train[df_train['uid'] == user_id].values[0, :-1]

        # get X_obf
        flag = np.random.choice([0, 1], 1, p=[1 - p_rand, p_rand])[0]
        if flag == 0:
            X_obf[user_id] = df_train[df_train['uid'] == user_id].values[0, :-3]
        else:
            candidate_users = list(set(ageGroup_dict[age_group]) - set([user_id]))
            # print(candidate_users)
            uidx = np.random.choice(candidate_users, 1)[0]
            X_obf[user_id] = df_train[df_train['uid'] == uidx].values[0, :-3]

    return X_obf, X_ori


def get_similarity_obf_X_withAgeGroup(sim_mat, df_train, pp):
    X_ori = {}
    X_obf = {}

    sim_mat_copy = copy.deepcopy(sim_mat)
    user_size = df_train.shape[0]

    sim_dict = {}
    groupUser_dict = {}
    userIndex_dict = {}
    for i in range(user_size):
        user_id = df_train['uid'][i]
        user_group = df_train['grid_group'][i]
        if user_group not in groupUser_dict:
            groupUser_dict[user_group] = [user_id]
        else:
            groupUser_dict[user_group].append(user_id)

        userIndex_dict[user_id] = i

        sim_array = sim_mat_copy[i]
        sim_array[i] = 0
        sim_dict[user_id] = sim_array

    for i in range(user_size):
        user_id = df_train['uid'][i]
        user_group = df_train['grid_group'][i]
        X_ori[user_id] = df_train[df_train['uid'] == user_id].values[0, :-1]

        obf_flag = np.random.choice([0, 1], 1, p=[pp, 1 - pp])

        if obf_flag == 1:
            all_sameGroup_users = groupUser_dict[user_group]
            index_list = [userIndex_dict[u] for u in all_sameGroup_users]
            sim_arr = sim_dict[user_id]
            sim_series = pd.Series(sim_arr).loc[index_list]
            prob_arr = np.array(sim_series) / np.sum(np.array(sim_series))
            uidx = np.random.choice(all_sameGroup_users, 1, p=prob_arr)[0]
            X_obf[user_id] = df_train[df_train['uid']==uidx].values[0, :-3]
        else:
            X_obf[user_id] = df_train[df_train['uid']==user_id].values[0, :-3]

    return X_obf, X_ori


def get_DP_obf_X_withAgeGroup(df_train, dist_mat_dict, beta):
    X_ori = {}
    X_obf = {}
    user_size = df_train.shape[0]
    ageGroup_dict = {}
    user_group_dict = {}
    group_nums = df_train['grid_group'].nunique()
    for i in range(group_nums):
        user_group_dict[i] = list(df_train[df_train['grid_group'] == i].uid)

    for i in range(user_size):
        user_id = df_train['uid'][i]
        age_group = df_train['grid_group'][i]
        if age_group not in ageGroup_dict:
            ageGroup_dict[age_group] = [user_id]
        else:
            ageGroup_dict[age_group].append(user_id)

    for i in range(user_size):
        user_id = df_train['uid'][i]
        age_group = df_train['grid_group'][i]
        user_idxInGroup = user_group_dict[age_group].index(user_id)
        # get X_ori
        X_ori[user_id] = df_train[df_train['uid']==user_id].values[0, :-1]
        # get X_obf
        dist_mat = dist_mat_dict[age_group]
        dist_arr = np.array(list(dist_mat[user_idxInGroup]))
        dp_arr = np.exp(-beta * dist_arr)
        prob_list = list(dp_arr / sum(dp_arr))
        uidx = np.random.choice(ageGroup_dict[age_group], 1, p=prob_list)[0]
        X_obf[user_id] = df_train[df_train['uid'] == uidx].values[0, :-3]

    return X_obf, X_ori


def  get_frapp_obf_X_withAgeGroup(df_train, gamma):
    X_ori = {}
    X_obf = {}

    user_size = df_train.shape[0]
    ageGroup_dict = {}
    for i in range(user_size):
        user_id = df_train['uid'][i]
        age_group = df_train['grid_group'][i]
        if age_group not in ageGroup_dict:
            ageGroup_dict[age_group] = [user_id]
        else:
            ageGroup_dict[age_group].append(user_id)

    for i in range(user_size):
        user_id = df_train['uid'][i]
        age_group = df_train['grid_group'][i]
        # get X_ori
        X_ori[user_id] = df_train[df_train['uid'] == user_id].values[0, :-1]
        # get X_obf
        group_user_size = len(ageGroup_dict[age_group])
        e = 1 / (gamma + group_user_size)
        pfrapp = gamma * e
        p_list = [e] * group_user_size
        p_list[ageGroup_dict[age_group].index(user_id)] = pfrapp
        p_list = list(np.array(p_list) / sum(np.array(p_list)))

        uidx = np.random.choice(ageGroup_dict[age_group], 1, p=p_list)[0]
        X_obf[user_id] = df_train[df_train['uid'] == uidx].values[0, :-3]

    return X_obf, X_ori


def get_user_age_dict(df):
    user_age_dict = {}
    uids = list(df.index)
    age = list(df.age)
    for i in range(len(uids)):
        user_age_dict[uids[i]] = age[i]

    return user_age_dict

def get_user_age_dict_priv(df):
    user_age_dict = {}
    uids = list(df['uid'])
    age = list(df.age)
    for i in range(len(uids)):
        user_age_dict[uids[i]] = age[i]

    return user_age_dict

def get_uid_index_dict(df):
    uid_index_dict = {}
    indices = df.index
    for i in range(len(indices)):
        uid_index_dict[indices[i]] = i
    return uid_index_dict


def get_df_with_uid(df, age):
    df_age_uid = df.copy().reset_index()
    uid_col = df_age_uid['uid']
    df_age_uid.drop(['uid'], axis=1, inplace=True)
    df_age_uid['age'] = age
    df_age_uid.insert(len(df_age_uid.columns), 'uid', uid_col)
    return df_age_uid


def dice_similarity(u, v):
    upper = sum(np.logical_and(u,v))+1
    down = sum(u)+sum(v)
    return 2*upper/down


def Kmeans_clustering(test_df, cluster_num, item_size):
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(test_df.values[:, :item_size])
    P = kmeans.labels_
    dfn = test_df.copy()
    dfn['cluster'] = P+1
    return dfn


def Kmeans_clustering_for_cluster(test_df, cluster_num, item_size):
    kmeans = KMeans(n_clusters=cluster_num).fit(test_df.values[:, :item_size])
    P = kmeans.labels_
    dfn = test_df.copy()
    dfn['cluster'] = P+1
    euclidean_ss = silhouette_score(test_df.values[:, :item_size], P, metric='euclidean')
    cosine_ss = silhouette_score(test_df.values[:, :item_size], P, metric='cosine')
    return dfn, euclidean_ss, cosine_ss


def Spectral_clustering(test_df, cluster_num, item_size):
    s_cluster = SpectralClustering(n_clusters=cluster_num, random_state=0).fit(test_df.values[:, :item_size])
    P = s_cluster.labels_
    dfn = test_df.copy()
    dfn['cluster'] = P + 1
    return dfn


def hierarchical_clustering(test_df, cluster_num, item_size, affinity_type, linkage_type):
    hierarchy_cluster = AgglomerativeClustering(affinity=affinity_type, n_clusters=cluster_num, linkage=linkage_type).fit(test_df.values[:, :item_size])
    P = hierarchy_cluster.labels_
    dfn = test_df.copy()
    dfn['cluster'] = P + 1
    return dfn


def clustering(test_df, cluster_num, item_size):
    DistMat = hierarchy.distance.pdist(test_df.values[:, :item_size], metric = 'cosine')# metric='Dice')
    for i in range(len(DistMat)):
        if math.isnan(DistMat[i]):
            DistMat[i] = 0
    Z = hierarchy.linkage(y=DistMat, method='complete')
    P = hierarchy.fcluster(Z, t=cluster_num, criterion='maxclust')
    dfn = test_df.copy()
    dfn['cluster'] = P
    return dfn


def Mean_JSD(JSD, X):
    return np.sum(np.multiply(JSD,X))/X.shape[1]


def Mean_KL_div(pgy, X):
    pg = pgy.sum(axis = 1).reshape(pgy.shape[0],1)
    pyhat = np.dot(pgy.sum(axis = 0),X).reshape(pgy.shape[1],1)
    pgyhat = np.dot(pgy,X)
    return KL_div(pgyhat, np.dot(pg, pyhat.T))


def KL_div(X, Y):
    kl = np.multiply(X, np.log(np.divide(X,Y)))-X+Y    
    return np.sum(kl)/np.log(2)


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def cal_JSD_Matrix_withAgeGroup(df_cluster, cluster_dim, age_group_size, non_item_col):
    df_cluster_dict = {}
    for i in range(1, cluster_dim+1):
        df_cluster_dict[i] = df_cluster.loc[df_cluster['cluster'] == i]
    
    default_max_JSD = 1
    JSD_Mat = np.ones((cluster_dim* age_group_size, cluster_dim* age_group_size))*default_max_JSD
    
    JSD_cluster = []
    for ag in range(0, age_group_size):
#         print(df_cluster.loc[df_cluster['age_group'] == ag])
        for i in range(1, cluster_dim+1):            
#             print(ag, i, df_cluster_dict[i].shape, df_cluster_dict[i])
            df_cluster_i_ag = df_cluster_dict[i].loc[df_cluster_dict[i]['age_group'] == ag]
            if df_cluster_i_ag.empty:
                continue
            else:
                items_array_i = df_cluster_i_ag.values[:, :-non_item_col]
#                 print(ag, i, items_array_i.shape, 'items_array_i',items_array_i)
                for j in range(i, cluster_dim+1):
#                     print(i,j,ag,df_cluster_dict[j])
                    df_cluster_j_ag = df_cluster_dict[j].loc[df_cluster_dict[j]['age_group'] == ag]
#                     print(ag, j, df_cluster_j_ag.shape, 'df_cluster_j_ag',df_cluster_j_ag)
                    if df_cluster_j_ag.empty:
                        JSD_Mat[ag+(i-1)*age_group_size, ag+(j-1)*age_group_size] = default_max_JSD
                        JSD_Mat[ag+(j-1)*age_group_size, ag+(i-1)*age_group_size] = default_max_JSD
                    else:
                        items_array_j = df_cluster_j_ag.values[:, :-non_item_col]
                        for P in items_array_i:
                            for Q in items_array_j:
                                JSD_cluster.append(JSD(P, Q))
                        # if i == 10 and j ==10:
                            # print(ag, i, items_array_i.shape, 'items_array_i',items_array_i)
                            # print(ag, j, items_array_j.shape, 'items_array_j',items_array_j)
                            # print(np.mean(np.array(JSD_cluster)))
                        JSD_Mat[ag+(i-1)*age_group_size, ag+(j-1)*age_group_size] = np.mean(np.array(JSD_cluster))
                        JSD_Mat[ag+(j-1)*age_group_size, ag+(i-1)*age_group_size] = np.mean(np.array(JSD_cluster))
                        del JSD_cluster[:]
    
    return JSD_Mat


def cal_JSD_Matrix_withoutAgeGroup(df_cluster, cluster_dim, non_item_col):
    df_cluster_dict = {}
    for i in range(1, cluster_dim+1):
        df_cluster_dict[i] = df_cluster.loc[df_cluster['cluster'] == i]
    
    default_max_JSD = 1
    JSD_Mat = np.ones((cluster_dim, cluster_dim))*default_max_JSD
    
    JSD_cluster = []
    for i in range(1, cluster_dim+1): 
        if df_cluster_dict[i].empty:
            continue
        else:
            items_array_i = df_cluster_dict[i].values[:, :-non_item_col]
            for j in range(i, cluster_dim+1):
                if df_cluster_dict[j].empty:
                    continue
                else:
                    items_array_j = df_cluster_dict[j].values[:, :-non_item_col]
                    for P in items_array_i:
                        for Q in items_array_j:
                            ###
                            # if norm(P, ord=1) == 0 or norm(Q, ord=1)== 0:
                                # JSD_cluster.append(1)
                            # else:
                            JSD_cluster.append(JSD(P, Q))
                    JSD_Mat[i-1, j-1] = np.mean(np.array(JSD_cluster))
                    JSD_Mat[j-1, i-1] = np.mean(np.array(JSD_cluster))
                    del JSD_cluster[:]
    
    return JSD_Mat


def cal_JSD_Matrix(data, cluster_dim, non_item_col):
    data_length = data.shape[1] - non_item_col
    cluster_dict = {}
    for i in range(1, cluster_dim+1):
        if i not in cluster_dict:
            cluster_dict[i] = []

    for i in data:
        cluster_dict[i[-1]].append(i[:-non_item_col])

    clusters = []
    for i in range(1, cluster_dim+1):
        if len(cluster_dict[i]) == 0:
            clusters.append([0] * data_length)
        else:
            sum_before = np.sum(np.array(cluster_dict[i]).reshape((len(cluster_dict[i]), -1)), axis=0)
            print(i, np.sum(sum_before), sum_before.shape)
            clusters.append(sum_before / np.sum(sum_before))

    arr_list = []
    for i in range(cluster_dim):
        alist = []
        for j in range(cluster_dim):
            alist.append(JSD(clusters[i], clusters[j]))
        arr_list.append(alist)
    JSD_Mat = np.array(arr_list).reshape((cluster_dim, cluster_dim))

    return JSD_Mat


def cal_pgy_withAgeGroup(df_cluster, cluster_dim, age_group_size, age_list):
    user_size = df_cluster.shape[0]
    ageGroup_age_dict = {}
    for i in range(age_group_size):
        ageGroup_age_dict[i] = set(df_cluster.loc[df_cluster['age_group'] == i, 'age'])
    
    df_cluster_dict = {}
    for i in range(1, cluster_dim+1):
        df_cluster_dict[i] = df_cluster.loc[df_cluster['cluster'] == i, ['age', 'age_group', 'cluster']]
#         print(i, df_cluster_dict[i])
    
    pgy_Mat = np.zeros((len(age_list), cluster_dim* age_group_size))
    
    for i in range(1, cluster_dim+1):
        for ag in range(0, age_group_size):
            df_cluster_i_ag = df_cluster_dict[i].loc[df_cluster_dict[i]['age_group'] == ag]
            if df_cluster_i_ag.empty:
#                 age_group_len = len(ageGroup_age_dict[ag])
                for age in ageGroup_age_dict[ag]:
                    age_j = age - age_list[0]
                    pgy_Mat[age_j, ag+(i-1)*age_group_size] = 0.0000001
            else:
                group_age_cnt = df_cluster_i_ag.groupby(['age']).size().reset_index(name='count')
#                 print(group_age_cnt)
                group_size = df_cluster_i_ag.shape[0]
#                 print(group_size)
                for age in group_age_cnt['age']:
                    age_j = age - age_list[0]
                    pgy_Mat[age_j, ag+(i-1)*age_group_size] = group_age_cnt.loc[group_age_cnt['age'] == age,'count']/user_size
    return pgy_Mat


def cal_pgy_withoutAgeGroup(df_cluster, cluster_dim, age_list):
    user_size = df_cluster.shape[0]
    
    df_cluster_dict = {}
    for i in range(1, cluster_dim+1):
        df_cluster_dict[i] = df_cluster.loc[df_cluster['cluster'] == i, ['age', 'age_group', 'cluster']]
    
    pgy_Mat = np.zeros((len(age_list), cluster_dim))
    
    for i in range(1, cluster_dim+1):
        if df_cluster_dict[i].empty:
            for age in age_list:
                age_j = age - age_list[0]
                pgy_Mat[age_j, i-1] = 0.0000001
        else:
            group_age_cnt = df_cluster_dict[i].groupby(['age']).size().reset_index(name='count')
#                 print(group_age_cnt)
            for age in group_age_cnt['age']:
                age_j = age - age_list[0]
                pgy_Mat[age_j, i-1] = group_age_cnt.loc[group_age_cnt['age'] == age, 'count']/user_size
    return pgy_Mat


def cal_pgy_check_in_data(df_cluster, cluster_dim, grid_list):
    user_size = df_cluster.shape[0]
    df_cluster_dict = {}
    for i in range(1, cluster_dim+1):
        df_cluster_dict[i] = df_cluster.loc[df_cluster['cluster'] == i, ['grid', 'grid_group', 'cluster']]

    pgy_Mat = np.zeros((len(grid_list), cluster_dim))

    for i in range(1, cluster_dim+1):
        if df_cluster_dict[i].empty:
            for j in range(len(grid_list)):
                pgy_Mat[j, i-1] = 0.0000001
        else:
            group_grid_cnt = df_cluster_dict[i].groupby(['grid']).size().reset_index(name='count')
            for grid in group_grid_cnt['grid']:
                j = grid_list.index(grid)
                pgy_Mat[j, i-1] = group_grid_cnt.loc[group_grid_cnt['grid'] == grid, 'count']/user_size
    return pgy_Mat


def cal_pgy(data, cluster_dim):
    cluster_uid_dict = {}
    cluster_age_dict = {}
    for i in range(1, cluster_dim+1):
        if i not in cluster_uid_dict:
            cluster_uid_dict[i] = []
        if i not in cluster_age_dict:
            cluster_age_dict[i] = []

    for i in data:
        cluster_uid_dict[i[-1]].append(i[-3])
        cluster_age_dict[i[-1]].append(i[-2])

    cluster_age_count_dict = {}
    for i in range(1, cluster_dim+1):
        age_list = cluster_age_dict[i]
        count_dict = {}
        for j in range(18, 51):
            if j not in count_dict:
                count_dict[j] = 0
        for age in age_list:
            try:
                count_dict[age] = count_dict[age] + 1
            except:
                print(age)

        age_count_list = []
        for j in range(18, 51):
            age_count_list.append(count_dict[j])

        cluster_age_count_dict[i] = age_count_list

    total_num = len(data)
    pgy = []
    for i in range(1, cluster_dim+1):
        pgy.append(np.array(cluster_age_count_dict[i]) / total_num)
    pgy = np.array(pgy).reshape((cluster_dim, -1))

    return pgy


def f1_score(recall, precision):
    score = 2 * recall * precision / (recall + precision)
    return score


def RFC(X, Y, K=5):
    Y_pred = np.zeros(Y.shape)
    k_fold = KFold(K, random_state=10)
    for train, test in k_fold.split(X, Y):
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        model.fit(X[train], Y[train])
        Y_pred[test] = model.predict(X[test])
    accuracy = accuracy_score(Y, Y_pred)
    return accuracy


def XGBC(X, Y, K=5):
    Y_pred = np.zeros(Y.shape)
    k_fold = KFold(K, random_state=10)
    for train, test in k_fold.split(X, Y):
        model = xgb.XGBClassifier(n_estimators=40, random_state=0)
        model.fit(X[train], Y[train])
        Y_pred[test] = model.predict(X[test])
    accuracy = accuracy_score(Y, Y_pred)
    return accuracy


def grid_prediction_rf(X_obf, X_ori, alg):
    df_X_obf = pd.DataFrame.from_dict(X_obf).T
    df_x_obf_items = df_X_obf.values
    df_X_ori = pd.DataFrame.from_dict(X_ori).T
    df_x_ori_items = df_X_ori.values[:, :-2]
    df_x_y = df_X_ori.values[:, -2]

    acc_ori = RFC(df_x_ori_items, df_x_y, 2)
    acc_obf = RFC(df_x_obf_items, df_x_y, 2)

    return acc_ori, acc_obf


def grid_prediction_xgb(X_obf, X_ori, alg):
    df_X_obf = pd.DataFrame.from_dict(X_obf).T
    df_x_obf_items = df_X_obf.values
    df_X_ori = pd.DataFrame.from_dict(X_ori).T
    df_x_ori_items = df_X_ori.values[:, :-2]
    df_x_y = df_X_ori.values[:, -2]

    acc_ori = XGBC(df_x_ori_items, df_x_y, 2)
    acc_obf = XGBC(df_x_obf_items, df_x_y, 2)

    return acc_ori, acc_obf

