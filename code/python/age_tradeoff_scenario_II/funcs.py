import numpy as np
import pandas as pd
import copy
import xgboost as xgb
from scipy.stats import entropy
from numpy.linalg import norm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import normalize


def update_age_group(df_train, age_group_dict):
    for k, v in age_group_dict.items():
        df_train.loc[df_train['age'] == k, 'age_group'] = v
    return df_train


def age_group_adjust_greedy(df_train, group_age_dict, k, l):
    reducible_groups = {}
    for group in group_age_dict:
        left_range, right_range = reducible_scale(df_train, group_age_dict[group], k, l)
        if (left_range > 0 and group > 0) or (right_range > 0 and group < len(group_age_dict) - 1):
            reducible_groups[group] = [left_range, right_range]

    adjustable_groups = {}
    #### selecting the best move
    j = 0
    for reduce_group in reducible_groups:
        left_range, right_range = reducible_groups[reduce_group]
        if reduce_group == 0:
            for reduce_range in range(1, right_range + 1):
                adjustable_groups[j] = copy.deepcopy(group_age_dict)
                age_move = adjustable_groups[j][reduce_group][-reduce_range:]
                #                 print(j, adjustable_groups[j],adjustable_groups[j][reduce_group][:-reduce_range],adjustable_groups[j][reduce_group+1],age_move)
                adjustable_groups[j][reduce_group] = adjustable_groups[j][reduce_group][:-reduce_range]
                adjustable_groups[j][reduce_group + 1].extend(age_move)
                adjustable_groups[j][reduce_group + 1].sort()
                del age_move[:]
                j = j + 1
        elif reduce_group == len(group_age_dict) - 1:
            for reduce_range in range(1, left_range + 1):
                adjustable_groups[j] = copy.deepcopy(group_age_dict)
                age_move = group_age_dict[reduce_group][:reduce_range]
                adjustable_groups[j][reduce_group] = adjustable_groups[j][reduce_group][reduce_range:]
                adjustable_groups[j][reduce_group - 1].extend(age_move)
                adjustable_groups[j][reduce_group - 1].sort()
                del age_move[:]
                j = j + 1
        else:
            for reduce_range in range(1, right_range + 1):
                adjustable_groups[j] = copy.deepcopy(group_age_dict)
                age_move = group_age_dict[reduce_group][-reduce_range:]
                adjustable_groups[j][reduce_group] = adjustable_groups[j][reduce_group][:-reduce_range]
                adjustable_groups[j][reduce_group + 1].extend(age_move)
                adjustable_groups[j][reduce_group + 1].sort()
                del age_move[:]
                j = j + 1
            for reduce_range in range(1, left_range + 1):
                adjustable_groups[j] = copy.deepcopy(group_age_dict)
                age_move = group_age_dict[reduce_group][:reduce_range]
                adjustable_groups[j][reduce_group] = adjustable_groups[j][reduce_group][reduce_range:]
                adjustable_groups[j][reduce_group - 1].extend(age_move)
                adjustable_groups[j][reduce_group - 1].sort()
                del age_move[:]
                j = j + 1

    #     for group,group_age_list in group_age_dict.items():
    #         for age in group_age_list:
    #             age_group_dict[age] = group

    return adjustable_groups, reducible_groups


def reducible_scale(df_train, group_age_list, k, l):
    l_cur = l_diversity(df_train, group_age_list)
    range = int(np.exp(l_cur) - np.exp(l))
    group_age_list.sort()
    left_range = range
    right_range = range
    while left_range > 0:
        group_age_list_left = group_age_list[left_range:]
        k_left = k_anonymity(df_train, group_age_list_left)
        if k_left >= k:
            break
        else:
            left_range = left_range - 1

    while right_range > 0:
        group_age_list_right = group_age_list[:-right_range]
        k_right = k_anonymity(df_train, group_age_list_right)
        if k_right >= k:
            break
        else:
            right_range = right_range - 1

    return left_range, right_range


def k_anonymity(df_train, group_age_list):
    return df_train.loc[df_train['age'].isin(group_age_list)].shape[0]


def l_diversity(df_train, group_age_list):
    age_dtr = np.zeros(len(group_age_list))
    for i in range(len(group_age_list)):
        age_dtr[i] = df_train.loc[df_train['age'] == group_age_list[i]].shape[0]
    age_dtr_norm = age_dtr / norm(age_dtr, ord=1)
    return entropy(age_dtr_norm)


def get_obf_X(df_cluster, xpgg, pp):
    user_cluster_dict = {}
    cluster_vec = {}
    user_ageGroup_dict = {}
    ageGroup_vec = {}
    user_size = df_cluster.shape[0]
    for i in range(user_size):
        user_id = df_cluster['uid'][i]
        cluster_id = df_cluster['cluster'][i]
        user_cluster_dict[user_id] = cluster_id
        if cluster_id in cluster_vec:
            cluster_vec[cluster_id].append(user_id)
        else:
            cluster_vec[cluster_id] = [user_id]

        age_group = df_cluster['age_group'][i]
        user_ageGroup_dict[user_id] = age_group
        if age_group in ageGroup_vec:
            ageGroup_vec[age_group].append(user_id)
        else:
            ageGroup_vec[age_group] = [user_id]

    cluster_size = len(cluster_vec)
    ageGroup_size = len(ageGroup_vec)

    X_obf = {}
    X_ori = {}

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
                change_index = np.random.choice(range(0, cluster_size * ageGroup_size), 1, p=xpgg_norm[:,
                                                                                             user_ageGroup_dict[
                                                                                                 user_id] + (
                                                                                                         user_cluster_dict[
                                                                                                             user_id] - 1) * ageGroup_size])[
                    0]
                change_cluster_index = int(change_index / ageGroup_size) + 1
                change_ageGroup_index = change_index % ageGroup_size
                potential_users = list(
                    set(cluster_vec[change_cluster_index]) & set(ageGroup_vec[change_ageGroup_index]))
                if len(potential_users) > 0:  # potential users may be empty by a slight probability
                    break
                else:
                    print("not find potential users, re-pick")
            uidx = np.random.choice(potential_users, 1)[0]
            X_obf[user_id] = df_cluster[df_cluster['uid'] == uidx].values[0, :-3]
        else:
            X_obf[user_id] = df_cluster[df_cluster['uid'] == user_id].values[0, :-3]
    return X_obf, X_ori


def get_frapp_obf_X(df_train, gamma):
    X_ori = {}
    X_obf = {}

    user_size = df_train.shape[0]
    uid_list = list(df_train['uid'].values)

    e = 1 / (gamma + user_size - 1)
    pfrapp = gamma * e
    for i in range(user_size):
        user_id = df_train['uid'][i]

        p_list = [e] * user_size
        p_list[uid_list.index(user_id)] = pfrapp

        # get X_ori
        X_ori[user_id] = df_train[df_train['uid'] == user_id].values[0, :-1]

        # get X_obf
        uidx = np.random.choice(uid_list, 1, p=p_list)[0]
        X_obf[user_id] = df_train[df_train['uid'] == uidx].values[0, :-3]

    return X_obf, X_ori


def get_DP_obf_X(df_train, dist_matrix, beta):
    X_ori = {}
    X_obf = {}
    user_size = df_train.shape[0]
    uid_list = list(df_train['uid'].values)
    for i in range(user_size):
        user_id = df_train['uid'][i]
        # get X_ori
        X_ori[user_id] = df_train[df_train['uid'] == user_id].values[0, :-1]

        # get X_obf
        dist_arr = np.array(list(dist_matrix[i]))
        dp_arr = np.exp(-beta*dist_arr)
        prob_list = list(dp_arr / sum(dp_arr))
        uidx = np.random.choice(uid_list, 1, p=prob_list)[0]
        X_obf[user_id] = df_train[df_train['uid'] == uidx].values[0, :-3]

    return X_obf, X_ori


def get_similarity_obf_X(sim_mat, df_train, pp):
    user_size = df_train.shape[0]
    uid_list = list(df_train['uid'].values)
    prob_dict = {}

    # get obfuscation probability
    for i in range(user_size):
        sim_array = sim_mat[i]
        sim_array[i] = 0
        prob_dict[i] = sim_array/sum(sim_array)

    X_ori = {}
    X_obf = {}

    obf_flag = np.random.choice([0, 1], 1, p=[pp, 1-pp])

    for i in range(user_size):
        user_id = df_train['uid'][i]
        # get X_ori
        X_ori[user_id] = df_train[df_train['uid']==user_id].values[0, :-1]
        # get X_obf
        if obf_flag == 1:
            uidx = np.random.choice(uid_list, 1, p=prob_dict[i])[0]
            X_obf[user_id] = df_train[df_train['uid']==uidx].values[0, :-3]
        else:
            X_obf[user_id] = df_train[df_train['uid']==user_id].values[0, :-3]

    return X_obf, X_ori


def get_random_obf_X(df_train, p_rand):
    X_ori = {}
    X_obf = {}

    user_size = df_train.shape[0]
    uid_list = list(df_train['uid'].values)

    for i in range(user_size):
        user_id = df_train['uid'][i]

        # get X_ori
        X_ori[user_id] = df_train[df_train['uid']==user_id].values[0, :-1]

        # get X_obf
        flag = np.random.choice([0, 1], 1, p=[1-p_rand, p_rand])[0]
        if flag == 0:
            X_obf[user_id] = df_train[df_train['uid']==user_id].values[0, :-3]
        else:
            ul = [user_id]
            uidx = np.random.choice(list(set(uid_list) - set(ul)), 1)[0]
            X_obf[user_id] = df_train[df_train['uid']==uidx].values[0, :-3]

    return X_obf, X_ori


def Kmeans_clustering(test_df, cluster_num, item_size):
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(test_df.values[:, :item_size])
    P = kmeans.labels_
    dfn = test_df.copy()
    dfn['cluster'] = P+1
    return dfn


def hierarchical_clustering(test_df, cluster_num, item_size, affinity_type, linkage_type):
    hierarchy_cluster = AgglomerativeClustering(affinity=affinity_type, n_clusters=cluster_num, linkage=linkage_type).fit(test_df.values[:, :item_size])
    P = hierarchy_cluster.labels_
    dfn = test_df.copy()
    dfn['cluster'] = P + 1
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


def XGboost_acc(X, Y, threshold_list, K=5):
    Y_pred = np.zeros(Y.shape)
    k_fold = KFold(K, random_state=10)
    for train, test in k_fold.split(X, Y):
        model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, random_state=0)
        model.fit(X[train], Y[train])
        Y_pred[test] = model.predict(X[test])

    mae = mean_absolute_error(Y, Y_pred)

    return mae


def RandomForest_acc(X, Y, threshold_list, K=5):
    Y_pred = np.zeros(Y.shape)
    k_fold = KFold(K, random_state=10)
    for train, test in k_fold.split(X, Y):
        model = RandomForestRegressor(n_estimators=20, random_state=0)
        model.fit(X[train], Y[train])
        Y_pred[test] = model.predict(X[test])

    mae = mean_absolute_error(Y, Y_pred)

    return mae


def Age_prediction_accuracy_rf(X_obf, X_ori, threshold_list):
    df_X_obf = pd.DataFrame.from_dict(X_obf).T
    df_x_obf_items = df_X_obf.values
    df_X_ori = pd.DataFrame.from_dict(X_ori).T
    df_x_ori_items = df_X_ori.values[:, :-2]
    df_x_y = df_X_ori.values[:, -2]

    mae_ori = RandomForest_acc(df_x_ori_items, df_x_y, threshold_list, 2)
    mae_obf = RandomForest_acc(df_x_obf_items, df_x_y, threshold_list, 2)

    return mae_ori, mae_obf


def Age_prediction_accuracy_xgb(X_obf, X_ori, threshold_list):
    df_X_obf = pd.DataFrame.from_dict(X_obf).T
    df_x_obf_items = df_X_obf.values
    df_X_ori = pd.DataFrame.from_dict(X_ori).T
    df_x_ori_items = df_X_ori.values[:, :-2]
    df_x_y = df_X_ori.values[:, -2]

    mae_ori = XGboost_acc(df_x_ori_items, df_x_y, threshold_list, 2)
    mae_obf = XGboost_acc(df_x_obf_items, df_x_y, threshold_list, 2)

    return mae_ori, mae_obf
