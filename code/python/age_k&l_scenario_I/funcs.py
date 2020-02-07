import numpy as np
import xgboost as xgb
from scipy.stats import entropy
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor


def Kmeans_clustering(test_df, cluster_num, item_size):
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(test_df.values[:, :item_size])
    P = kmeans.labels_
    dfn = test_df.copy()
    dfn['cluster'] = P+1
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


def train_rf_model(df):
    Y = list(df['age'])
    cols = list(df.columns)
    cols.remove('age')
    cols.remove('uid')
    cols.remove('cluster')
    X = df[cols].values

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, Y)

    return model


def train_xgb_model(df):
    Y = list(df['age'])
    cols = list(df.columns)
    cols.remove('age')
    cols.remove('uid')
    cols.remove('cluster')
    X = df[cols].values

    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, random_state=0)
    model.fit(X, Y)

    return model
