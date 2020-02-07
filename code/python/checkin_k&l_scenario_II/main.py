import pandas as pd
import random
import funcs
import os
import matlab.engine
import numpy as np
import copy
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from scipy.stats import entropy


def read_data():
    df = pd.read_csv('../../../check_in_data/check_in_data_3users.csv')
    return df


def get_JSD_PGY(df_cluster, area_grid_dict):
    grid_area_number = len(list(area_grid_dict.keys()))
    group_usersize_dict = {}
    group_grid_size = {}
    group_grid_size[0] = 0
    for gg in range(0, grid_area_number):
        df_cluster_gg = df_cluster.loc[df_cluster['grid_group'] == gg]
        grid_list_gg = area_grid_dict[gg]
        grid_list_gg.sort()        
        if gg > 0:
            group_grid_size[gg] = len(area_grid_dict[gg-1])+group_grid_size[gg-1]
        group_usersize_dict[gg] = df_cluster_gg.shape[0]
    
        JSD_Mat_dict[gg] = cal_JSD_Matrix_withoutGridGroup(df_cluster_gg, cluster_num, 4)
        pgy_dict[gg] = cal_pgy_withoutGridGroup(df_cluster_gg, cluster_num, grid_list_gg)
        pd.DataFrame(JSD_Mat_dict[gg]).to_csv('tmp/JSDM_check_in_'+str(gg)+'.csv', index=False, header=None)
        pd.DataFrame(pgy_dict[gg]).to_csv('tmp/pgy_check_in_'+str(gg)+'.csv', index=False, header=None)
    
    for gg in range(grid_area_number):
        grid_list_gg = area_grid_dict[gg]
        grid_list_gg.sort()
        for grid_index in range(len(grid_list_gg)):
            for col in range(cluster_num):
                if gg == 0:
                    row_num = grid_index
                else:
                    row_num = grid_index + group_grid_size[gg]
                pgy[row_num, gg + col * grid_area_number] = pgy_dict[gg][grid_index, col] * group_usersize_dict[gg] / df_cluster.shape[0]
                 
    for gg in range(grid_area_number):
        for row in range(cluster_num):
            for col in range(cluster_num):
                JSD_Mat[gg + row * grid_area_number, gg + col * grid_area_number] = JSD_Mat_dict[gg][row, col]
                
    return JSD_Mat, pgy, JSD_Mat_dict, pgy_dict
     

def get_grid_loc(grid_list):
    grid_rowcol = {}
    grid_colrow = {}
    
    rows = int(max(np.array(grid_list)) / 60) + 1
    grid_loc_mat = np.zeros((rows, 60))
    for grid in grid_list:
        lan = int(grid / 60)
        lon = grid % 60
        if lan in grid_rowcol:
            grid_rowcol[lan].append(lon)
        else:
            grid_rowcol[lan] = [lon]
        if lon in grid_colrow:
            grid_colrow[lon].append(lan)
        else:
            grid_colrow[lon] = [lan]
        grid_loc_mat[lan, lon] = 1

    return grid_colrow, grid_rowcol


def area_initiate(grid_loc, area_number):
    area_grid_dict = {}
    area_grid_rowcol_dict = {}
    area_grid_colrow_dict = {}
    grid_area_dict = {}
    area_reducibility = {}

    area_rows = int(np.sqrt(area_number))
    extra_cols = area_number - area_rows * area_rows
    grid_rows = list(grid_loc.keys())
    grid_rows.sort()

    area = -1
    area_row_len = 1
    area_row_lens = []
    for r in range(area_rows):
        if r < area_rows - 1:
            area_row_len = int(len(grid_rows) / area_rows)
        else:
            area_row_len = int(len(grid_rows) - (area_rows - 1) * area_row_len)
        area_row_lens.append(area_row_len)
        #         print(area_row_len)
        if extra_cols > 0:
            area_cols = area_rows + 1
            extra_cols = extra_cols - 1
        else:
            area_cols = area_rows
        for c in range(area_cols):
            area = area + 1
            area_grid_rowcol_dict[area] = {}
            area_reducibility[area] = [0, 0, 0, 0]
            if r > 0:
                area_reducibility[area][2] = 1
            if r < area_rows - 1:
                area_reducibility[area][3] = 1
            if c > 0:
                area_reducibility[area][1] = 1
            if c < area_cols - 1:
                area_reducibility[area][0] = 1
            for i in range(area_row_len):
                cur_row = grid_rows[r * area_row_lens[r - 1] + i]
                area_col_len = int(len(grid_loc[cur_row]) / area_cols)
                if c < area_cols - 1:
                    area_grid_rowcol_dict[area][cur_row] = grid_loc[cur_row][c * area_col_len:(c + 1) * area_col_len]
                else:
                    area_grid_rowcol_dict[area][cur_row] = grid_loc[cur_row][(area_cols - 1) * area_col_len:]

    for i in area_grid_rowcol_dict:
        for j in area_grid_rowcol_dict[i]:
            for k in area_grid_rowcol_dict[i][j]:
                grid_code = j * 60 + k
                grid_area_dict[grid_code] = i
                if i in area_grid_dict:
                    area_grid_dict[i].append(grid_code)
                    if k in area_grid_colrow_dict[i]:
                        area_grid_colrow_dict[i][k].append(j)
                    else:
                        area_grid_colrow_dict[i][k] = [j]
                else:
                    area_grid_dict[i] = [grid_code]
                    area_grid_colrow_dict[i] = {}
                    area_grid_colrow_dict[i][k] = [j]
    return area_grid_rowcol_dict, area_grid_colrow_dict, area_grid_dict, grid_area_dict, area_reducibility


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


def grid_to_rowcol(grid):
    row = int(grid/60)
    col = grid % 60
    return row,col


def rowcol_to_grid(row, col):
    return row*60 + col


def colrow_to_grid(col, row):
    return row*60 + col


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
        obf_flag = np.random.choice([0, 1], 1, p=[pp, 1-pp])
        if obf_flag == 1:
            # selecting one cluster to change
            while (True):
                change_index = np.random.choice(range(0, cluster_size*gridGroup_size), 1, p=xpgg_norm[:, int(user_gridGroup_dict[user_id] + (user_cluster_dict[user_id] - 1)*gridGroup_size)])[0]
                change_cluster_index = int(change_index/gridGroup_size) + 1
                change_ageGroup_index = change_index % gridGroup_size
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


def cal_recall(rec_list, true_list):
    rec_true_num = 0
    for i in rec_list:
        if i in true_list:
            rec_true_num += 1

    return rec_true_num / len(true_list)


def cal_precision(rec_list, true_list):
    rec_true_num = 0
    for i in rec_list:
        if i in true_list:
            rec_true_num += 1

    return rec_true_num / len(rec_list)


def cal_map(rec_list, true_list):
    cal_list = []
    all_precisions = []
    for i in rec_list:
        cal_list.append(i)
        if i in true_list:
            recall_now = cal_recall(cal_list, true_list)
            precision_now = cal_precision(cal_list, true_list)
            all_precisions.append(precision_now)
            if recall_now == 1:
                break
    if len(all_precisions) == 0:
        print("None all precisions")
    avg_precision = np.mean(all_precisions)

    return avg_precision


def recommendation(X_obf, X_ori, df_test):
    df_X_obf = pd.DataFrame.from_dict(X_obf).T
    uids = list(df_X_obf.index)
    df_X_obf.reset_index(drop=True, inplace=True)
    df_X_obf['uid'] = uids
    df_X_obf.set_index('uid', inplace=True)

    df_X_ori = pd.DataFrame.from_dict(X_ori).T
    df_X_ori.reset_index(drop=True, inplace=True)
    uids = df_X_ori[df_X_ori.columns[-1]]
    df_X_ori.drop(df_X_ori.columns[-2:], axis=1, inplace=True)
    df_X_ori['uid'] = uids
    df_X_ori.set_index('uid', inplace=True)

    users = list(X_obf.keys())
    random.seed(10)
    random.shuffle(users)
    user_train = users[:int(len(users)*0.7)]
    user_test = list(set(users) - set(user_train))

    # obf recommendation
    print("obf recommendation...")
    df_obf_trainUser_KnownItem = df_X_obf.drop(user_test)
    df_obf_testUser_KnownItem = df_X_obf.drop(user_train)

    # sim_matrix of trainUser and testUser
    similarity_matrix = cosine_similarity(df_obf_trainUser_KnownItem.values, df_obf_testUser_KnownItem.values)

    # store similarities in a dict
    tr_user = list(df_obf_trainUser_KnownItem.index)
    te_user = list(df_obf_testUser_KnownItem.index)
    sim_dict = {}
    for i in range(len(te_user)):
        sim_dict[te_user[i]] = {}
        for j in range(len(tr_user)):
            sim_dict[te_user[i]][tr_user[j]] = similarity_matrix[j, i]

    df_test_n = df_test.drop(['grid_group', 'grid', 'cluster'], axis=1)
    df_test_n = df_test_n.set_index('uid')

    testItem_trainUser = df_test_n.loc[user_train]
    testItem_testUser = df_test_n.loc[user_test]

    train_u_testItem_dict = {}
    for u in user_train:
        train_u_testItem_dict[u] = []
        idx = testItem_trainUser.loc[u].index
        value = testItem_trainUser.loc[u].values
        for i in zip(idx, value):
            if i[1] != 0:
                train_u_testItem_dict[u].append(i[0])

    test_u_testItem_dict = {}
    for u in user_test:
        test_u_testItem_dict[u] = []
        idx = testItem_testUser.loc[u].index
        value = testItem_testUser.loc[u].values
        for i in zip(idx, value):
            if i[1] != 0:
                test_u_testItem_dict[u].append(i[0])

    # start obf recommendation
    obf_maps = []
    for ui in user_test:
        score_dict = {}

        for uj in user_train:
            train_u_items = train_u_testItem_dict[uj]
            for item in train_u_items:
                if item not in score_dict:
                    score_dict[item] = sim_dict[ui][uj]
                else:
                    score_dict[item] += sim_dict[ui][uj]
        # recommend all resorted test items
        sorted_items = [i[0] for i in sorted(score_dict.items(), key=lambda x: x[1], reverse=True)]

        ui_items = test_u_testItem_dict[ui]

        # skip the 0 test_item user
        if len(ui_items) == 0:
            continue
        u_map = cal_map(sorted_items, ui_items)
        obf_maps.append(u_map)

    avg_obf_map = np.mean(obf_maps)


    # ori recommendation
    print("ori recommendation...")
    df_ori_trainUser_KnownItem = df_X_ori.drop(user_test)
    df_ori_testUser_KnownItem = df_X_ori.drop(user_train)

    # ori sim_matrix of trainUser and testUser
    similarity_matrix_ori = cosine_similarity(df_ori_trainUser_KnownItem.values, df_ori_testUser_KnownItem.values)

    # store similarities in a dict
    tr_user_ori = list(df_ori_trainUser_KnownItem.index)
    te_user_ori = list(df_ori_testUser_KnownItem.index)
    sim_dict_ori = {}
    for i in range(len(te_user_ori)):
        sim_dict_ori[te_user_ori[i]] = {}
        for j in range(len(tr_user_ori)):
            sim_dict_ori[te_user_ori[i]][tr_user_ori[j]] = similarity_matrix_ori[j, i]

    # start ori recommendation
    ori_maps = []
    for ui in user_test:
        score_dict = {}

        for uj in user_train:
            train_u_items = train_u_testItem_dict[uj]
            for item in train_u_items:
                if item not in score_dict:
                    score_dict[item] = sim_dict_ori[ui][uj]
                else:
                    score_dict[item] += sim_dict_ori[ui][uj]
        # recommend all resorted test items
        sorted_items = [i[0] for i in sorted(score_dict.items(), key=lambda x: x[1], reverse=True)]

        ui_items = test_u_testItem_dict[ui]
        # skip the 0 test_item user
        if len(ui_items) == 0:
            continue
        u_map = cal_map(sorted_items, ui_items)
        ori_maps.append(u_map)

    avg_ori_map = np.mean(ori_maps)

    return avg_ori_map, avg_obf_map


def cal_JSD_Matrix_withoutGridGroup(df_cluster, cluster_dim, non_item_col):
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
                        if np.sum(P) > 0:
                            for Q in items_array_j:
                                if np.sum(Q) > 0:
                                    JSD_cluster.append(funcs.JSD(P, Q))
                    mean_JSD = np.mean(np.array(JSD_cluster))
                    mean_JSD = mean_JSD if mean_JSD > 0 else 0.000001
                    JSD_Mat[i-1, j-1] = mean_JSD
                    JSD_Mat[j-1, i-1] = mean_JSD
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
                    pgy_Mat[j, i - 1] = group_grid_cnt.loc[group_grid_cnt['grid'] == grid, 'count'] / user_size
                j += 1
    return pgy_Mat


if __name__ == '__main__':

    # HyObscure settings for k&l experiments

    deltaX = 0.6
    pp = 0
    cluster_num = 8
    grid_area_number = 4
    k_threshold_list = [20, 40, 60, 80]
    l_threshold_list = [3, 5, 7, 9]


    # clustering and area initialization
    df = read_data()
    grid_list = list(set(df['grid'].values))
    grid_list.sort()
    grid_colrow, grid_rowcol = get_grid_loc(grid_list)

    print("initiate area...")
    area_grid_rowcol_dict, area_grid_colrow_dict, area_grid_dict, grid_area_dict, area_reducibility = area_initiate(
        grid_rowcol, grid_area_number)

    ### add grid groups
    df['grid_group'] = pd.Series(np.zeros(df.shape[0]), index=df.index, dtype='int32')
    df_grid_group = update_grid_group(df, grid_area_dict)
    cols = list(df.columns.values)
    cols_change = cols[:-3]
    cols_change.extend(['grid_group', 'grid', 'uid'])
    df_item_gridGroup_uid = df_grid_group[cols_change]

    df_cluster = funcs.hierarchical_clustering(df_item_gridGroup_uid, cluster_num, -3, 'cosine', 'complete')
    drop_idx = df_cluster.loc[df_cluster['cluster']==5].index
    df_cluster.drop(drop_idx, inplace=True)
    drop_idx = []
    for i in df_cluster.grid.value_counts().index:
        if df_cluster.grid.value_counts()[i] < 3:
            drop_idx.extend(df_cluster.loc[df_cluster['grid']==i].index)
    df_cluster.drop(drop_idx, inplace=True)
    df_cluster.reset_index(inplace=True, drop=True)
    cluster_col = []
    for i in df_cluster.cluster:
        if i == 6:
            cluster_col.append(5)
        elif i == 7:
            cluster_col.append(6)
        elif i == 8:
            cluster_col.append(7)
        else:
            cluster_col.append(i)
    df_cluster.cluster = cluster_col
    cluster_num = len(set(df_cluster.cluster))

    if os.path.exists('tmp'):
        pass
    else:
        os.makedirs('tmp')


    for l_threshold in l_threshold_list:
        for k_threshold in k_threshold_list:

            results = {}
            results['acc_ori_rf'] = []
            results['acc_obf_rf'] = []
            results['acc_ori_xgb'] = []
            results['acc_obf_xgb'] = []
            results['rec_ori'] = []
            results['rec_obf'] = []

            for r in range(20):
                area_grid_rowcol_dict, area_grid_colrow_dict, area_grid_dict, grid_area_dict, area_reducibility = area_initiate(
                    grid_rowcol, grid_area_number)
                df_cluster = update_grid_group(df_cluster, grid_area_dict)

                random.seed(r*10)
                print("run seed {}...".format(r))

                items = list(df)
                items.remove('grid')
                items.remove('grid_group')
                items.remove('uid')
                random.shuffle(items)
                items_train = items[:int(len(items) * 0.8)]
                items_test = list(set(items) - set(items_train))

                df_train = df_cluster.drop(items_test, axis=1)
                df_test = df_cluster.drop(items_train, axis=1)

                df_train_copy = copy.deepcopy(df_train)
                df_train_copy['grid_group'] = pd.Series(np.zeros(df_train_copy.shape[0]), index=df_train_copy.index,
                                                         dtype='int32')
                user_num = df_train_copy.shape[0]
                X_ori = {}
                for k in range(user_num):
                    user_id = df_train_copy['uid'][k]
                    X_ori[user_id] = df_train_copy[df_train_copy['uid'] == user_id].values[0, :-1]
                for k in X_ori.keys():
                    user_grid = X_ori[k][-2]
                    X_ori[k][-3] = grid_area_dict[user_grid]

                for i in area_grid_dict:
                    print("user number in area ", i, " is ", k_anonymity(df_cluster, area_grid_dict[i]))

                print("start solving xpgg...")
                xpgg = np.ones((cluster_num * grid_area_number, cluster_num * grid_area_number)) * 0.00000001
                JSD_Mat = np.ones((cluster_num * grid_area_number, cluster_num * grid_area_number))
                pgy = np.ones((len(grid_list), cluster_num * grid_area_number)) * 0.00000001

                JSD_Mat_dict = {}
                pgy_dict = {}

                for op in range(0, 6):
                    ## compute JSD and pgy
                    JSD_Mat, pgy, JSD_Mat_dict, pgy_dict = get_JSD_PGY(df_train, area_grid_dict)
                    print('op:', op)
                    grid_xpgg_dict = {}
                    ## compute xpgg
                    for gg in range(0, grid_area_number):
                        eng = matlab.engine.start_matlab()
                        eng.edit('../../matlab/checkin_k&l_scenario_II/HyObscure', nargout=0)
                        eng.cd('../../matlab/checkin_k&l_scenario_II')
                        grid_xpgg_dict[gg] = np.array(eng.HyObscure(deltaX, gg))

                        for row in range(cluster_num):
                            for col in range(cluster_num):
                                xpgg[gg + row * grid_area_number, gg + col * grid_area_number] = grid_xpgg_dict[gg][
                                    row, col]

                    mean_Utility = funcs.Mean_JSD(JSD_Mat, xpgg)
                    mean_Privacy = funcs.Mean_KL_div(pgy, xpgg)
                    min_mean_Utility = mean_Utility
                    min_mean_Privacy = mean_Privacy
                    ## area_grid_rowcol_dict, area_grid_colrow_dict, area_grid_dict, grid_area_dict, area_reducibility
                    areas = list(area_grid_dict.keys())
                    random.shuffle(areas)
                    ### change grid group (area) by stochastic privacy-utility boosting
                    for area_code in areas:  ##select one area to adjust
                        area_grids = area_grid_dict[area_code]  ## get all the grids in the area

                        l_cur = l_diversity(df_cluster, area_grids)  ## check l diversity
                        l_range = int(np.exp(l_cur) - np.exp(np.log(l_threshold)))
                        print('start adjusting area: ', area_code)
                        if l_range > 0:
                            ### select one direction to adjust: left (0); right (1); up (2); down(3)
                            d = np.random.choice([0, 1, 2, 3],
                                                 p=area_reducibility[area_code] / np.sum(area_reducibility[area_code]))
                            # the selected area can be reduced through the selected direction
                            if d < 2:  ## change left or right
                                area_grid_line_list_dict = area_grid_rowcol_dict
                                line_list_to_grid = rowcol_to_grid
                                grid_linelist = grid_rowcol
                            else:  ## change up or down
                                area_grid_line_list_dict = area_grid_colrow_dict
                                line_list_to_grid = colrow_to_grid
                                grid_linelist = grid_colrow
                            area_lines = list(area_grid_line_list_dict[area_code].keys())
                            area_lines.sort()
                            for line in area_lines:
                                # recheck area l diversity
                                area_grids = area_grid_dict[area_code]  ## get all the grids in the area
                                l_cur = l_diversity(df_cluster, area_grids)  ## check l diversity
                                l_range = int(np.exp(l_cur) - np.exp(np.log(l_threshold)))

                                change_range = l_range
                                line_lists = area_grid_line_list_dict[area_code][line]
                                line_lists.sort()
                                line_lists_len = len(line_lists)
                                if change_range > line_lists_len:
                                    change_range = line_lists_len
                                for i in range(1, change_range + 1):
                                    if d == 0 or d == 3:
                                        moveout_grid_lists = line_lists[-i:]
                                    elif d == 1 or d == 2:
                                        moveout_grid_lists = line_lists[:i]
                                    moveout_grids = []
                                    for mgc in moveout_grid_lists:
                                        moveout_grids.append(line_list_to_grid(line, mgc))
                                    adjusted_area_grids = list(set(area_grids) - set(moveout_grids))

                                    ## check k anonymity
                                    k_adjust = k_anonymity(df_cluster, adjusted_area_grids)

                                    ## the adjusted schema meets both k-anonymity and l-diversity
                                    if k_adjust >= k_threshold:
                                        if d == 0:
                                            to_area = area_code + 1
                                        elif d == 1:
                                            to_area = area_code - 1
                                        elif d == 2:
                                            to_area = area_code - int(grid_area_number / int(np.sqrt(grid_area_number)))
                                        elif d == 3:
                                            to_area = area_code + int(grid_area_number / int(np.sqrt(grid_area_number)))

                                        ## adjust grid groups (areas): update area_grid_dict and grid_area_dict
                                        area_grid_dict_cur = copy.deepcopy(area_grid_dict)
                                        adjusted_area_grids.sort()
                                        area_grid_dict_cur[area_code] = adjusted_area_grids
                                        area_grid_dict_cur[to_area] = list(
                                            set(area_grid_dict_cur[to_area]) | set(moveout_grids))
                                        area_grid_dict_cur[to_area].sort()
                                        grid_area_dict_cur = copy.deepcopy(grid_area_dict)
                                        for grid in moveout_grids:
                                            grid_area_dict_cur[grid] = to_area

                                        for i in area_grid_dict_cur:
                                            print("area:", i, "grid number:", len(area_grid_dict_cur[i]))


                                        print('from area: ', area_code, 'to area: ', to_area, 'change line: ', line,
                                              'moveout_grids: ', moveout_grids)

                                        df_train_new = update_grid_group(df_train, grid_area_dict_cur)
                                        # try:
                                        new_JSD_Mat, new_pgy, new_JSD_Mat_dict, new_pgy_dict = get_JSD_PGY(df_train_new,
                                                                                                           area_grid_dict_cur)

                                        new_mean_Utility = funcs.Mean_JSD(new_JSD_Mat, xpgg)
                                        new_mean_Privacy = funcs.Mean_KL_div(new_pgy, xpgg)

                                        if new_mean_Privacy < min_mean_Privacy and new_mean_Utility < min_mean_Utility:
                                            min_mean_Utility = new_mean_Utility
                                            min_mean_Privacy = new_mean_Privacy
                                            min_grid_area_dict = grid_area_dict_cur
                                            min_area_grid_dict = area_grid_dict_cur
                                            min_df_train = df_train_new

                                            grid_area_dict = min_grid_area_dict
                                            area_grid_dict = min_area_grid_dict
                                            df_train = min_df_train
                                            min_distortion_budget = min_mean_Utility
                                            area_grid_rowcol_dict, area_grid_colrow_dict = update_rowcol_colrow_dict(
                                                area_grid_dict)
                                            print("! Find a better area group")
                                            break

                                        print(op, area_code, to_area, line, mgc, mean_Privacy, mean_Utility,
                                              min_mean_Privacy,
                                              min_mean_Utility, new_mean_Privacy, new_mean_Utility)

                                    else:
                                        print("*** area not meet k_anonymity requirement")


                        else:
                            print("*** area not meet l_diversity requirement")

                df_train = update_grid_group(df_train, grid_area_dict)

                X_obf_dict = {}
                for i in range(25):
                    X_obf_dict[i], _ = get_obf_X(df_train, xpgg, pp)

                rec_oris = []
                rec_obfs = []
                acc_oris_rf = []
                acc_obfs_rf = []
                acc_oris_xgb = []
                acc_obfs_xgb = []

                df_test = update_grid_group(df_test, grid_area_dict)

                for i in range(25):
                    map_ori, map_obf = recommendation(X_obf_dict[i], X_ori, df_test)
                    rec_oris.append(map_ori)
                    rec_obfs.append(map_obf)

                    acc_ori_rf, acc_obf_rf = funcs.grid_prediction_rf(X_obf_dict[i], X_ori, 'rf')
                    acc_ori_xgb, acc_obf_xgb = funcs.grid_prediction_xgb(X_obf_dict[i], X_ori, 'xgb')
                    acc_oris_rf.append(acc_ori_rf)
                    acc_obfs_rf.append(acc_obf_rf)
                    acc_oris_xgb.append(acc_ori_xgb)
                    acc_obfs_xgb.append(acc_obf_xgb)

                results['acc_ori_rf'].append(np.mean(acc_oris_rf))
                results['acc_obf_rf'].append(np.mean(acc_obfs_rf))
                results['acc_ori_xgb'].append(np.mean(acc_oris_xgb))
                results['acc_obf_xgb'].append(np.mean(acc_obfs_xgb))
                results['rec_ori'].append(np.mean(rec_oris))
                results['rec_obf'].append(np.mean(rec_obfs))

            avg_mae_ori_rf = np.mean(np.array(results['acc_ori_rf']))
            avg_mae_obf_rf = np.mean(np.array(results['acc_obf_rf']))
            avg_mae_ori_xgb = np.mean(np.array(results['acc_ori_xgb']))
            avg_mae_obf_xgb = np.mean(np.array(results['acc_obf_xgb']))
            avg_rec_ori = np.mean(np.array(results['rec_ori']))
            avg_rec_obf = np.mean(np.array(results['rec_obf']))

            with open("HyObscure_checkin_k&l_results", 'a') as file_out_overall:
                file_out_overall.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (deltaX, grid_area_number,
                                                                                        k_threshold, l_threshold,
                                                                                        avg_mae_ori_rf,
                                                                                        avg_mae_obf_rf,
                                                                                        avg_mae_ori_xgb,
                                                                                        avg_mae_obf_xgb,
                                                                                        avg_rec_ori, avg_rec_obf))

                file_out_overall.flush()
