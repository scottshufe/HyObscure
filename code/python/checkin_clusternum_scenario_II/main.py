import pandas as pd
import os
import random
import funcs
import numpy as np
import obfuscations
from sklearn.metrics.pairwise import cosine_similarity


def read_data():
    df = pd.read_csv('../../../check_in_data/check_in_data_3users.csv')
    return df


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


if __name__ == '__main__':
    # obfuscation method settings
    # all_methods = ['HyObscure', 'PrivCheck']
    method = "HyObscure"
    # method = "PrivCheck"

    cluster_num_list = [5, 8, 10]
    
    # fixed settings
    deltaX = 0.6
    pp = 0
    grid_area_number = 4
    k_threshold = 10
    l_threshold = 5
    
    # clustering and area initialization
    if os.path.exists('tmp'):
        pass
    else:
        os.makedirs('tmp')

    df = read_data()
    grid_list = list(set(df['grid'].values))
    grid_list.sort()
    grid_colrow, grid_rowcol = get_grid_loc(grid_list)

    print("initiate area...")
    area_grid_rowcol_dict, area_grid_colrow_dict, area_grid_dict, grid_area_dict, area_reducibility = area_initiate(
        grid_rowcol, grid_area_number)
        
    df['grid_group'] = pd.Series(np.zeros(df.shape[0]), index=df.index, dtype='int32')
    if method in ['HyObscure', 'YGen', 'XObf']:
        df_grid_group = funcs.update_grid_group(df, grid_area_dict)
    else:
        df_grid_group = df
    cols = list(df.columns.values)
    cols_change = cols[:-3]
    cols_change.extend(['grid_group', 'grid', 'uid'])
    df_item_gridGroup_uid = df_grid_group[cols_change]

    
    for cluster_num in cluster_num_list:

        df_cluster = funcs.hierarchical_clustering(df_item_gridGroup_uid, cluster_num, -3, 'cosine', 'complete')

        results = {}
        results['acc_ori_rf'] = []
        results['acc_obf_rf'] = []
        results['acc_ori_xgb'] = []
        results['acc_obf_xgb'] = []
        results['rec_ori'] = []
        results['rec_obf'] = []
        
        for r in range(20):
            if method in ['HyObscure', 'YGen']:
                area_grid_rowcol_dict, area_grid_colrow_dict, area_grid_dict, grid_area_dict, area_reducibility = area_initiate(
                    grid_rowcol, grid_area_number)
                df_cluster = funcs.update_grid_group(df_cluster, grid_area_dict)
                
            random.seed(r*10)
           
            items = list(df)
            items.remove('grid')
            items.remove('grid_group')
            items.remove('uid')
            random.shuffle(items)
            items_train = items[:int(len(items) * 0.8)]
            items_test = list(set(items) - set(items_train))

            df_train = df_cluster.drop(items_test, axis=1)
            df_test = df_cluster.drop(items_train, axis=1)
            
            if method == 'HyObscure':
                X_obf_dict, X_ori, model_rf, model_xgb = obfuscations.HyObscure(df_train, grid_area_dict, area_grid_dict, cluster_num, grid_area_number, grid_list,
                    area_reducibility, area_grid_rowcol_dict, area_grid_colrow_dict, method,
                    grid_rowcol, grid_colrow, l_threshold, k_threshold, deltaX, pp)
            elif method == 'PrivCheck':
                X_obf_dict, X_ori, model_rf, model_xgb = obfuscations.PrivCheck(df_train, cluster_num, grid_list, grid_area_dict, grid_area_number, area_grid_dict, deltaX, pp)
            else:
                print('Method error. Check method setting.')
                break
            
            df_test = funcs.update_grid_group(df_test, grid_area_dict)
            
            rec_oris = []
            rec_obfs = []
            acc_oris_rf = []
            acc_obfs_rf = []
            acc_oris_xgb = []
            acc_obfs_xgb = []
            
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
            
        avg_acc_ori_rf = np.mean(np.array(results['acc_ori_rf']))
        avg_acc_obf_rf = np.mean(np.array(results['acc_obf_rf']))
        avg_acc_ori_xgb = np.mean(np.array(results['acc_ori_xgb']))
        avg_acc_obf_xgb = np.mean(np.array(results['acc_obf_xgb']))
        avg_rec_ori = np.mean(np.array(results['rec_ori']))
        avg_rec_obf = np.mean(np.array(results['rec_obf']))
        
        with open('checkin_clusternum_experiments_results', 'a') as result_file:
            result_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (method, deltaX, cluster_num, avg_acc_ori_rf, avg_acc_obf_rf,
                avg_acc_ori_xgb, avg_acc_obf_xgb, avg_rec_ori, avg_rec_obf))
            result_file.flush()