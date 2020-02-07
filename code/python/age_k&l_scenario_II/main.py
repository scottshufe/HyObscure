import pandas as pd
import numpy as np
import funcs
import random
import matlab.engine
import copy
import os
import utility_privacy_test
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy


def item_user_selection():
    df_item_age_uid = pd.read_csv('../../../movielens/movielens_rating.csv')
    df_item_age_dropU = df_item_age_uid.drop(['uid'], axis=1)
    
    df_item_age = df_item_age_dropU.groupby(['age']).sum()
    df_item_age.loc['sum_user'] = df_item_age.sum()

    ### drop the item has very limited users
    df_item_age_dropI = df_item_age.drop(df_item_age.columns[df_item_age.apply(lambda col:col.sum()<4)], axis=1)
    df_item_age_uid_dropI = df_item_age_uid.drop(df_item_age.columns[df_item_age.apply(lambda col: ((col!=0).astype(int).sum()) < 3)], axis=1)
    # df_item_age_uid_dropI = df_item_age_uid.drop(df_item_age.columns[df_item_age.apply(lambda col:col.sum()<4)], axis=1)
    df_item_age_uid_dropI = df_item_age_uid_dropI[df_item_age_uid_dropI.age >17]
    df_item_age_uid_dropI = df_item_age_uid_dropI[df_item_age_uid_dropI.age <51]
    
    df_item_age_dropI.to_csv('../../../movielens/movie_ages.csv', index=True)
    df_item_age_uid_dropI.to_csv('../../../movielens/movie_ages_uid.csv', index=False)

    return df_item_age_uid_dropI.reset_index(drop=True)


def get_obf_X(df_cluster, xpgg):
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
    for i in range(xpgg_norm.shape[1]):
        print(xpgg_norm[:, i].sum())
    print("obfuscating...")    
    for i in range(user_size):
        user_id = df_cluster['uid'][i]
        # selecting one cluster to change
        while (True):
            change_index = np.random.choice(range(0, cluster_size*ageGroup_size), 1, p=xpgg_norm[:, user_ageGroup_dict[user_id] + (user_cluster_dict[user_id]-1)*ageGroup_size])[0]
            change_cluster_index = int(change_index/ageGroup_size) + 1
            change_ageGroup_index = change_index % ageGroup_size
            potential_users = list(set(cluster_vec[change_cluster_index]) & set(ageGroup_vec[change_ageGroup_index]))
            if len(potential_users) > 0: # potential users may be empty by a slight probability
                break
            else:
                print("not find potential users, re-pick")
        uidx = np.random.choice(potential_users, 1)[0]
        X_ori[user_id] = df_cluster[df_cluster['uid']== user_id].values[0,:-1]
        X_obf[user_id] = df_cluster[df_cluster['uid']== uidx].values[0,:-3]
    return X_obf,X_ori


def k_anonymity(df_train, group_age_list):
    return df_train.loc[df_train['age'].isin(group_age_list)].shape[0]


def l_diversity(df_train, group_age_list):
    age_dtr = np.zeros(len(group_age_list)) 
    for i in range(len(group_age_list)):
        age_dtr[i] = df_train.loc[df_train['age'] == group_age_list[i]].shape[0]
    age_dtr_norm = age_dtr / norm(age_dtr, ord=1)
    return entropy(age_dtr_norm)


def age_group_initiate_movieLens(age_list):
    #     *  1:  "Under 18"
    #     * 18:  "18-24"
    #     * 25:  "25-34"
    #     * 35:  "35-44"
    #     * 45:  "45-49"
    #     * 50:  "50-55"
    #     * 56:  "56+"
    age_group = {}
    group_age_dict = {}  ## key: group; value: age list
    age_max = age_list[-1]
    for age in age_list:
        if age < 25:
            age_group_code = 0
        elif age < 35:
            age_group_code = 1
        elif age < 45:
            age_group_code = 2
        else:
            age_group_code = 3
        age_group[age] = age_group_code

        if age_group_code in group_age_dict:
            group_age_dict[age_group_code].append(age)
        else:
            group_age_dict[age_group_code] = [age]

    return age_group, group_age_dict


def reducible_scale(df_train, group_age_list, k, l):
    l_cur = l_diversity(df_train, group_age_list)
    range = int(np.exp(l_cur)-np.exp(l))
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


def age_group_adjust_greedy(df_train, group_age_dict, k, l):
    reducible_groups = {}
    for group in group_age_dict:
        left_range, right_range = reducible_scale(df_train, group_age_dict[group], k, l)
        if (left_range > 0 and group > 0) or (right_range > 0 and group < len(group_age_dict)-1):
            reducible_groups[group] = [left_range, right_range]
    
    adjustable_groups = {}
    #### selecting the best move
    j = 0
    for reduce_group in reducible_groups:
        left_range, right_range = reducible_groups[reduce_group]
        if reduce_group == 0:
            for reduce_range in range(1, right_range+1):
                adjustable_groups[j] = copy.deepcopy(group_age_dict)
                age_move = adjustable_groups[j][reduce_group][-reduce_range:]
                adjustable_groups[j][reduce_group] = adjustable_groups[j][reduce_group][:-reduce_range]
                adjustable_groups[j][reduce_group+1].extend(age_move)
                adjustable_groups[j][reduce_group+1].sort()
                del age_move[:]
                j = j+1
        elif reduce_group == len(group_age_dict)-1:
            for reduce_range in range(1, left_range+1):
                adjustable_groups[j] = copy.deepcopy(group_age_dict)
                age_move = group_age_dict[reduce_group][:reduce_range]
                adjustable_groups[j][reduce_group] = adjustable_groups[j][reduce_group][reduce_range:]
                adjustable_groups[j][reduce_group-1].extend(age_move)
                adjustable_groups[j][reduce_group-1].sort()
                del age_move[:]
                j = j+1
        else:
            for reduce_range in range(1, right_range+1):
                adjustable_groups[j] = copy.deepcopy(group_age_dict)
                age_move = group_age_dict[reduce_group][-reduce_range:]
                adjustable_groups[j][reduce_group] = adjustable_groups[j][reduce_group][:-reduce_range]
                adjustable_groups[j][reduce_group+1].extend(age_move)
                adjustable_groups[j][reduce_group+1].sort()
                del age_move[:]
                j = j+1
            for reduce_range in range(1, left_range+1):
                adjustable_groups[j] = copy.deepcopy(group_age_dict)
                age_move = group_age_dict[reduce_group][:reduce_range]
                adjustable_groups[j][reduce_group] = adjustable_groups[j][reduce_group][reduce_range:]
                adjustable_groups[j][reduce_group-1].extend(age_move)
                adjustable_groups[j][reduce_group-1].sort()
                del age_move[:]
                j = j+1
           
    return adjustable_groups, reducible_groups


def update_age_group(df_train, age_group_dict):
    for k,v in age_group_dict.items():
        df_train.loc[df_train['age'] == k, 'age_group'] = v
    return df_train


def recommendation(X_obf, X_ori, df_test):
    df_X_obf = pd.DataFrame.from_dict(X_obf).T
    df_X_ori = pd.DataFrame.from_dict(X_ori).T
    users = list(X_obf.keys())

    random.seed(10)
    random.shuffle(users)
    user_train = users[:int(len(users) * 0.7)]
    user_test = list(set(users) - set(user_train))
    df_obf_trainUser_KnownItem = df_X_obf.drop(user_test).values
    df_obf_testUser_KnownItem = df_X_obf.drop(user_train).values

    testUser_obf_similarity = cosine_similarity(df_obf_testUser_KnownItem, df_obf_trainUser_KnownItem)
    normed_testUser_obf_similarity = normalize(testUser_obf_similarity, axis=1, norm='l1')

    df_test_n = df_test.set_index('uid')
    df_test_n = df_test_n.drop(['age'], axis=1)

    df_trainUser_RcdItem_df = df_test_n.drop(user_test)
    df_trainUser_RcdItem = df_trainUser_RcdItem_df.values
    df_testUser_RcdItem = df_test_n.drop(user_train).values

    trainUser_RcdItem_rating_or_not = df_trainUser_RcdItem_df.copy()
    trainUser_RcdItem_rating_or_not[trainUser_RcdItem_rating_or_not > 0] = 1
    # RcdItem_rating_user_num = (np.array(RcdItem_rating_user_num.values)+1)/len(user_train)
    trainUser_RcdItem_rating_or_not_value = trainUser_RcdItem_rating_or_not.values

    small_number = 0.00000001
    testUser_obf_items = np.matrix(normed_testUser_obf_similarity) * np.matrix(df_trainUser_RcdItem) / (
                np.matrix(normed_testUser_obf_similarity) * np.matrix(
            trainUser_RcdItem_rating_or_not_value) + small_number)

    binary_df_testUser_RcdItem = df_testUser_RcdItem.copy()
    binary_df_testUser_RcdItem[binary_df_testUser_RcdItem > 0] = 1  # get the items rated by test users
    testUser_obf_items = np.array(testUser_obf_items) * np.array(binary_df_testUser_RcdItem)  # element-wise multiply

    row, col = testUser_obf_items.shape
    testUser_obf_items = testUser_obf_items.reshape(row * col, )
    df_testUser_RcdItem = df_testUser_RcdItem.reshape(row * col, )
    rmse_obf = np.sqrt(np.sum((testUser_obf_items - df_testUser_RcdItem) ** 2) / np.count_nonzero(df_testUser_RcdItem))
    df_ori_trainUser_KnownItem = df_X_ori.drop(user_test).values[:, :-2]
    df_ori_testUser_KnownItem = df_X_ori.drop(user_train).values[:, :-2]
    testUser_ori_similarity = cosine_similarity(df_ori_testUser_KnownItem, df_ori_trainUser_KnownItem)
    normed_testUser_ori_similarity = normalize(testUser_ori_similarity, axis=1, norm='l1')

    testUser_ori_items = np.matrix(normed_testUser_ori_similarity) * np.matrix(df_trainUser_RcdItem) / (
                np.matrix(normed_testUser_ori_similarity) * np.matrix(
            trainUser_RcdItem_rating_or_not_value) + small_number)
    testUser_ori_items = np.array(testUser_ori_items) * np.array(binary_df_testUser_RcdItem)  # element-wise multiply

    row, col = testUser_ori_items.shape
    testUser_ori_items = testUser_ori_items.reshape(row * col, )
    rmse_ori = np.sqrt(np.sum((testUser_ori_items - df_testUser_RcdItem) ** 2) / np.count_nonzero(df_testUser_RcdItem))

    return rmse_ori, rmse_obf


if __name__ == '__main__':

    # HyObscure settings for k&l experiments
    cluster_num = 10
    age_group_number = 4
    deltaX = 0.55

    l_threshold_list = [3, 4, 5, 6]
    k_threshold_list = [60, 80, 100, 120]

    seeds = [3, 4, 5, 7, 9, 10, 12, 13, 16, 19, 20, 21, 22, 23, 25, 26, 27, 29, 31, 34]

    cluster_seed = 34

    # clustering and age group initialization
    random.seed(cluster_seed)
    df_item_age_uid = item_user_selection()
    age_list = list(set(df_item_age_uid['age'].values))
    age_list.sort()
    min_age = age_list[0]
    age_group_dict, group_age_dict = age_group_initiate_movieLens(age_list)
    ###### add age groups
    df_item_age_uid['age_group'] = pd.Series(np.zeros(df_item_age_uid.shape[0]), index=df_item_age_uid.index, dtype='int32')
    df_item_ageGroup_uid = update_age_group(df_item_age_uid, age_group_dict)
    cols = list(df_item_ageGroup_uid.columns.values)
    cols_change = cols[:-3]
    cols_change.extend(['age_group', 'age', 'uid'])
    df_item_ageGroup_uid = df_item_ageGroup_uid[cols_change]

    df_cluster = funcs.Kmeans_clustering(df_item_ageGroup_uid, cluster_num, -3)

    if os.path.exists('tmp'):
        pass
    else:
        os.makedirs('tmp')

    for l_threshold in l_threshold_list:
        for k_threshold in k_threshold_list:
            results = {}
            results['mae_ori_rf'] = []
            results['mae_obf_rf'] = []
            results['mae_ori_xgb'] = []
            results['mae_obf_xgb'] = []
            results['rec_ori'] = []
            results['rec_obf'] = []

            for r in seeds:
                age_group_dict, group_age_dict = age_group_initiate_movieLens(age_list)
                update_age_group(df_cluster, age_group_dict)

                random.seed(r*10)
                items = list(df_item_age_uid)
                items.remove('age')
                items.remove('age_group')
                items.remove('uid')
                random.shuffle(items)
                items_train = items[:int(len(items)*0.8)]
                items_test = list(set(items)-set(items_train))
                df_train = df_cluster.drop(items_test, axis=1)
                df_test = df_cluster.drop(items_train, axis=1)
                df_train_copy = copy.deepcopy(df_train)
                df_train_copy['age_group'] = pd.Series(np.zeros(df_train_copy.shape[0]), index=df_train_copy.index,
                                                         dtype='int32')
                user_num = df_train_copy.shape[0]
                X_ori = {}
                for k in range(user_num):
                    user_id = df_train_copy['uid'][k]
                    X_ori[user_id] = df_train_copy[df_train_copy['uid'] == user_id].values[0, :-1]
                for k in X_ori.keys():
                    user_age = X_ori[k][-2]
                    X_ori[k][-3] = age_group_dict[user_age]

                ### For obfuscation matrix
                ### Calculate obfuscation matrix xpgg via matlab
                xpgg = np.ones((cluster_num * age_group_number, cluster_num * age_group_number)) * 0.00000001
                JSD_Mat = np.ones((cluster_num * age_group_number, cluster_num * age_group_number))
                pgy = np.ones((len(age_list), cluster_num * age_group_number)) * 0.00000001
                group_min_age_dict = {}
                group_usersize_dict = {}
                min_distortion_budget = 0
                for op in range(0, 5):
                    age_xpgg_dict = {}
                    ###### Compute JSD, pgy, xpgg
                    JSD_Mat_dict = {}
                    pgy_dict = {}

                    for ag in range(age_group_number):
                        group_min_age_dict[ag] = group_age_dict[ag][0]
                        print(group_min_age_dict[ag])
                        df_train_ag = df_train.loc[df_train['age_group'] == ag]
                        age_list_ag = group_age_dict[ag]
                        group_usersize_dict[ag] = df_train_ag.shape[0]

                        JSD_Mat_dict[ag] = funcs.cal_JSD_Matrix_withoutAgeGroup(df_train_ag, cluster_num, 4)
                        pgy_dict[ag] = funcs.cal_pgy_withoutAgeGroup(df_train_ag, cluster_num, age_list_ag)
                        pd.DataFrame(JSD_Mat_dict[ag]).to_csv('tmp/JSDM_ageGroup_hyobscure.csv', index=False, header=None)
                        pd.DataFrame(pgy_dict[ag]).to_csv('tmp/pgy_ageGroup_hyobscure.csv', index=False, header=None)

                        eng = matlab.engine.start_matlab()
                        eng.edit('../../matlab/age_k&l_scenario_II/HyObscure', nargout=0)
                        eng.cd('../../matlab/age_k&l_scenario_II', nargout=0)
                        age_xpgg_dict[ag], db = np.array(eng.HyObscure(deltaX, nargout=2))
                        age_xpgg_dict[ag] = np.array(age_xpgg_dict[ag])

                    for ag in range(age_group_number):
                        for age in group_age_dict[ag]:
                            for col in range(cluster_num):
                                pgy[age - group_min_age_dict[0], ag + col * age_group_number] = pgy_dict[ag][
                                                                                                    age - group_min_age_dict[
                                                                                                        ag], col] * \
                                                                                                group_usersize_dict[ag] / \
                                                                                                df_train.shape[0]

                    for ag in range(age_group_number):
                        for row in range(cluster_num):
                            for col in range(cluster_num):
                                xpgg[ag + row * age_group_number, ag + col * age_group_number] = age_xpgg_dict[ag][row, col]
                                JSD_Mat[ag + row * age_group_number, ag + col * age_group_number] = JSD_Mat_dict[ag][row, col]

                    # pd.DataFrame(xpgg).to_csv('xpgg.csv', index=False, header=None)
                    # pd.DataFrame(pgy).to_csv('pgy_full.csv', index=False, header=None)
                    # pd.DataFrame(JSD_Mat).to_csv('JSD_full.csv', index=False, header=None)

                    min_JSD_Mat = JSD_Mat
                    min_pgy = pgy
                    ### change age group by greedy approach
                    mean_Utility = funcs.Mean_JSD(JSD_Mat, xpgg)
                    mean_Privacy = funcs.Mean_KL_div(pgy, xpgg)
                    min_mean_Utility = mean_Utility
                    min_mean_Privacy = mean_Privacy

                    adjustable_groups, reducible_groups = age_group_adjust_greedy(df_item_age_uid, group_age_dict, k_threshold,
                                                                                  np.log(l_threshold))
                    min_group = 0
                    for i in adjustable_groups:
                        age_group_dict_cur = {}
                        for group, group_age_list in adjustable_groups[i].items():
                            for age in group_age_list:
                                age_group_dict_cur[age] = group

                        df_train_new = update_age_group(df_train, age_group_dict_cur)
                        new_JSD_Mat = funcs.cal_JSD_Matrix_withAgeGroup(df_train_new, cluster_num, age_group_number, 4)
                        new_pgy = funcs.cal_pgy_withAgeGroup(df_train_new, cluster_num, age_group_number, age_list)
                        new_mean_Utility = funcs.Mean_JSD(new_JSD_Mat, xpgg)
                        new_mean_Privacy = funcs.Mean_KL_div(new_pgy, xpgg)
                        if new_mean_Utility < min_mean_Utility and new_mean_Privacy < min_mean_Privacy:
                            min_mean_Utility = new_mean_Utility
                            min_mean_Privacy = new_mean_Privacy
                            min_group_age_dict = copy.deepcopy(adjustable_groups[i])
                            min_age_group_dict = copy.deepcopy(age_group_dict_cur)
                            min_JSD_Mat = new_JSD_Mat
                            min_pgy = new_pgy
                            min_group = i
                        print(op, i, min_group, mean_Privacy, mean_Utility, min_mean_Privacy, min_mean_Utility, new_mean_Privacy,
                              new_mean_Utility)

                    if min_mean_Privacy < mean_Privacy and min_mean_Utility < mean_Utility:
                        print("find a better age group:", group_age_dict)
                        age_group_dict = min_age_group_dict
                        group_age_dict = min_group_age_dict
                        df_train = update_age_group(df_train, age_group_dict)
                        min_distortion_budget = min_mean_Utility
                    else:
                        break

                X_obf_dict = {}

                for i in range(25):
                    X_obf_dict[i], _ = get_obf_X(df_train, xpgg)

                rec_oris = []
                rec_obfs = []
                mae_oris_rf = []
                mae_obfs_rf = []
                mae_oris_xgb = []
                mae_obfs_xgb = []

                threshold_list = [1, 2, 3]

                update_age_group(df_test, age_group_dict)

                for i in range(25):
                    rmse_ori, rmse_obf = recommendation(X_obf_dict[i], X_ori, df_test)
                    rec_oris.append(rmse_ori)
                    rec_obfs.append(rmse_obf)

                    mae_ori_rf, mae_obf_rf = funcs.Age_prediction_accuracy_rf(
                        X_obf_dict[i], X_ori, threshold_list=threshold_list)

                    mae_ori_xgb, mae_obf_xgb = funcs.Age_prediction_accuracy_xgb(
                        X_obf_dict[i], X_ori, threshold_list=threshold_list)

                    mae_oris_rf.append(mae_ori_rf)
                    mae_obfs_rf.append(mae_obf_rf)
                    mae_oris_xgb.append(mae_ori_xgb)
                    mae_obfs_xgb.append(mae_obf_xgb)

                results['mae_ori_rf'].append(np.mean(mae_oris_rf))
                results['mae_obf_rf'].append(np.mean(mae_obfs_rf))
                results['mae_ori_xgb'].append(np.mean(mae_oris_xgb))
                results['mae_obf_xgb'].append(np.mean(mae_obfs_xgb))
                results['rec_ori'].append(np.mean(rec_oris))
                results['rec_obf'].append(np.mean(rec_obfs))

            avg_mae_ori_rf = np.mean(np.array(results['mae_ori_rf']))
            avg_mae_obf_rf = np.mean(np.array(results['mae_obf_rf']))
            avg_mae_ori_xgb = np.mean(np.array(results['mae_ori_xgb']))
            avg_mae_obf_xgb = np.mean(np.array(results['mae_obf_xgb']))
            avg_rec_ori = np.mean(np.array(results['rec_ori']))
            avg_rec_obf = np.mean(np.array(results['rec_obf']))

            with open('HyObscure_age_k&l_scenario_II_results', 'a') as file_out_overall:
                file_out_overall.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' %(deltaX, k_threshold, l_threshold,
                            avg_mae_ori_rf, avg_mae_obf_rf, avg_mae_ori_xgb, avg_mae_obf_xgb, avg_rec_ori, avg_rec_obf))
                file_out_overall.flush()
