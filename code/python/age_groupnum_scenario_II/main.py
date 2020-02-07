import random
import os
import pandas as pd
import numpy as np
import funcs
import obfuscations
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def item_user_selection():
    df_item_age_uid = pd.read_csv('../../../movielens/movielens_rating.csv')
    df_item_age_dropU = df_item_age_uid.drop(['uid'], axis=1)

    df_item_age = df_item_age_dropU.groupby(['age']).sum()
    df_item_age.loc['sum_user'] = df_item_age.sum()

    ### drop the item has very limited users
    df_item_age_dropI = df_item_age.drop(df_item_age.columns[df_item_age.apply(lambda col: col.sum() < 4)], axis=1)
    df_item_age_uid_dropI = df_item_age_uid.drop(df_item_age.columns[df_item_age.apply(lambda col: ((col != 0).astype(int).sum()) < 3)], axis=1)
    df_item_age_uid_dropI = df_item_age_uid_dropI[df_item_age_uid_dropI.age > 17]
    df_item_age_uid_dropI = df_item_age_uid_dropI[df_item_age_uid_dropI.age < 51]

    df_item_age_dropI.to_csv('../../../movielens/movie_ages.csv', index=True)
    df_item_age_uid_dropI.to_csv('../../../movielens/movie_ages_uid.csv', index=False)
    return df_item_age_uid_dropI.reset_index(drop=True)


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
    # group_size = 8
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


def age_group_initiate_average(df_train, age_list, age_group_number, k, l):
    age_list.sort()
    age_group_dict = {}  ## key: age; value: group
    group_age_dict = {}  ## key: group; value: age list

    age_group_len = int(len(age_list) / age_group_number)
    age_group_code = 0
    age_min = age_list[0]
    for i in range(age_group_number):
        if i < age_group_number - 1:
            group_age_dict[i] = age_list[age_group_len * i:age_group_len * (i + 1)]
        else:
            group_age_dict[i] = age_list[age_group_len * i:]
        k_cur = funcs.k_anonymity(df_train, group_age_dict[i])
        if k_cur < k:
            print("k-anominity: age group initiate failure...")
            group_age_dict = {}
            break
        else:
            l_cur = funcs.l_diversity(df_train, group_age_dict[i])
            if l_cur >= l:
                age_group_code = age_group_code + 1
            else:
                print("l-diversity: age group initiate failure...")
                group_age_dict = {}
                break

    for group, group_age_list in group_age_dict.items():
        for age in group_age_list:
            age_group_dict[age] = group

    return age_group_dict, group_age_dict


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
    df_test_n = df_test_n.drop(['age', 'age_group', 'cluster'], axis=1)

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
    # rmse_obf = sqrt(mean_squared_error(testUser_obf_items.reshape(1,row*col), df_testUser_RcdItem.reshape(1, row*col)))
    #
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


if __name__ == "__main__":
    # obfuscation method settings
    # all_methods = ['HyObscure', 'YGen', 'XObf', 'PrivCheck', 'DP', 'Frapp', 'Random', 'Sim']

    age_group_number_list = [3, 4, 5]

    method = 'HyObscure'
    deltaX = 0.6

    # fixed settings
    cluster_num = 10
    pp = 0
    k_threshold = 50
    l_threshold = np.log(5)
    cluster_seed = 34
    
    # clustering and age group initialization
    if os.path.exists('tmp'):
        pass
    else:
        os.makedirs('tmp')

    random.seed(cluster_seed)
    df_item_age_uid = item_user_selection()
    age_list = list(set(df_item_age_uid['age'].values))
    age_list.sort()
    min_age = age_list[0]
    # age_group_dict, group_age_dict = age_group_initiate_movieLens(age_list)
    
    df_item_age_uid['age_group'] = pd.Series(np.zeros(df_item_age_uid.shape[0]), index=df_item_age_uid.index, dtype='int32')
    # if method in ['HyObscure', 'YGen', 'XObf']:
    #     df_item_age_uid = funcs.update_age_group(df_item_age_uid, age_group_dict)
    cols = list(df_item_age_uid.columns.values)
    cols_change = cols[:-3]
    cols_change.extend(['age_group', 'age', 'uid'])
    df_item_ageGroup_uid = df_item_age_uid[cols_change]

    df_cluster = funcs.Kmeans_clustering(df_item_ageGroup_uid, cluster_num, -3)
    
    for age_group_number in age_group_number_list:

        age_group_dict, group_age_dict = age_group_initiate_average(df_item_ageGroup_uid, age_list, age_group_number,
                                                                    k_threshold, l_threshold)

        if method in ['HyObscure', 'YGen', 'XObf']:
            funcs.update_age_group(df_item_ageGroup_uid, age_group_dict)

        results = {}
        results['mae_ori_rf'] = []
        results['mae_obf_rf'] = []
        results['mae_ori_xgb'] = []
        results['mae_obf_xgb'] = []
        results['rec_ori'] = []
        results['rec_obf'] = []
        
        for r in range(20):

            if method in ['HyObscure', 'YGen']:
                age_group_dict, group_age_dict = age_group_initiate_average(df_item_ageGroup_uid, age_list,
                                                                            age_group_number,
                                                                            k_threshold, l_threshold)
            
            random.seed(r * 10)
            items = list(df_item_age_uid)
            items.remove('age')
            items.remove('age_group')
            items.remove('uid')
            random.shuffle(items)
            items_train = items[:int(len(items) * 0.8)]
            items_test = list(set(items) - set(items_train))
            df_train = df_cluster.drop(items_test, axis=1)
            df_test = df_cluster.drop(items_train, axis=1)
            
            if method == 'HyObscure':
                X_obf_dict, X_ori = obfuscations.HyObscure(cluster_num, age_group_number, age_group_dict, group_age_dict,
              age_list, deltaX, k_threshold, l_threshold, df_train, df_item_age_uid, pp)
            elif method == 'YGen':
                X_obf_dict, X_ori = obfuscations.YGen(df_train, age_group_number, cluster_num, age_list, age_group_dict, group_age_dict,
         df_item_age_uid, deltaX, k_threshold, l_threshold, pp)
            elif method == 'XObf':
                X_obf_dict, X_ori = obfuscations.XObf(deltaX, cluster_num, age_group_number, age_list, group_age_dict, df_train, pp)
            elif method == 'PrivCheck':
                X_obf_dict, X_ori = obfuscations.PrivCheck(deltaX, cluster_num, age_group_number, df_cluster, df_train, age_list,
              age_group_dict, group_age_dict, pp)
            elif method == 'DP':
                X_obf_dict, X_ori = obfuscations.differential_privacy(df_train, age_group_dict, beta)
            elif method == 'Frapp':
                X_obf_dict, X_ori = obfuscations.Frapp(df_train, df_test, age_group_dict, gamma)
            elif method == 'Random':
                X_obf_dict, X_ori = obfuscations.Random(df_train, age_group_dict, p_rand)
            elif method == 'Sim':
                X_obf_dict, X_ori = obfuscations.Similarity(df_train, age_group_dict, pp)
            else:
                print('Method error. Check method setting.')
                break
            
            rec_oris = []
            rec_obfs = []
            mae_oris_rf = []
            mae_obfs_rf = []
            mae_oris_xgb = []
            mae_obfs_xgb = []
            
            threshold_list = [1, 2, 3]
            
            if method in ['HyObscure', 'YGen']:
                funcs.update_age_group(df_test, age_group_dict)
                
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
        
        with open('age_groupnum_experiments_results', 'a') as result_file:
            result_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" %(method, deltaX, age_group_number, avg_mae_ori_rf,
                                                                        avg_mae_obf_rf, avg_mae_ori_xgb, avg_mae_obf_xgb, 
                                                                        avg_rec_ori, avg_rec_obf))
            result_file.flush()