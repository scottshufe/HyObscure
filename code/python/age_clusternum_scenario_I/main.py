import random
import os
import pandas as pd
import numpy as np
import funcs
import obfuscations
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error



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


if __name__ == "__main__":
    # obfuscation method settings
    # all_methods = ['HyObscure', 'PrivCheck']
    # method = 'HyObscure'
    method = 'PrivCheck'

    cluster_num_list = [5, 10, 15]


    # fixed settings
    deltaX = 0.6
    age_group_number = 4
    pp = 0
    k_threshold = 100
    l_threshold = 4
    available_seeds = [1, 2, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 27]
    
    # clustering and age group initialization
    if os.path.exists('tmp'):
        pass
    else:
        os.makedirs('tmp')

    df_item_age_uid = item_user_selection()
    age_list = list(set(df_item_age_uid['age'].values))
    age_list.sort()
    min_age = age_list[0]
    df_item_age_uid['age_group'] = pd.Series(np.zeros(df_item_age_uid.shape[0]), index=df_item_age_uid.index,
                                             dtype='int32')
    cols = list(df_item_age_uid.columns.values)
    cols_change = cols[:-3]
    cols_change.extend(['age_group', 'age', 'uid'])
    df_item_ageGroup_uid = df_item_age_uid[cols_change]
    age_group_dict, group_age_dict = age_group_initiate_movieLens(age_list)
    if method in ['HyObscure', 'YGen', 'XObf']:
        funcs.update_age_group(df_item_ageGroup_uid, age_group_dict)
    random.seed(32)

    for cluster_num in cluster_num_list:

        df_cluster = funcs.Kmeans_clustering(df_item_ageGroup_uid, cluster_num, -3)

        results = {}
        results['ori_age_rf'] = []
        results['obf_age_rf'] = []
        results['ori_age_xgb'] = []
        results['obf_age_xgb'] = []
        results['ori_rec'] = []
        results['obf_rec'] = []
        
        for r in available_seeds:
            if method in ['HyObscure', 'YGen']:
                age_group_dict, group_age_dict = age_group_initiate_movieLens(age_list)
            random.seed(r*10)
            items = list(df_item_age_uid)
            items.remove('age')
            items.remove('age_group')
            items.remove('uid')
            random.shuffle(items)
            items_train = items[:int(len(items) * 0.8)]
            items_test = list(set(items) - set(items_train))
            df_train_items = df_cluster.drop(items_test, axis=1)
            df_test_items = df_cluster.drop(items_train, axis=1)

            train_idx_list = []
            test_idx_list = []
            print("run seed {}...".format(r * 10))
            for i in range(cluster_num):
                df_c = df_train_items.loc[df_train_items['cluster'] == i + 1]
                df_c_train = df_c.sample(frac=0.5, random_state=r * 10)
                df_c_test = df_c.drop(df_c_train.index)

                train_idx_list.extend(list(df_c_train.index))
                test_idx_list.extend(list(df_c_test.index))

            df_train = df_train_items.loc[train_idx_list]
            df_test = df_train_items.loc[test_idx_list]
            df_train = df_train.reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)

            df_test_rec_items = df_test_items.loc[test_idx_list]
            df_test_rec_items = df_test_rec_items.reset_index(drop=True)

            print("all users num: {}".format(len(df_item_ageGroup_uid)))
            print("split train and test over")
            print("train num {}".format(len(df_train)))
            print("test num {}".format(len(df_test)))
            print("train items {}".format(df_train_items.shape[1]))
            print("test items {}".format(df_test_items.shape[1]))
                  
            if method == 'HyObscure':
                X_obf_dict, X_ori, model_rf, model_xgb = obfuscations.HyObscure(df_train, df_test, df_test_rec_items,
                                df_item_age_uid,age_group_dict, group_age_dict, cluster_num, age_group_number,age_list,
                                                                                deltaX, k_threshold, l_threshold, pp)
            elif method == 'PrivCheck':
                X_obf_dict, X_ori, model_rf, model_xgb = obfuscations.PrivCheck(df_train, df_test, df_test_rec_items,
                                                                                age_group_number, cluster_num,
                                                                                deltaX, age_list, age_group_dict,
                                                                                group_age_dict, pp)
            else:
                print('Method error. Check method setting.')
                break
            
            mae_oris_rf = []
            mae_obfs_rf = []
            mae_oris_xgb = []
            mae_obfs_xgb = []
            
            rec_oris = []
            rec_obfs = []
            
            for i in range(100):
                rmse_ori, rmse_obf = recommendation(X_obf_dict[i], X_ori, df_test_rec_items)
                rec_oris.append(rmse_ori)
                rec_obfs.append(rmse_obf)

                df_X_obf = pd.DataFrame.from_dict(X_obf_dict[i]).T
                df_x_obf_items = df_X_obf.values
                df_X_ori = pd.DataFrame.from_dict(X_ori).T
                df_x_ori_items = df_X_ori.values[:, :-2]
                df_x_y = df_X_ori.values[:, -2]
                y_pred_ori_rf = model_rf.predict(df_x_ori_items)
                y_pred_obf_rf = model_rf.predict(df_x_obf_items)
                y_pred_ori_xgb = model_xgb.predict(df_x_ori_items)
                y_pred_obf_xgb = model_xgb.predict(df_x_obf_items)

                mae_ori_rf = mean_absolute_error(df_x_y, y_pred_ori_rf)
                mae_obf_rf = mean_absolute_error(df_x_y, y_pred_obf_rf)
                mae_ori_xgb = mean_absolute_error(df_x_y, y_pred_ori_xgb)
                mae_obf_xgb = mean_absolute_error(df_x_y, y_pred_obf_xgb)

                mae_oris_rf.append(mae_ori_rf)
                mae_obfs_rf.append(mae_obf_rf)
                mae_oris_xgb.append(mae_ori_xgb)
                mae_obfs_xgb.append(mae_obf_xgb)
                
            m_rmse_ori = np.mean(rec_oris)
            m_rmse_obf = np.mean(rec_obfs)
            
            mae_ori_rf = np.mean(mae_oris_rf)
            mae_obf_rf = np.mean(mae_obfs_rf)
            mae_ori_xgb = np.mean(mae_oris_xgb)
            mae_obf_xgb = np.mean(mae_obfs_xgb)

            results['ori_age_rf'].append(mae_ori_rf)
            results['obf_age_rf'].append(mae_obf_rf)
            results['ori_age_xgb'].append(mae_ori_xgb)
            results['obf_age_xgb'].append(mae_obf_xgb)
            results['ori_rec'].append(m_rmse_ori)
            results['obf_rec'].append(m_rmse_obf)
            
        avg_mae_ori_rf = np.mean(np.array(results['ori_age_rf']))
        avg_mae_obf_rf = np.mean(np.array(results['obf_age_rf']))
        avg_mae_ori_xgb = np.mean(np.array(results['ori_age_xgb']))
        avg_mae_obf_xgb = np.mean(np.array(results['obf_age_xgb']))
        avg_rec_ori = np.mean(np.array(results['ori_rec']))
        avg_rec_obf = np.mean(np.array(results['obf_rec']))
        
        with open('age_clusternum_experiments_results', 'a') as result_file:
            result_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" %(method, deltaX, cluster_num, avg_mae_ori_rf,
                                                                        avg_mae_obf_rf, avg_mae_ori_xgb, avg_mae_obf_xgb, 
                                                                        avg_rec_ori, avg_rec_obf))
            result_file.flush()
