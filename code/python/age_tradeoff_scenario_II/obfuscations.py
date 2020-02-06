import copy
import pandas as pd
import numpy as np
import funcs
import matlab.engine
import scipy.io
import scipy.spatial.distance as dist
from sklearn.metrics.pairwise import cosine_similarity


def HyObscure(cluster_num, age_group_number, age_group_dict, group_age_dict,
              age_list, deltaX, k_threshold, l_threshold, df_train, df_item_age_uid, pp):
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
            pd.DataFrame(JSD_Mat_dict[ag]).to_csv('JSDM_ageGroup_11.csv', index=False, header=None)
            pd.DataFrame(pgy_dict[ag]).to_csv('pgy_ageGroup_11.csv', index=False, header=None)

            eng = matlab.engine.start_matlab()
            eng.edit('ageGroupPrivacyObf_allObf_11', nargout=0)
            age_xpgg_dict[ag], db = np.array(eng.ageGroupPrivacyObf_allObf_11(deltaX, nargout=2))
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

        pd.DataFrame(xpgg).to_csv('xpgg.csv', index=False, header=None)
        pd.DataFrame(pgy).to_csv('pgy_full.csv', index=False, header=None)
        pd.DataFrame(JSD_Mat).to_csv('JSD_full.csv', index=False, header=None)

        min_JSD_Mat = JSD_Mat
        min_pgy = pgy
        ### change age group by greedy approach
        mean_Utility = funcs.Mean_JSD(JSD_Mat, xpgg)
        mean_Privacy = funcs.Mean_KL_div(pgy, xpgg)
        min_mean_Utility = mean_Utility
        min_mean_Privacy = mean_Privacy

        adjustable_groups, reducible_groups = funcs.age_group_adjust_greedy(df_item_age_uid, group_age_dict, k_threshold,
                                                                      np.log(l_threshold))
        min_group = 0
        for i in adjustable_groups:
            age_group_dict_cur = {}
            for group, group_age_list in adjustable_groups[i].items():
                for age in group_age_list:
                    age_group_dict_cur[age] = group

            df_train_new = funcs.update_age_group(df_train, age_group_dict_cur)
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
            df_train = funcs.update_age_group(df_train, age_group_dict)
        else:
            break

    X_obf_dict = {}

    for i in range(25):
        X_obf_dict[i], _ = funcs.get_obf_X(df_train, xpgg, pp)
    
    return X_obf_dict, X_ori


def YGen(df_train, age_group_number, cluster_num, age_list, age_group_dict, group_age_dict,
         df_item_age_uid, deltaX, k_threshold, l_threshold, pp):

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

    xpgg = np.ones((cluster_num * age_group_number, cluster_num * age_group_number)) * 0.00000001
    JSD_Mat = np.ones((cluster_num * age_group_number, cluster_num * age_group_number))
    pgy = np.ones((len(age_list), cluster_num * age_group_number)) * 0.00000001

    group_min_age_dict = {}
    group_usersize_dict = {}

    age_xpgg_dict = {}
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
        # print(JSD_Mat_dict[ag].shape)
        # print(pgy_dict[ag].shape)
        pd.DataFrame(JSD_Mat_dict[ag]).to_csv('JSDM_ageGroup_approach1_y.csv', index=False, header=None)
        pd.DataFrame(pgy_dict[ag]).to_csv('pgy_ageGroup_approach1_y.csv', index=False, header=None)

        eng = matlab.engine.start_matlab()
        eng.edit('ageGroupPrivacyObf_approach1_getBudget_y', nargout=0)
        age_xpgg_dict[ag], distortion_budget = np.array(eng.ageGroupPrivacyObf_approach1_getBudget_y(deltaX, nargout=2))
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

    JSD_Mat = np.ones((cluster_num * age_group_number, cluster_num * age_group_number))
    pgy = np.ones((len(age_list), cluster_num * age_group_number)) * 0.00000001
    group_min_age_dict = {}
    group_usersize_dict = {}

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

    for ag in range(age_group_number):
        for age in group_age_dict[ag]:
            for col in range(cluster_num):
                pgy[age - group_min_age_dict[0], ag + col * age_group_number] = pgy_dict[ag][
                                                                                    age -
                                                                                    group_min_age_dict[
                                                                                        ag], col] * \
                                                                                group_usersize_dict[ag] / \
                                                                                df_train.shape[0]

    for ag in range(age_group_number):
        for row in range(cluster_num):
            for col in range(cluster_num):
                # xpgg[ag + row * age_group_number, ag + col * age_group_number] = age_xpgg_dict[ag][row, col]
                JSD_Mat[ag + row * age_group_number, ag + col * age_group_number] = JSD_Mat_dict[ag][
                    row, col]

    min_JSD_Mat = JSD_Mat
    min_pgy = pgy
    ### change age group by greedy approach
    mean_Utility = funcs.Mean_JSD(JSD_Mat, xpgg)
    mean_Privacy = funcs.Mean_KL_div(pgy, xpgg)
    min_mean_Utility = mean_Utility
    min_mean_Privacy = mean_Privacy

    adjustable_groups, reducible_groups = funcs.age_group_adjust_greedy(df_item_age_uid, group_age_dict,
                                                                        k_threshold,
                                                                        np.log(l_threshold))
    min_group = 0
    print("start adjusting...")
    better_group_flag = 0
    for i in adjustable_groups:
        age_group_dict_cur = {}
        for group, group_age_list in adjustable_groups[i].items():
            for age in group_age_list:
                age_group_dict_cur[age] = group

        df_train_new = funcs.update_age_group(df_train, age_group_dict_cur)
        new_JSD_Mat = funcs.cal_JSD_Matrix_withAgeGroup(df_train_new, cluster_num, age_group_number, 4)
        new_pgy = funcs.cal_pgy_withAgeGroup(df_train_new, cluster_num, age_group_number, age_list)
        new_mean_Utility = funcs.Mean_JSD(new_JSD_Mat, xpgg)
        new_mean_Privacy = funcs.Mean_KL_div(new_pgy, xpgg)
        print(new_mean_Privacy)
        print(new_mean_Utility)
        if new_mean_Utility < min_mean_Utility and new_mean_Privacy < min_mean_Privacy:
            min_mean_Utility = new_mean_Utility
            min_mean_Privacy = new_mean_Privacy
            min_group_age_dict = copy.deepcopy(adjustable_groups[i])
            min_age_group_dict = copy.deepcopy(age_group_dict_cur)
            min_JSD_Mat = new_JSD_Mat
            min_pgy = new_pgy
            min_group = i
            print('Find better group!')
            better_group_flag = 1
            print(i, min_group, mean_Privacy, mean_Utility, min_mean_Privacy, min_mean_Utility,
                    new_mean_Privacy,new_mean_Utility)

    if better_group_flag == 1:
        age_group_dict = min_age_group_dict
        group_age_dict = min_group_age_dict
    else:
        print("find better group failed.")

    df_train = funcs.update_age_group(df_train, age_group_dict)

    # 使用得到的xpgg求解混淆后的df_train
    X_obf_dict = {}
    for i in range(25):
        X_obf_dict[i], _ = funcs.get_obf_X(df_train, xpgg, pp)
    
    return X_obf_dict, X_ori


def XObf(deltaX, cluster_num, age_group_number, age_list, group_age_dict, df_train, pp):
    xpgg = np.ones((cluster_num * age_group_number, cluster_num * age_group_number)) * 0.00000001
    JSD_Mat = np.ones((cluster_num * age_group_number, cluster_num * age_group_number))
    pgy = np.ones((len(age_list), cluster_num * age_group_number)) * 0.00000001

    group_min_age_dict = {}
    group_usersize_dict = {}

    age_xpgg_dict = {}
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

        pd.DataFrame(JSD_Mat_dict[ag]).to_csv('JSDM_ageGroup_approach1_y.csv', index=False, header=None)
        pd.DataFrame(pgy_dict[ag]).to_csv('pgy_ageGroup_approach1_y.csv', index=False, header=None)

        eng = matlab.engine.start_matlab()
        eng.edit('ageGroupPrivacyObf_approach1_getBudget_y', nargout=0)
        age_xpgg_dict[ag], distortion_budget = np.array(eng.ageGroupPrivacyObf_approach1_getBudget_y(deltaX, nargout=2))
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

    X_obf_dict = {}
    for i in range(25):
        X_obf_dict[i], _ = funcs.get_obf_X(df_train, xpgg, pp)

    _, X_ori = funcs.get_obf_X(df_train, xpgg, pp)
    
    return X_obf_dict, X_ori


def PrivCheck(deltaX, cluster_num, age_group_number, df_cluster, df_train, age_list,
              age_group_dict, group_age_dict, pp):

    pd.DataFrame(funcs.cal_JSD_Matrix_withAgeGroup(df_cluster, cluster_num, 1, 4)).to_csv('JSDM_ageGroup.csv', index=False, header=None)
    pd.DataFrame(funcs.cal_pgy_withAgeGroup(df_train, cluster_num, 1, age_list)).to_csv(
        'pgy_ageGroup_yang_origin.csv', index=False, header=None)

    funcs.update_age_group(df_train, age_group_dict)
    JSD_Mat_dict = np.zeros((cluster_num, cluster_num, age_group_number))
    group_min_age_dict = {}
    group_usersize_dict = {}
    for ag in range(age_group_number):
        group_min_age_dict[ag] = group_age_dict[ag][0]
        df_train_ag = df_train.loc[df_train['age_group'] == ag]
        age_list_ag = group_age_dict[ag]
        group_usersize_dict[ag] = df_train_ag.shape[0]

        JSD_Mat_dict[:, :, ag] = funcs.cal_JSD_Matrix_withoutAgeGroup(df_train_ag, cluster_num, 4)
        print(JSD_Mat_dict[:, :, ag])
    scipy.io.savemat('JSDM_ageGroup_origin.mat', {"JSD_Mat_input_origin": JSD_Mat_dict})

    pd.DataFrame(JSD_Mat_dict[ag]).to_csv('JSDM_ageGroup_yang_origin.csv', index=False, header=None)

    ### Calculate obfuscation matrix xpgg via matlab
    eng = matlab.engine.start_matlab()
    eng.edit('ageGroupPrivacyObf_yang_getBudget_origin', nargout=0)
    xpgg, distortion_budget = np.array(eng.ageGroupPrivacyObf_yang_getBudget_origin(deltaX, nargout=2))
    xpgg = np.array(xpgg)

    df_train['age_group'] = pd.Series(np.zeros(df_train.shape[0]), index=df_train.index, dtype='int32')

    X_obf_dict = {}
    for i in range(25):
        X_obf_dict[i], _ = funcs.get_obf_X(df_train, xpgg, pp)

    _, X_ori = funcs.get_obf_X(df_train, xpgg, pp)
    
    return X_obf_dict, X_ori


def differential_privacy(df_train, age_group_dict, beta):
    print("generate distance matrix...")
    dist_mat = dist.squareform(dist.pdist(df_train, 'jaccard'))

    print("start obfuscating...")
    X_obf_dict = {}
    for i in range(25):
        X_obf_dict[i], _ = funcs.get_DP_obf_X(df_train, dist_mat, beta)
    _, X_ori = funcs.get_DP_obf_X(df_train, dist_mat, beta)
    print("obfuscating done.")

    for i in X_ori.keys():
        user_age = X_ori[i][-2]
        X_ori[i][-3] = age_group_dict[user_age]
        for j in range(25):
            X_obf_dict[j][i][-1] = age_group_dict[user_age]
    
    return X_obf_dict, X_ori


def Frapp(df_train, df_test, age_group_dict, gamma):
    print("start obfuscating...")
    X_obf_dict = {}
    for i in range(25):
        X_obf_dict[i], _ = funcs.get_frapp_obf_X(df_train, gamma)
    _, X_ori = funcs.get_frapp_obf_X(df_train, gamma)
    print("obfuscating done.")
    # 更新混淆前后测试集的age group
    for i in X_ori.keys():
        user_age = X_ori[i][-2]
        X_ori[i][-3] = age_group_dict[user_age]
        for j in range(25):
            X_obf_dict[j][i][-1] = age_group_dict[user_age]

    df_test_ag = funcs.update_age_group(df_test, age_group_dict)
    
    return X_obf_dict, X_ori


def Random(df_train, age_group_dict, p_rand):
    print("start obfuscating...")
    X_obf_dict = {}
    for i in range(25):
        X_obf_dict[i], _ = funcs.get_random_obf_X(df_train, p_rand)
    _, X_ori = funcs.get_random_obf_X(df_train, p_rand)
    print("obfuscating done.")

    for i in X_ori.keys():
        user_age = X_ori[i][-2]
        X_ori[i][-3] = age_group_dict[user_age]
        for j in range(25):
            X_obf_dict[j][i][-1] = age_group_dict[user_age]
    
    return X_obf_dict, X_ori


def Similarity(df_train, age_group_dict, pp):
    print("start obfuscating...")
    itemCols = df_train.columns[:-4]
    df_items = df_train[itemCols]
    sim_mat = cosine_similarity(df_items.values)

    X_obf_dict = {}
    for i in range(25):
        X_obf_dict[i], _ = funcs.get_similarity_obf_X(sim_mat, df_train, pp)
    _, X_ori = funcs.get_similarity_obf_X(sim_mat, df_train, pp)
    print("obfuscating done.")
    for i in X_ori.keys():
        user_age = X_ori[i][-2]
        X_ori[i][-3] = age_group_dict[user_age]
        for j in range(25):
            X_obf_dict[j][i][-1] = age_group_dict[user_age]
    
    return X_obf_dict, X_ori
