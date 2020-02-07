import os
import copy
import pandas as pd
import numpy as np
import funcs
import matlab.engine
import scipy.io


def HyObscure(df_train, df_test, df_test_rec_items, df_item_age_uid,
             age_group_dict, group_age_dict, cluster_num, age_group_number,
              age_list, deltaX, k_threshold, l_threshold, pp):
    df_test_copy = copy.deepcopy(df_test)
    df_test_copy['age_group'] = pd.Series(np.zeros(df_test_copy.shape[0]), index=df_test_copy.index,
                                          dtype='int32')
    xpgg = np.ones((cluster_num * age_group_number, cluster_num * age_group_number)) * 0.00000001
    JSD_Mat = np.ones((cluster_num * age_group_number, cluster_num * age_group_number))
    pgy = np.ones((len(age_list), cluster_num * age_group_number)) * 0.00000001
    group_min_age_dict = {}
    group_usersize_dict = {}

    for op in range(0, 5):
        age_xpgg_dict = {}
        ###### Compute JSD, pgy, xpgg
        JSD_Mat_dict = {}
        pgy_dict = {}

        for ag in range(age_group_number):
            group_min_age_dict[ag] = group_age_dict[ag][0]
            print(group_min_age_dict[ag])
            df_test_ag = df_test.loc[df_test['age_group'] == ag]
            age_list_ag = group_age_dict[ag]
            group_usersize_dict[ag] = df_test_ag.shape[0]

            JSD_Mat_dict[ag] = funcs.cal_JSD_Matrix_withoutAgeGroup(df_test_ag, cluster_num, 4)
            print(ag, cluster_num, age_list_ag)
            pgy_dict[ag] = funcs.cal_pgy_withoutAgeGroup(df_test_ag, cluster_num, age_list_ag)

            pd.DataFrame(JSD_Mat_dict[ag]).to_csv('tmp/JSDM_ageGroup_hyobscure.csv', index=False, header=None)
            pd.DataFrame(pgy_dict[ag]).to_csv('tmp/pgy_ageGroup_hyobscure.csv', index=False, header=None)

            eng = matlab.engine.start_matlab()
            eng.edit('../../matlab/age_clusternum_scenario_I/HyObscure', nargout=0)
            eng.cd('../../matlab/age_clusternum_scenario_I', nargout=0)
            age_xpgg_dict[ag], distortion_budget = np.array(eng.HyObscure(deltaX, nargout=2))
            age_xpgg_dict[ag] = np.array(age_xpgg_dict[ag])

        for ag in range(age_group_number):
            for age in group_age_dict[ag]:
                for col in range(cluster_num):
                    pgy[age - group_min_age_dict[0], ag + col * age_group_number] = pgy_dict[ag][age -
                                                                                                 group_min_age_dict[
                                                                                                     ag], col] * \
                                                                                    group_usersize_dict[
                                                                                        ag] / \
                                                                                    df_test.shape[0]

        for ag in range(age_group_number):
            for row in range(cluster_num):
                for col in range(cluster_num):
                    xpgg[ag + row * age_group_number, ag + col * age_group_number] = age_xpgg_dict[ag][
                        row, col]
                    JSD_Mat[ag + row * age_group_number, ag + col * age_group_number] = JSD_Mat_dict[ag][
                        row, col]

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

        adjustable_groups, reducible_groups = funcs.age_group_adjust_greedy(df_item_age_uid, group_age_dict,
                                                                      k_threshold, np.log(l_threshold))
        min_group = 0
        for i in adjustable_groups:
            age_group_dict_cur = {}
            for group, group_age_list in adjustable_groups[i].items():
                for age in group_age_list:
                    age_group_dict_cur[age] = group

            df_test_new = funcs.update_age_group(df_test, age_group_dict_cur)
            new_JSD_Mat = funcs.cal_JSD_Matrix_withAgeGroup(df_test_new, cluster_num, age_group_number, 4)
            new_pgy = funcs.cal_pgy_withAgeGroup(df_test_new, cluster_num, age_group_number, age_list)
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
            print(op, i, min_group, mean_Privacy, mean_Utility, min_mean_Privacy, min_mean_Utility,
                  new_mean_Privacy, new_mean_Utility)

        if min_mean_Privacy < mean_Privacy and min_mean_Utility < mean_Utility:
            print("find a better age group:", group_age_dict)
            age_group_dict = min_age_group_dict
            group_age_dict = min_group_age_dict
            df_test = funcs.update_age_group(df_test, age_group_dict)
        else:
            break

    user_num = df_test_copy.shape[0]
    X_ori = {}
    for k in range(user_num):
        user_id = df_test_copy['uid'][k]
        X_ori[user_id] = df_test_copy[df_test_copy['uid'] == user_id].values[0, :-1]
    for k in X_ori.keys():
        user_age = X_ori[k][-2]
        X_ori[k][-3] = age_group_dict[user_age]

    df_test = funcs.update_age_group(df_test, age_group_dict)
    df_train = funcs.update_age_group(df_train, age_group_dict)
    df_test_rec_items = funcs.update_age_group(df_test_rec_items, age_group_dict)
    
    model_rf = funcs.train_rf_model(df_train)
    model_xgb = funcs.train_xgb_model(df_train)
    print("model train over, start obfuscating...")

    X_obf_dict = {}

    for i in range(100):
        X_obf_dict[i], _ = funcs.get_obf_X(df_test, xpgg, pp)
            
    return X_obf_dict, X_ori, model_rf, model_xgb


def PrivCheck(df_train, df_test, df_test_rec_items, age_group_number, cluster_num,
              deltaX, age_list, age_group_dict, group_age_dict, pp):
    funcs.update_age_group(df_train, age_group_dict)
    # random forest
    model_rf = funcs.train_rf_model(df_train)
    # xgboost
    model_xgb = funcs.train_xgb_model(df_train)


    pd.DataFrame(funcs.cal_pgy_withAgeGroup(df_test, cluster_num, 1, age_list)).to_csv(
        'tmp/pgy_ageGroup_privcheck.csv',
        index=False, header=None)

    funcs.update_age_group(df_test, age_group_dict)
    JSD_Mat_dict = np.zeros((cluster_num, cluster_num, age_group_number))
    group_min_age_dict = {}
    group_usersize_dict = {}
    for ag in range(age_group_number):
        group_min_age_dict[ag] = group_age_dict[ag][0]
        df_test_ag = df_test.loc[df_test['age_group'] == ag]
        age_list_ag = group_age_dict[ag]
        group_usersize_dict[ag] = df_test_ag.shape[0]

        JSD_Mat_dict[:, :, ag] = funcs.cal_JSD_Matrix_withoutAgeGroup(df_test_ag, cluster_num, 4)
    scipy.io.savemat('tmp/JSDM_ageGroup_privcheck.mat', {"JSD_Mat_input": JSD_Mat_dict})

    pd.DataFrame(JSD_Mat_dict[ag]).to_csv('tmp/JSDM_ageGroup_yang.csv', index=False, header=None)

    eng = matlab.engine.start_matlab()
    eng.edit('../../matlab/age_clusternum_scenario_I/PrivCheck', nargout=0)
    eng.cd('../../matlab/age_clusternum_scenario_I', nargout=0)
    xpgg, distortion_budget = np.array(eng.PrivCheck(deltaX, nargout=2))
    xpgg = np.array(xpgg)

    df_test['age_group'] = pd.Series(np.zeros(df_test.shape[0]), index=df_test.index, dtype='int32')
    funcs.update_age_group(df_test_rec_items, age_group_dict)

    X_obf_dict = {}
    for i in range(100):
        X_obf_dict[i], _ = funcs.get_obf_X(df_test, xpgg, pp)

    _, X_ori = funcs.get_obf_X(df_test, xpgg, pp)

    for i in X_ori.keys():
        user_age = X_ori[i][-2]
        X_ori[i][-3] = age_group_dict[user_age]
        for j in range(1):
            X_obf_dict[j][i][-1] = age_group_dict[user_age]
            
    return X_obf_dict, X_ori, model_rf, model_xgb

