import pandas as pd
import numpy as np
import copy
import funcs
import random
import scipy.io
import matlab.engine


def HyObscure(deltaX, grid_area_number, cluster_num, k_threshold, l_threshold, df_test, grid_list,
              area_reducibility, grid_area_dict, area_grid_dict, area_grid_colrow_dict, area_grid_rowcol_dict,
              grid_colrow, grid_rowcol, df_train, df_test_rec_items, pp, method):
    df_test_copy = copy.deepcopy(df_test)
    df_test_copy['grid_group'] = pd.Series(np.zeros(df_test_copy.shape[0]), index=df_test_copy.index,
                                          dtype='int32')

    xpgg = np.ones((cluster_num * grid_area_number, cluster_num * grid_area_number)) * 0.00000001
    JSD_Mat = np.ones((cluster_num * grid_area_number, cluster_num * grid_area_number))
    pgy = np.ones((len(grid_list), cluster_num * grid_area_number)) * 0.00000001

    JSD_Mat_dict = {}
    pgy_dict = {}

    for op in range(0, 3):
        ## compute JSD and pgy
        JSD_Mat, pgy, JSD_Mat_dict, pgy_dict = funcs.get_JSD_PGY(df_test, area_grid_dict,
                                                                 JSD_Mat_dict, pgy_dict, JSD_Mat, pgy, cluster_num, method)
        print('op:', op)
        grid_xpgg_dict = {}
        ## compute xpgg
        for gg in range(0, grid_area_number):
            eng = matlab.engine.start_matlab()
            eng.edit('../../matlab/checkin_clusternum_scenario_I/HyObscure', nargout=0)
            eng.cd('../../matlab/checkin_clusternum_scenario_I', nargout=0)
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

            l_cur = funcs.l_diversity(df_test, area_grids)  ## check l diversity
            l_range = int(np.exp(l_cur) - np.exp(np.log(l_threshold)))
            print('start adjusting area: ', area_code)
            if l_range > 0:
                ### select one direction to adjust: left (0); right (1); up (2); down(3)
                d = np.random.choice([0, 1, 2, 3],
                                     p=area_reducibility[area_code] / np.sum(area_reducibility[area_code]))
                # the selected area can be reduced through the selected direction
                if d < 2:  ## change left or right
                    area_grid_line_list_dict = area_grid_rowcol_dict
                    line_list_to_grid = funcs.rowcol_to_grid
                    grid_linelist = grid_rowcol
                else:  ## change up or down
                    area_grid_line_list_dict = area_grid_colrow_dict
                    line_list_to_grid = funcs.colrow_to_grid
                    grid_linelist = grid_colrow
                area_lines = list(area_grid_line_list_dict[area_code].keys())
                area_lines.sort()
                for line in area_lines:
                    # recheck area l diversity
                    area_grids = area_grid_dict[area_code]  ## get all the grids in the area
                    l_cur = funcs.l_diversity(df_test, area_grids)  ## check l diversity
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
                        k_adjust = funcs.k_anonymity(df_test, adjusted_area_grids)

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

                            print('from area: ', area_code, 'to area: ', to_area, 'change line: ', line, 'moveout_grids: ', moveout_grids)

                            df_test_new = funcs.update_grid_group(df_test, grid_area_dict_cur)

                            new_JSD_Mat, new_pgy, new_JSD_Mat_dict, new_pgy_dict = funcs.get_JSD_PGY(df_test_new,
                                                                                                     area_grid_dict_cur,
                                                                                                     JSD_Mat_dict, pgy_dict, JSD_Mat, pgy, cluster_num, method)
                            new_mean_Utility = funcs.Mean_JSD(new_JSD_Mat, xpgg)
                            new_mean_Privacy = funcs.Mean_KL_div(new_pgy, xpgg)

                            if new_mean_Privacy < min_mean_Privacy and new_mean_Utility < min_mean_Utility:
                                min_mean_Utility = new_mean_Utility
                                min_mean_Privacy = new_mean_Privacy
                                min_grid_area_dict = grid_area_dict_cur
                                min_area_grid_dict = area_grid_dict_cur
                                min_df_test = df_test_new

                                grid_area_dict = min_grid_area_dict
                                area_grid_dict = min_area_grid_dict
                                df_test = min_df_test
                                min_distortion_budget = min_mean_Utility
                                area_grid_rowcol_dict, area_grid_colrow_dict = funcs.update_rowcol_colrow_dict(area_grid_dict)
                                print("! Find a better area group")
                                break
                            print(op, area_code, to_area, line, mgc, mean_Privacy, mean_Utility, min_mean_Privacy,
                                  min_mean_Utility, new_mean_Privacy, new_mean_Utility)
                        else:
                            print("*** area not meet k_anonymity requirement")

            else:
                print("*** area not meet l_diversity requirement")

    user_num = df_test_copy.shape[0]
    X_ori = {}
    for k in range(user_num):
        user_id = df_test_copy['uid'][k]
        X_ori[user_id] = df_test_copy[df_test_copy['uid'] == user_id].values[0, :-1]
    for k in X_ori.keys():
        user_grid = X_ori[k][-2]
        X_ori[k][-3] = grid_area_dict[user_grid]

    df_test = funcs.update_grid_group(df_test, grid_area_dict)
    df_train = funcs.update_grid_group(df_train, grid_area_dict)
    df_test_rec_items = funcs.update_grid_group(df_test_rec_items, grid_area_dict)
    
    # random forest
    model_rf = funcs.train_rf_model_check_in(df_train)
    # xgboost
    model_xgb = funcs.train_xgb_model_check_in(df_train)

    print("model train over, start obfuscating...")

    X_obf_dict = {}

    for i in range(50):
        X_obf_dict[i], _ = funcs.get_obf_X(df_test, xpgg, pp)
    
    return X_obf_dict, X_ori, model_rf, model_xgb


def PrivCheck(df_train, df_test, df_test_rec_items, grid_area_dict, area_grid_dict, grid_list,
              cluster_num, grid_area_number, deltaX, pp):
    df_train = funcs.update_grid_group(df_train, grid_area_dict)
    model_rf = funcs.train_rf_model_check_in(df_train)
    # xgboost
    model_xgb = funcs.train_xgb_model_check_in(df_train)

    pd.DataFrame(funcs.cal_pgy_withoutGridGroup(df_test, cluster_num, grid_list)).to_csv(
        'tmp/pgy_check_in_privcheck.csv', index=False, header=None)

    df_test = funcs.update_grid_group(df_test, grid_area_dict)

    JSD_Mat_dict = np.zeros((cluster_num, cluster_num, grid_area_number))
    group_user_size_dict = {}

    for gg in range(grid_area_number):
        df_test_gg = df_test.loc[df_test['grid_group'] == gg]
        grid_list_gg = area_grid_dict[gg]
        group_user_size_dict[gg] = df_test_gg.shape[0]

        JSD_Mat_dict[:, :, gg] = funcs.cal_JSD_Matrix_withoutGridGroup(df_test_gg, cluster_num, 4)

    scipy.io.savemat('tmp/JSDM_girdGroup_privcheck.mat', {"JSD_Mat_input_Yang_trueTrain": JSD_Mat_dict})

    eng = matlab.engine.start_matlab()
    eng.edit("../../matlab/checkin_clusternum_scenario_I/PrivCheck", nargout=0)
    eng.cd('../../matlab/checkin_clusternum_scenario_I', nargout=0)
    xpgg, distortion_budget = np.array(eng.PrivCheck(deltaX, nargout=2))
    xpgg = np.array(xpgg)

    df_test['grid_group'] = pd.Series(np.zeros(df_test.shape[0]), index=df_test.index, dtype='int32')

    X_obf_dict = {}
    for i in range(50):
        X_obf_dict[i], _ = funcs.get_obf_X(df_test, xpgg, pp)

    _, X_ori = funcs.get_obf_X(df_test, xpgg, pp)

    for i in X_ori.keys():
        user_grid = X_ori[i][-2]
        X_ori[i][-3] = grid_area_dict[user_grid]
        for j in range(50):
            X_obf_dict[j][i][-1] = grid_area_dict[user_grid]

    df_test_rec_items = funcs.update_grid_group(df_test_rec_items, grid_area_dict)
    
    return X_obf_dict, X_ori, model_rf, model_xgb
