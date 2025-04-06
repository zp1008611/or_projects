import copy
import random
import pandas as pd
import math
import time
import re
from collections import Counter
import json
pd.set_option('display.max_columns', None)  # 设置显示所有列


def data_process(position_df_path, used_volume_df_path, goods_df_path, classification_list_path):
    # # # 数据处理
    # 读取仓位数据 (仓位 最大摆放数 仓位高度 横梁长度 托盘尺寸 体积 center_point_x center_point_y)
    position_df = pd.read_excel(position_df_path).reset_index(drop=True)

    # 读取已使用体积数据 (仓位 最大摆放数 仓位高度 横梁长度 托盘尺寸 体积 使用体积 剩余体积 已使用体积)
    used_volume_df = pd.read_excel(used_volume_df_path).reset_index(drop=True)
    used_volume_df['已使用体积'] = used_volume_df['使用体积'].fillna(0)

    position_volume_df = pd.merge(position_df[["仓位","体积","center_point_x", "center_point_y"]], 
                                  used_volume_df[["仓位","已使用体积"]], on="仓位", how='left').reset_index(drop=True)

    position_volume_df["剩余体积"] = position_volume_df['体积'].sub(position_volume_df['已使用体积']).clip(lower=0)
    position_volume_df["仓位"] = position_volume_df["仓位"].astype("str")
    position_volume_df["name-prefix"] = position_volume_df["仓位"].apply(lambda x: re.findall(r'[A-Za-z]+', x)[0])

    # 读取货物数据 (物料编码 物料名称 体积(CBM) 长度(cm) 宽度(cm) 高度(cm) 仓位区域判断 heat_score 30d_count 总体积)
    goods_df = pd.read_excel(goods_df_path)

    # 将货物数据的每一列存储为一个列表
    goods_df["物料编码"] = goods_df["物料编码"].astype("str")
    goods_df['体积(CBM)'] = goods_df['体积(CBM)'].fillna(0.000001)  # 处理nan数据
    goods_df['体积(CBM)'] = goods_df['体积(CBM)'].replace(0, 0.000001)
    goods_df['总体积'] = goods_df['30d_count'] * goods_df['体积(CBM)']
    goods_df = goods_df.reset_index(drop=True)

    with open(classification_list_path, 'r', encoding='utf-8') as file:
        classification_list = json.load(file)['correlation_classification']

    
    # 关联的类别信息获取
    air_item_lis = []  # 包含空调的种类
    other_item_lis = []  # 不包含空调的种类
    classification_item = []
    for group_list in classification_list:
        group_items = []
        group_items_heat = 0
        group_items_volume = 0
        for code in group_list:
            code_index = goods_df[goods_df['物料编码'] == code].index.tolist()[0]
            row_data = goods_df.loc[code_index]
            if row_data["30d_count"] != 0:
                item_info = {"code": code,
                    "heat_score":row_data["heat_score"],
                    "30d_count":int(row_data["30d_count"]),
                    "volumn":row_data["体积(CBM)"],
                    "area":row_data["仓位区域判断"],
                    "unit_heat_score":row_data["heat_score"]/row_data["体积(CBM)"]}
                group_items.append(item_info)
                group_items_volume += row_data["总体积"]
                group_items_heat += row_data["heat_score"] * row_data["30d_count"]
                if row_data["仓位区域判断"] == "空调":
                    air_item_lis.append(item_info)
                else:
                    other_item_lis.append(item_info)
            else:
                item_info = {"code": code,
                    "heat_score":row_data["heat_score"],
                    "30d_count":1,
                    "volumn":row_data["体积(CBM)"],
                    "area":row_data["仓位区域判断"],
                    "unit_heat_score":row_data["heat_score"]/row_data["体积(CBM)"]}
                group_items.append(item_info)
                group_items_volume += row_data["总体积"]
                group_items_heat += row_data["heat_score"] * 1
                if row_data["仓位区域判断"] == "空调":
                    air_item_lis.append(item_info)
                else:
                    other_item_lis.append(item_info)
        group_items = sorted(group_items, key=lambda x: x["unit_heat_score"], reverse=True)  # 类内物品根据单位热度升序
        if group_items_volume == 0:
            group_items_volume = 0.000001
        group_items.append(group_items_heat/group_items_volume)
        classification_item.append(group_items)

    # 将预测数量不为0的，没有关联类别的物品归为一类
    list1 = goods_df[goods_df["30d_count"]!=0]["物料编码"].tolist()  # 得到预测数量不为0的物品编号，一维列表
    list2 = [code for xlist in classification_list for code in xlist]  # 在分类表中的物品
    code_nogroup_list = [code for code in list1 if code not in list2]

    air_list = []
    other_list = []
    group_items = []
    group_items_heat = 0
    group_items_volume = 0
    for code in code_nogroup_list:
        code_index = goods_df[goods_df['物料编码'] == code].index.tolist()[0]
        row_data = goods_df.loc[code_index]
        item_info = {"code": code,
                    "heat_score":row_data["heat_score"],
                    "30d_count":int(row_data["30d_count"]),
                    "volumn":row_data["体积(CBM)"],
                    "area":row_data["仓位区域判断"],
                    "unit_heat_score":row_data["heat_score"]/row_data["体积(CBM)"]}
        group_items.append(item_info)
        group_items_volume += row_data["总体积"]
        group_items_heat += row_data["heat_score"] * row_data["30d_count"]
        if row_data["仓位区域判断"] == "空调":
            air_list.append(item_info)  # 空调
        else:
            other_list.append(item_info)  # 非空调
    group_items = sorted(group_items, key=lambda x: x["unit_heat_score"], reverse=True)  # 类内物品根据单位热度升序
    if group_items_volume == 0:
        group_items_volume = 0.000001
    group_items.append(group_items_heat/group_items_volume)
    air_item_lis += air_list
    other_item_lis += other_list
    classification_item.append(group_items)
     
    # 将预测数量为0的，没有关联类别的物品归为一类
    # list1 = goods_df[goods_df["30d_count"]==0]["物料编码"].tolist()  # 得到预测数量不为0的物品编号，一维列表
    # list2 = [code for xlist in classification_list for code in xlist]  # 在分类表中的物品
    # code_nogroup_list = [code for code in list1 if code not in list2]

    # air_list = []
    # other_list = []
    # group_items = []
    # group_items_heat = 0
    # group_items_volume = 0
    # for code in code_nogroup_list:
    #     code_index = goods_df[goods_df['物料编码'] == code].index.tolist()[0]
    #     row_data = goods_df.loc[code_index]
    #     item_info = {"code": code,
    #                 "heat_score":row_data["heat_score"],
    #                 "30d_count":1,
    #                 "volumn":row_data["体积(CBM)"],
    #                 "area":row_data["仓位区域判断"],
    #                 "unit_heat_score":row_data["heat_score"]/row_data["体积(CBM)"]}
    #     group_items.append(item_info)
    #     group_items_volume += row_data["总体积"]
    #     group_items_heat += row_data["heat_score"] * row_data["30d_count"]
    #     if row_data["仓位区域判断"] == "空调":
    #         air_list.append(item_info)  # 空调
    #     else:
    #         other_list.append(item_info)  # 非空调
    # group_items = sorted(group_items, key=lambda x: x["unit_heat_score"], reverse=True)  # 类内物品根据单位热度升序
    # if group_items_volume == 0:
    #     group_items_volume = 0.000001
    # group_items.append(group_items_heat/group_items_volume)
    # air_item_lis += air_list
    # other_item_lis += other_list
    # classification_item.append(group_items)

    """
    position_dict: dict
    {
      "A1-1":{
        "point" : (center_point_x,center_point_y),
        "rest_volumn": 29.30,
        "distance_to_origin" : 111
     },
     ...
     }
    """
    # 空调放置指定仓位
    air_position_list = ["C","B"]

    air_position_dict = {}
    other_position_dict = {}
    for index, row in position_volume_df.iterrows():
        position_name = row['仓位']
        position_nameprefix = row["name-prefix"]
        if position_nameprefix in air_position_list:
            if position_name not in air_position_dict:
                air_position_dict[position_name] = {
                    "coordinates": (row['center_point_x'], row['center_point_y']),
                    "rest_volumn": row["剩余体积"],
                    "distance_to_origin": abs(row['center_point_x']) + abs(row['center_point_y'])
                }
        else:
            if position_name not in other_position_dict:
                other_position_dict[position_name] = {
                    "coordinates": (row['center_point_x'], row['center_point_y']),
                    "rest_volumn": row["剩余体积"],
                    "distance_to_origin": abs(row['center_point_x']) + abs(row['center_point_y'])
                }
    return air_item_lis , other_item_lis, air_position_dict, other_position_dict, goods_df, classification_item

# 贪婪生成初始个体
def greedy_initial(air_item_sorted, other_item_sorted, air_position_dict, other_position_dict,greedy_rare):
    air_position_dict_sorted =  dict(sorted(air_position_dict.items(), key=lambda item: item[1]["distance_to_origin"]))
    other_position_dict_sorted = dict(sorted(other_position_dict.items(), key=lambda item: item[1]["distance_to_origin"]))
    position_scheduling = {position: {} for position in list(air_position_dict.keys())+list(other_position_dict.keys())}
    # 对分类中包含空调的物品进行调度,热度高的优先放进离原点近的位置
    for item in air_item_sorted:
        remaining_quantity = item["30d_count"]
        for position in air_position_dict_sorted.keys():
            if air_position_dict_sorted[position]["rest_volumn"] == 0:
                continue
            if random.random() < greedy_rare:
                max_quantity = math.floor(air_position_dict_sorted[position]["rest_volumn"]/item["volumn"])
                quantity_to_place = min(remaining_quantity, max_quantity)
                if quantity_to_place > 0:
                    position_scheduling[position][item["code"]] = quantity_to_place
                    air_position_dict_sorted[position]["rest_volumn"] -= quantity_to_place * item["volumn"]
                    remaining_quantity -= quantity_to_place
                if remaining_quantity == 0:
                    break
                else:
                    continue
            if remaining_quantity == 0:
                break
        if remaining_quantity == 0:
            continue
    # 对其它物品进行调度
    for item in other_item_sorted:
        remaining_quantity = item["30d_count"]
        for position in other_position_dict_sorted.keys():
            if other_position_dict_sorted[position]["rest_volumn"] == 0:
                continue
            if random.random() < greedy_rare:
                max_quantity = math.floor(other_position_dict_sorted[position]["rest_volumn"]/item["volumn"])
                quantity_to_place = min(remaining_quantity, max_quantity)
                if quantity_to_place > 0:
                    position_scheduling[position][item["code"]] = quantity_to_place
                    other_position_dict_sorted[position]["rest_volumn"] -= quantity_to_place * item["volumn"]
                    remaining_quantity -= quantity_to_place
                if remaining_quantity == 0:
                    break
                else:
                    continue
            if remaining_quantity == 0:
                break
        if remaining_quantity == 0:
            continue
    return position_scheduling, air_position_dict_sorted, other_position_dict_sorted


# 计算第一个目标——总热度指标，根据货物的热度(heat_score)乘以距离原点的曼哈顿距离来计算
def calculate_first_objective(position_scheduling, goods_df, position_dict):
    total_heat_distance = 0
    # 总热度指标=各仓位存放的货品的热度 * 仓位到达原点的距离
    for position, goods_dict in position_scheduling.items():
        distance = position_dict[position]["distance_to_origin"]
        for goods_code in goods_dict:
            code_index = goods_df[goods_df['物料编码'] == goods_code].index.tolist()[0]
            row_data = goods_df.loc[code_index]
            heat_distance = row_data["heat_score"] * distance  # 热度乘以距离
            total_heat_distance += heat_distance
    return total_heat_distance



# 查找每个物品放置在哪些仓位中
def find_positionsxy_for_goods(position_scheduling, position_dict):
    goods_positions = {}
    for position, goods_dict in position_scheduling.items():
        for good_code in goods_dict.keys():
            if good_code not in goods_positions:
                goods_positions[good_code] = []
            goods_positions[good_code].append(position_dict[position]["coordinates"])
    # 返回记录物品摆放仓位的字典 dict goods_positions[物品编码] = [仓位坐标点,仓位坐标点,...]
    return goods_positions

# 计算第二个目标——分散度指标, 类内距离，不计算每个类之间的距离
def calculate_second_objective(position_scheduling, classification_item, position_dict):
    goods_positions = find_positionsxy_for_goods(position_scheduling,position_dict)
    interclass_distance_sum = 0
    for class_item in classification_item:
        # 计算一类物品中所有物品两两间距离的和
        class_distance_sum = 0
        # 遍历每对不同的物料
        for i in range(len(class_item)-1):
            code1 = class_item[i]['code']
            if code1 in goods_positions:
                positions1 = goods_positions[code1]
                for j in range(i + 1, len(class_item)-1):
                    code2 = class_item[j]['code']
                    if code2 in goods_positions:
                        positions2 = goods_positions[code2]
                        # 计算每对坐标之间的曼哈顿距离
                        for pos1 in positions1:
                            for pos2 in positions2:
                                manhattan_distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                                class_distance_sum += manhattan_distance
        # 将一类物品的类间距加到总的类间距和中
        interclass_distance_sum += class_distance_sum
    return interclass_distance_sum


# 大邻域搜索
def large_neighborhood_search(search_rare, position_scheduling, goods_df, air_position_dict, other_position_dict, greedy_rare):
    # 确定需要重新分配的物品及数量，更新仓位可用体积和仓位安排
    # 把position_scheuling的部分物料拿出来(破坏)再重新分配(修复)
    change_air_item = {}
    change_other_item = {}
    new_position_scheduling = copy.deepcopy(position_scheduling)
    for position, goods_dic in new_position_scheduling.items():
        if new_position_scheduling[position] == {}:
            continue
        if random.random() < search_rare:
            for item_code, item_num in goods_dic.items():
                code_index = goods_df[goods_df['物料编码'] == item_code].index.tolist()[0]
                row_data = goods_df.loc[code_index]
                if row_data["仓位区域判断"] == "空调":
                    if item_code not in change_air_item:
                        change_air_item[item_code] = item_num
                    else:
                        change_air_item[item_code] += item_num
                    air_position_dict[position]["rest_volumn"] += item_num * row_data["体积(CBM)"]
                else:
                    if item_code not in change_other_item:
                        change_other_item[item_code] = item_num
                    else:
                        change_other_item[item_code] += item_num
                    other_position_dict[position]["rest_volumn"] += item_num * row_data["体积(CBM)"]
            new_position_scheduling[position] = {}

    def shuffle_dict(input_dict):
        # 获取字典的键
        keys = list(input_dict.keys())
        # 打乱键的顺序
        random.shuffle(keys)
        # 根据打乱后的键顺序构建新字典
        shuffled_dict = {key: input_dict[key] for key in keys}
        return shuffled_dict
    
    air_position_dict_sorted = copy.deepcopy(air_position_dict)
    other_position_dict_sorted = copy.deepcopy(other_position_dict)

    air_position_dict_sorted = shuffle_dict(air_position_dict)
    other_position_dict_sorted = shuffle_dict(other_position_dict)
    # air_position_dict_sorted =  dict(sorted(air_position_dict.items(), key=lambda item: item[1]["distance_to_origin"]))
    # other_position_dict_sorted = dict(sorted(other_position_dict.items(), key=lambda item: item[1]["distance_to_origin"]))

    # 随机分配空调
    for item_code, item_num in change_air_item.items():
        code_index = goods_df[goods_df['物料编码'] == item_code].index.tolist()[0]
        volume_goods = goods_df.loc[code_index]["体积(CBM)"]
        remaining_quantity = item_num
        for position_name in air_position_dict_sorted.keys():
            if air_position_dict_sorted[position_name]["rest_volumn"] == 0:
                continue
            if random.random() < greedy_rare:
                max_quantity = math.floor(air_position_dict_sorted[position_name]["rest_volumn"]/volume_goods)
                quantity_to_place = min(remaining_quantity, max_quantity)
                if quantity_to_place > 0:
                    if item_code not in new_position_scheduling[position_name]:
                        new_position_scheduling[position_name][item_code] = quantity_to_place
                        air_position_dict_sorted[position_name]["rest_volumn"] -= quantity_to_place * volume_goods
                        remaining_quantity -= quantity_to_place
                    else:
                        new_position_scheduling[position_name][item_code] += quantity_to_place
                        air_position_dict_sorted[position_name]["rest_volumn"] -= quantity_to_place * volume_goods
                        remaining_quantity -= quantity_to_place
                if remaining_quantity == 0:
                    break
            else:
                continue
            if remaining_quantity == 0:
                break
        if remaining_quantity == 0:
            continue
    # 随机分配其它物品
    for item_code, item_num in change_other_item.items():
        code_index = goods_df[goods_df['物料编码'] == item_code].index.tolist()[0]
        volume_goods = goods_df.loc[code_index]["体积(CBM)"]
        remaining_quantity = item_num
        for position_name in other_position_dict_sorted.keys():
            if other_position_dict_sorted[position_name]["rest_volumn"] == 0:
                continue
            if random.random() < greedy_rare:
                max_quantity = math.floor(other_position_dict_sorted[position_name]["rest_volumn"]/volume_goods)
                quantity_to_place = min(remaining_quantity, max_quantity)
                if quantity_to_place > 0:
                    if item_code not in new_position_scheduling[position_name]:
                        new_position_scheduling[position_name][item_code] = quantity_to_place
                        other_position_dict_sorted[position_name]["rest_volumn"] -= quantity_to_place * volume_goods
                        remaining_quantity -= quantity_to_place
                    else:
                        new_position_scheduling[position_name][item_code] += quantity_to_place
                        other_position_dict_sorted[position_name]["rest_volumn"] -= quantity_to_place * volume_goods
                        remaining_quantity -= quantity_to_place
                if remaining_quantity == 0:
                    break
            else:
                continue
            if remaining_quantity == 0:
                break
        if remaining_quantity == 0:
            continue
    return new_position_scheduling, air_position_dict_sorted, other_position_dict_sorted

# 确定全局最优解
def find_best_solution(best_solution, best_value, position_scheduling, chromosome_value):
    # 通过计算两个目标函数值的均值，对目标值进行归一化处理，然后比较当前最优解和新解在两个目标函数上的加权和
    mean_value1 = (best_value[0] + chromosome_value[0])/2
    mean_value2 = (best_value[1] + chromosome_value[1])/2
    if best_value[0]/mean_value1 + best_value[1]/mean_value2 >= chromosome_value[0]/mean_value1 + chromosome_value[1]/mean_value2:
        best_value = chromosome_value
        best_solution = position_scheduling
    return best_solution, best_value

def allocation_count_checking(position_scheduling,goods_df):
    """
    查看分配的数量与30dcount
    """
    allocation_total_sum = 0
    for sub_dict in position_scheduling.values():
        allocation_total_sum += sum(sub_dict.values())
    totoal_count_30d = goods_df[goods_df["30d_count"]!=0]["30d_count"].sum()
    print(allocation_total_sum)
    print(totoal_count_30d)

if __name__ == '__main__':
    # 开始时间
    start_time = time.time()
  
    air_item_lis , other_item_lis, air_position_dict, other_position_dict, goods_df, classification_item = data_process(position_df_path="仓位数据.xlsx", 
                                                                                 used_volume_df_path="体积.xlsx", 
                                                                                 goods_df_path="货物清单.xlsx", 
                                                                                 classification_list_path="classification.json"
                                                                                 )
    
    # # # 算法部分
    # 参数设置
    greedy_rare = 0.8  # 贪婪概率
    pr = 0.8  # 择优概率
    base_search = 0.1  # 自适应参数
    max_gen = 50  # 最大迭代次数


    # 贪婪生成初始个体
    # 根据种类单位体积热度排序
    air_item_sorted = sorted(air_item_lis, key=lambda x: x['unit_heat_score'], reverse=True)
    other_item_sorted = sorted(other_item_lis, key=lambda x: x['unit_heat_score'], reverse=True)
    position_scheduling, air_position_dict, other_position_dict = greedy_initial(air_item_sorted , 
                                                                                 other_item_sorted, 
                                                                                 air_position_dict, 
                                                                                 other_position_dict, 
                                                                                 greedy_rare)
    # print(position_scheduling)
    # position_scheduling: 字典，键为仓位号，每个键对应一个字典，内嵌字典的键为物料编码，值为物料存放数目

    position_dict = {**air_position_dict, **other_position_dict}
    chromosome_value = [calculate_first_objective(position_scheduling, goods_df, position_dict),
                        calculate_second_objective(position_scheduling, classification_item, position_dict)]

    # 记录全局最优解
    best_solution = position_scheduling
    best_value = chromosome_value
    best_air_position_dict, best_other_position_dict = air_position_dict, other_position_dict

    # 自适应大邻域搜索 adaptive_large_neighborhood_search， 
    # 破坏算子是从某个仓位拿出某个物品，修复算子是重新分配这个物品（优先考虑距离原点最近的仓位）
    # 接受新解的标准：有三种，无论好坏，接受新解，只接受变好的解，不接受新解
    # 因为只有一种破坏算子和修复算子，所以只需调整破坏算子和修复算子的选择概率，即调整search_rare
    for gen in range(max_gen):
        print(f'第{gen}次迭代')
        # 在算法的早期阶段，给予搜索启发式操作相对较大的权重，鼓励算法进行更广泛的探索，以发现更多潜在的解空间；
        # 而随着迭代的推进，逐渐降低搜索的权重，使算法更聚焦于当前找到的较优解附近进行局部搜索和优化
        search_rare = base_search + base_search * math.exp(-2*gen/max_gen)
        # 大邻域搜索
        raw_air_position_dict = copy.deepcopy(air_position_dict)
        raw_other_position_dict = copy.deepcopy(other_position_dict)
        raw_best_value = best_value
        new_position_scheduling, new_air_position_dict, new_new_other_position_dict = large_neighborhood_search(search_rare,
                                                                    position_scheduling, goods_df, air_position_dict, other_position_dict, greedy_rare)
        new_position_dict = {**air_position_dict, **other_position_dict}
        chromosome_value = [calculate_first_objective(new_position_scheduling, goods_df, new_position_dict),
                    calculate_second_objective(new_position_scheduling, classification_item, new_position_dict)]
        
        best_solution, best_value = find_best_solution(best_solution, best_value, new_position_scheduling, chromosome_value)
        print(best_value)
        if best_value == raw_best_value:
            pass
        else:
            best_air_position_dict, best_other_position_dict = new_air_position_dict,new_new_other_position_dict
        choice_list = [0, 1, 2]
        choice_number = random.choice(choice_list)
        # 如何接受新解
        if choice_number == 0:
            # 无论好坏，接受新解
            position_scheduling = new_position_scheduling
            air_position_dict, other_position_dict = new_air_position_dict, new_new_other_position_dict
        elif choice_number == 1:
            # 只接受变好的解
            position_scheduling = best_solution
            air_position_dict, other_position_dict = best_air_position_dict, best_other_position_dict
        else:
            # 不接受新解
            air_position_dict, other_position_dict = raw_air_position_dict,raw_other_position_dict

    # 输出方案
    for key,value in best_solution.items():
        if value != {}:
            print(f"{key}:{value}")
    allocation_count_checking(position_scheduling,goods_df)

    # 计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("代码运行时间为：", elapsed_time, "秒")

