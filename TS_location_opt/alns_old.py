import copy
import random
import pandas as pd
import math
import time
import re
from collections import Counter
pd.set_option('display.max_columns', None)  # 设置显示所有列

# 返回仓位到原点的曼哈顿距离
def distance_to_origin(position):
    center_x, center_y = center_point_warehouse[position]
    return abs(center_x) + abs(center_y)

# 验证分配的正确性——分配的物品和输入的物品一致
def all_assign_test(position_scheduling, air_item_sort, other_item_sort):
    count_air = {}
    for key, value in position_scheduling.items():
        if value != {}:
            for key1, value1 in position_scheduling[key].items():
                if key1 not in count_air:
                    count_air[key1] = value1
                else:
                    count_air[key1] += value1
    count_air = dict(sorted(count_air.items(), key=lambda x: x[1]))
    raw_air = {}
    for i in range(len(air_item_sort)):
        for j in range(items_num):
            item = air_item_sort[i][j]
            raw_air[item[0]] = item[2]
    for i in range(len(other_item_sort)):
        for j in range(items_num):
            item = other_item_sort[i][j]
            raw_air[item[0]] = item[2]
    raw_air = dict(sorted(raw_air.items(), key=lambda x: x[1]))
    print(Counter(raw_air) == Counter(count_air))

# 得到以物料编码标识的调度方案
def output_scheduling(position_scheduling, material_code):
    final_scheduling = {}
    for key, value in position_scheduling.items():
        if value == {}:
            final_scheduling[key] = {}
        if value != {}:
            final_scheduling[key] = {}
            for item_id, item_num in value.items():
                final_scheduling[key][material_code[item_id-1]] = item_num
    return final_scheduling

# 贪婪生成初始个体
def greedy_initial(air_item_sort, other_item_sort, air_position_list, position_distance, position_type, position_dict, greedy_rare):
    position_dict_copy = copy.deepcopy(position_dict)
    position_type_copy = copy.deepcopy(position_type)
    position_scheduling = {key: {} for key in position_dict}
    # 对分类中包含空调的物品进行调度
    for i in range(len(air_item_sort)):
        for j in range(items_num):
            item = air_item_sort[i][j]
            remaining_quantity = item[2]
            for position in air_position_list:
                for id, volume in position_type_copy[position]:
                    if position_dict_copy[id] == 0:
                        position_type_copy[position].remove([id, volume])
                    if random.random() < greedy_rare:
                        max_quantity = math.floor(position_dict_copy[id]/item[3])
                        quantity_to_place = min(remaining_quantity, max_quantity)
                        if quantity_to_place > 0:
                            position_scheduling[id][item[0]] = quantity_to_place
                            position_dict_copy[id] -= quantity_to_place * item[3]
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
    for i in range(len(other_item_sort)):
        for j in range(items_num):
            item = other_item_sort[i][j]
            remaining_quantity = item[2]
            for position in position_distance.keys():
                for id, volume in position_type_copy[position]:
                    if position_dict_copy[id] == 0:
                        position_type_copy[position].remove([id, volume])
                    if random.random() < greedy_rare:
                        max_quantity = math.floor(position_dict_copy[id] / item[3])
                        quantity_to_place = min(remaining_quantity, max_quantity)
                        if quantity_to_place > 0:
                            position_scheduling[id][item[0]] = quantity_to_place
                            position_dict_copy[id] -= quantity_to_place * item[3]
                            remaining_quantity -= quantity_to_place
                        if remaining_quantity == 0:
                            break
                    else:
                        continue
                if remaining_quantity == 0:
                    break
            if remaining_quantity == 0:
                continue
    return position_scheduling, position_dict_copy

# 计算第一个目标——总热度指标，根据货物的热度(heat_score)乘以距离原点的曼哈顿距离来计算
def calculate_first_objective(position_scheduling, position_distance):
    total_heat_distance = 0
    # 总热度指标=各仓位存放的货品的热度 * 仓位到达原点的距离
    for position, goods_dict in position_scheduling.items():
        match = re.search(r"([^0-9]*)\d", position)
        content_before_number = position_warehouse[i][match.start():match.end() - 1]
        distance = position_distance[content_before_number]
        for goods_id in goods_dict:
            heat_distance = heat_score[goods_id - 1] * distance  # 热度乘以距离
            total_heat_distance += heat_distance
    return total_heat_distance

# 查找每个物品放置在哪些仓位中
def find_positions_for_goods(positions, one_dim_list2):
    goods_positions = {}
    for goods_id in one_dim_list2:
        goods_positions[goods_id] = [position for position, goods_dict in positions.items() if goods_id in goods_dict.keys()]
    # 返回记录物品摆放仓位的字典 dict goods_positions[物品ID] = [仓位名,仓位名,...]
    return goods_positions

# 计算第二个目标——分散度指标
def calculate_second_objective(position_scheduling, classification_item, one_dim_list2):
    goods_positions = find_positions_for_goods(position_scheduling, one_dim_list2)
    interclass_distance_sum = 0
    for class_item in classification_item:
        # 计算一类物品中所有物品两两间距离的和
        class_distance_sum = 0
        for goods_id1_index in range(len(class_item)-1):
            for goods_id2_index in range(goods_id1_index+1,len(class_item)-1):
                for position in goods_positions[class_item[goods_id1_index][0]]:
                    for other_position in goods_positions[class_item[goods_id2_index][0]]:
                        # 计算两个仓位的距离，并累加到类间距中
                        pos1 = center_point_warehouse[position]
                        pos2 = center_point_warehouse[other_position]
                        class_distance_sum += abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        # 将一类物品的类间距加到总的类间距和中
        interclass_distance_sum += class_distance_sum
    return interclass_distance_sum

# 大邻域搜索
def large_neighborhood_search(search_rare, position_scheduling, position_volume_dict, position_dict, air_position_list, position_distance, position_type, greedy_rare):
    # 确定需要重新分配的物品及数量，更新仓位可用体积和仓位安排
    change_air_item = {}
    change_other_item = {}
    position_type_copy = copy.deepcopy(position_type)
    new_position_scheduling = copy.deepcopy(position_scheduling)
    for position, scheduling in new_position_scheduling.items():
        if new_position_scheduling[position] == {}:
            continue
        if random.random() < search_rare:
            position_volume_dict[position] = position_dict[position]
            for item_id, item_num in scheduling.items():
                if area[item_id-1] == "空调":
                    if item_id not in change_air_item:
                        change_air_item[item_id] = item_num
                    else:
                        change_air_item[item_id] += item_num
                else:
                    if item_id not in change_other_item:
                        change_other_item[item_id] = item_num
                    else:
                        change_other_item[item_id] += item_num
            new_position_scheduling[position] = {}
    # 随机分配的物品————后续还需要提升的话扩展实验考虑是否需要加入引导性指标
    random_air_item = change_air_item
    random_other_item = change_other_item
    # 随机分配空调，优先离原点较近的仓位
    for item_id, item_num in random_air_item.items():
        remaining_quantity = item_num
        for position in air_position_list:
            for position_name, volume in position_type_copy[position]:
                if random.random() < greedy_rare:
                    max_quantity = math.floor(position_volume_dict[position_name] / volume_goods[item_id - 1])
                    quantity_to_place = min(remaining_quantity, max_quantity)
                    if quantity_to_place > 0:
                        if item_id not in new_position_scheduling[position_name]:
                            new_position_scheduling[position_name][item_id] = quantity_to_place
                            position_volume_dict[position_name] -= quantity_to_place * volume_goods[item_id - 1]
                            remaining_quantity -= quantity_to_place
                        else:
                            new_position_scheduling[position_name][item_id] += quantity_to_place
                            position_volume_dict[position_name] -= quantity_to_place * volume_goods[item_id - 1]
                            remaining_quantity -= quantity_to_place
                    if remaining_quantity == 0:
                        break
                else:
                    continue
            if remaining_quantity == 0:
                break
        if remaining_quantity == 0:
            continue
    # 随机分配其它物品，优先离原点较近的仓位
    for item_id, item_num in random_other_item.items():
        remaining_quantity = item_num
        for position in position_distance.keys():
            for position_name, volume in position_type_copy[position]:
                if random.random() < greedy_rare:
                    max_quantity = math.floor(position_volume_dict[position_name] / volume_goods[item_id - 1])
                    quantity_to_place = min(remaining_quantity, max_quantity)
                    if quantity_to_place > 0:
                        if item_id not in new_position_scheduling[position_name]:
                            new_position_scheduling[position_name][item_id] = quantity_to_place
                            position_volume_dict[position_name] -= quantity_to_place * volume_goods[item_id - 1]
                            remaining_quantity -= quantity_to_place
                        else:
                            new_position_scheduling[position_name][item_id] += quantity_to_place
                            position_volume_dict[position_name] -= quantity_to_place * volume_goods[item_id - 1]
                            remaining_quantity -= quantity_to_place
                    if remaining_quantity == 0:
                        break
                else:
                    continue
            if remaining_quantity == 0:
                break
        if remaining_quantity == 0:
            continue
    return new_position_scheduling, position_volume_dict

# 确定全局最优解
def find_best_solution(best_solution, best_value, position_scheduling, chromosome_value):
    mean_value1 = (best_value[0] + chromosome_value[0])/2
    mean_value2 = (best_value[1] + chromosome_value[1])/2
    if best_value[0]/mean_value1 + best_value[1]/mean_value2 >= chromosome_value[0]/mean_value1 + chromosome_value[1]/mean_value2:
        best_value = chromosome_value
        best_solution = position_scheduling
    return best_solution, best_value

if __name__ == '__main__':
    # 开始时间
    start_time = time.time()

    # # # 数据处理
    # 读取仓位数据 (仓位 最大摆放数 仓位高度 横梁长度 托盘尺寸 体积 center_point_x center_point_y)
    warehouse_df = pd.read_excel('仓位数据.xlsx')

    # 读取已使用体积数据 (仓位 最大摆放数 仓位高度 横梁长度 托盘尺寸 体积 使用体积 剩余体积 已使用体积)
    used_volume_df = pd.read_excel('体积.xlsx')
    used_volume_df['已使用体积'] = used_volume_df['使用体积'].fillna(0)

    # 将仓位数据的每一列存储为一个列表
    position_warehouse = warehouse_df['仓位'].tolist()  # 仓位
    max_placement = warehouse_df['最大摆放数'].tolist()  # 最大摆放数
    height_warehouse = warehouse_df['仓位高度'].tolist()  # 仓位高度
    beam_length = warehouse_df['横梁长度'].tolist()  # 横梁长度
    pallet_size = warehouse_df['托盘尺寸'].tolist()  # 托盘尺寸
    volume_warehouse = warehouse_df['体积'].tolist()  # 体积
    used_volume = used_volume_df['已使用体积'].tolist()  # 已使用体积

    center_point_x_warehouse = warehouse_df['center_point_x'].tolist()  # center_point_x
    center_point_y_warehouse = warehouse_df['center_point_y'].tolist()  # center_point_y

    # 实际仓位可用的体积
    volume_warehouse = [max(0, vol - used_vol) for vol, used_vol in zip(volume_warehouse, used_volume)]
    warehouse_df['体积'] = volume_warehouse

    # 获得每个仓位的中心点
    center_point_warehouse = {position: (center_x, center_y) for position, center_x, center_y in zip(position_warehouse, center_point_x_warehouse, center_point_y_warehouse)}

    # 读取货物数据 (物料编码 物料名称 体积(CBM) 长度(cm) 宽度(cm) 高度(cm) 仓位区域判断 heat_score 30d_count 总体积)
    goods_df = pd.read_excel('货物清单.xlsx')

    # 将货物数据的每一列存储为一个列表
    material_code = goods_df['物料编码'].tolist()  # 物料编码
    material_name = goods_df['物料名称'].tolist()  # 物料名称
    goods_df['体积(CBM)'] = goods_df['体积(CBM)'].fillna(0.000001)  # 处理nan数据
    volume_goods = goods_df['体积(CBM)'].tolist()  # 体积
    volume_goods = [x if x != 0 else 0.000001 for x in volume_goods]  # 如果体积为0则为0.000001
    length = goods_df['长度(cm)'].tolist()  # 长度
    width = goods_df['宽度(cm)'].tolist()  # 宽度
    height_goods = goods_df['高度(cm)'].tolist()  # 高度
    area = goods_df['仓位区域判断'].tolist()  # 仓位区域判断
    heat_score = goods_df['heat_score'].tolist()  # heat_score
    count_30d = goods_df['30d_count'].tolist()  # 30d_count
    pre_total_volume = goods_df['总体积'].tolist()  # 总体积
    total_volume = []  # 如果体积为0则为单件商品体积
    for i in range(len(pre_total_volume)):
        if pre_total_volume[i] != 0:
            total_volume.append(pre_total_volume[i])
        else:
            total_volume.append(volume_goods[i])

    # 将货物按顺序编号
    goods_df['编号'] = range(1, len(goods_df) + 1)

    # 删除30d_count为0(未来三十天预测为0)的行，并存储到新的DataFrame
    df_30d_count_zero = goods_df[goods_df['30d_count'] == 0]
    goods_df_notzero = goods_df[goods_df['30d_count'] != 0]

    # 按“仓位区域判断”列分成两个df
    df_other = df_30d_count_zero[df_30d_count_zero['仓位区域判断'] == '其他']
    df_air_conditioning = df_30d_count_zero[df_30d_count_zero['仓位区域判断'] == '空调']

    # 读取txt文件内容，对物料编码的分类文件
    with open('分类.txt', 'r') as f:
        content = f.read()

    # 按照空格和逗号分割文本，并去除空白字符
    elements = content.replace(' ', '').split(',')

    # 将元素分组为每个子列表包含五个数字，即货物的编号
    items_num = 5  # 每组的物品数量
    result = []
    sublist = []
    for element in elements:
        # 去除换行符和中括号
        element = element.replace('\n', '').replace('[', '').replace(']', '')
        if element:
            if element.startswith('"') and element.endswith('"'):
                element = element[1:-1]  # 去除引号
            sublist.append(int(element))
        if len(sublist) == items_num:
            result.append(sublist)
            sublist = []

    # 获取各分类的物品编号，热度，未来30天预测数量，单位体积,摆放区域，每个种类列表最后会有种类单位体积热度，三维列表[[[编号，热度，预测数量，体积,摆放区域，单位体积热度],...[],种类单位体积热度],[[],...]]
    classification_item = []
    air_item = []  # 包含空调的种类
    other_item = []  # 不包含空调的种类
    for xlist in result:
        item = []
        items_heat = 0
        items_volume = 0
        unit_heat = 0
        label = 0
        for num in xlist:
            index = material_code.index(num)
            if count_30d[index] != 0:
                item.append([index+1,heat_score[index],count_30d[index],volume_goods[index],area[index],heat_score[index]/volume_goods[index]])
                items_volume += total_volume[index]
                items_heat += heat_score[index] * count_30d[index]
                if area[index] == "空调":
                    label = 1
            else:
                item.append([index+1,heat_score[index],1,volume_goods[index],area[index],heat_score[index]/volume_goods[index]])
                items_volume += total_volume[index]
                items_heat += heat_score[index] * 1
                if area[index] == "空调":
                    label = 1
        item = sorted(item, key=lambda x: x[-1], reverse=True)  # 类内物品根据单位热度升序
        item.append(items_heat/items_volume)
        classification_item.append(item)
        if label == 1:
            air_item.append(item)
        else:
            other_item.append(item)

    # 合并预测数量不为0的物品列表及物品分类表
    lst = goods_df_notzero['编号'].tolist()  # 得到预测数量不为0的物品编号，一维列表
    one_dim_list2 = [material_code.index(num) + 1 for xlist in result for num in xlist]  # 在分类表中的物品
    lst = [x for x in lst if x not in one_dim_list2]
    final_list_sorted = one_dim_list2 + lst  # 最终需要首先进行上架物品编号列表

    # 根据种类单位体积热度排序
    unit_heat_sort = sorted(classification_item, key=lambda x: x[-1], reverse=True)
    air_item_sort = sorted(air_item, key=lambda x: x[-1], reverse=True)
    other_item_sort = sorted(other_item, key=lambda x: x[-1], reverse=True)

    # 将预测数量不为0的，没有分类的物品信息也加入
    item_list = []
    air_list = []
    other_list = []
    for index in lst:
        item_list.append([index, heat_score[index-1], count_30d[index-1], volume_goods[index-1], area[index-1]])
        if area[index-1] == "空调":
            air_list.append([index, heat_score[index-1], count_30d[index-1], volume_goods[index-1], area[index-1]])  # 空调
        else:
            other_list.append([index, heat_score[index-1], count_30d[index-1], volume_goods[index-1], area[index-1]])  # 非空调
    unit_heat_sort.append(item_list)
    air_item_sort.append(air_list)
    other_item_sort.append(other_list)

    # position_distance记录各个类型仓位的区域中心点位置(后面会改成距离) position_type记录各类型仓位包括哪些具体可用仓位名,position_dict记录具体各仓位对应的可用体积
    position_distance = {}
    position_type = {}
    position_dict = {}
    for i in range(len(position_warehouse)):
        match = re.search(r"([^0-9]*)\d", position_warehouse[i])
        content_before_number = position_warehouse[i][match.start():match.end() - 1]
        position_distance[content_before_number] = [center_point_x_warehouse[i],center_point_y_warehouse[i]]
        if content_before_number not in position_type:
            position_type[content_before_number] = []
            if volume_warehouse[i] > 0:
                position_type[content_before_number].append([position_warehouse[i],volume_warehouse[i]])
                position_dict[position_warehouse[i]] = volume_warehouse[i]
        else:
            if volume_warehouse[i] > 0:
                position_type[content_before_number].append([position_warehouse[i],volume_warehouse[i]])
                position_dict[position_warehouse[i]] = volume_warehouse[i]
    for key,value in position_type.items():
        value = sorted(value, key=lambda x: x[-1], reverse=True)
        position_type[key] = value

    # 计算各类型仓位的距离 嵌套字典,子字典的值根据距离升序  distance_dict[仓位类型1][仓位类型2] = 两者距离
    distance_dict = {}
    for key1 in position_distance.keys():
        key1_distance = {}
        for key2 in position_distance.keys():
            if key1 != key2:
                key1_distance[key2] = abs(position_distance[key1][0]-position_distance[key2][0]) + abs(position_distance[key1][1]-position_distance[key2][1])
        key1_distance = dict(sorted(key1_distance.items(), key=lambda x: x[1]))
        distance_dict[key1] = key1_distance

    # 记录各个类型仓位的到原点的距离
    for key,value in position_distance.items():
        position_distance[key] = abs(value[0]) + abs(value[1])
    position_distance = dict(sorted(position_distance.items(), key=lambda x: x[1]))

    # 记录放置空调的仓位类型和其它仓位类型
    air_position_list = []
    other_position_list = []
    for x in position_distance.keys():
        if len(x) == 1:
            air_position_list.append(x)
        else:
            other_position_list.append(x)

    # # # 算法部分
    # 参数设置
    greedy_rare = 0.8  # 贪婪概率
    pr = 0.8  # 择优概率
    base_search = 0.1  # 自适应参数
    max_gen = 50  # 最大迭代次数
    restart = 10  # 重随参数
    restart_label = 0

    # 贪婪生成初始个体
    position_scheduling, position_volume_dict = greedy_initial(air_item_sort, other_item_sort,
                                                               air_position_list, position_distance, position_type, position_dict, greedy_rare)
    chromosome_value = [calculate_first_objective(position_scheduling, position_distance),
                        calculate_second_objective(position_scheduling, classification_item, one_dim_list2)]
    all_assign_test(position_scheduling, air_item_sort, other_item_sort)

    # 记录全局最优解
    best_solution = position_scheduling
    best_value = chromosome_value
    best_position_volume = position_volume_dict
    print(best_value)

    # 结合重随的自适应大邻域搜索 adaptive_large_neighborhood_search with restart
    for gen in range(max_gen):
        print(f'第{gen}次迭代')
        if restart_label == restart:
            restart_label = 0
            # 重随
            position_scheduling, position_volume_dict = greedy_initial(air_item_sort, other_item_sort,
                                                                       air_position_list, position_distance,
                                                                       position_type, position_dict, greedy_rare)
            chromosome_value = [calculate_first_objective(position_scheduling, position_distance),
                                calculate_second_objective(position_scheduling, classification_item, one_dim_list2)]
            all_assign_test(position_scheduling, air_item_sort, other_item_sort)
            raw_best_value = best_value
            best_solution, best_value = find_best_solution(best_solution, best_value, position_scheduling, chromosome_value)
            if best_value == raw_best_value:
                restart_label += 1
            else:
                best_position_volume = position_volume_dict
                print(best_value)
        else:
            search_rare = base_search + base_search * math.exp(-2*gen/max_gen)
            # 大邻域搜索
            raw_position_volume_dict = position_volume_dict
            new_position_scheduling, new_position_volume_dict = large_neighborhood_search(search_rare, position_scheduling,
                                                                position_volume_dict, position_dict, air_position_list, position_distance, position_type, greedy_rare)
            chromosome_value = [calculate_first_objective(new_position_scheduling, position_distance),
                                calculate_second_objective(new_position_scheduling, classification_item, one_dim_list2)]
            all_assign_test(new_position_scheduling, air_item_sort, other_item_sort)
            raw_best_value = best_value
            best_solution, best_value = find_best_solution(best_solution, best_value, new_position_scheduling, chromosome_value)
            print(best_value)
            if best_value == raw_best_value:
                restart_label += 1
            else:
                best_position_volume = new_position_volume_dict
            choice_list = [0, 1, 2]
            choice_number = random.choice(choice_list)
            if choice_number == 0:
                position_scheduling = new_position_scheduling
                position_volume_dict = new_position_volume_dict
            elif choice_number == 1:
                position_scheduling = best_solution
                position_volume_dict = best_position_volume
            else:
                position_volume_dict = raw_position_volume_dict

    # 输出方案
    final_scheduling = output_scheduling(position_scheduling, material_code)
    for key,value in final_scheduling.items():
        if value != {}:
            print(f"{key}:{value}")

    # 计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("代码运行时间为：", elapsed_time, "秒")
