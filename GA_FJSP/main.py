import random
import numpy as np
from itertools import permutations

# 定义工件和机器相关参数
num_jobs = 2  # 工件数量
num_machines = 5  # 机器数量
# 每个工件的工序数量
operations_per_job = [2, 3]
total_operations = sum(operations_per_job)
# 加工时间矩阵，-1 表示该工序不能在该机器上加工
processing_times = [
    [[2, 6, 5, 3, 4], [-1, 8, -1, 4, -1]],
    [[3, -1, 6, -1, 5], [4, 6, 5, -1, -1], [-1, 7, 11, 5, 8]]
]

# 获取每个工序的可选机器集
def get_available_machines():
    available_machines = []
    for job in range(num_jobs):
        for op in range(operations_per_job[job]):
            available_machines.append([i for i, time in enumerate(processing_times[job][op]) if time != -1])
    return available_machines


available_machines = get_available_machines()

# 计算机器选择部分海明距离
def hamming_distance(ms1, ms2):
    return sum(m1 != m2 for m1, m2 in zip(ms1, ms2))

# 初始化记忆库
def init_memory(memory_size):
    return []

# 更新记忆库
def update_memory(memory, new_ms, new_os, memory_size):
    new_fitness = fitness(new_ms, new_os)
    if not memory:
        memory.append((new_ms, new_os))
        return memory
    # 按适应度排序，保留优个体
    memory.sort(key=lambda x: fitness(x[0], x[1]))
    if new_fitness < fitness(memory[-1][0], memory[-1][1]):
        memory.pop()
        memory.append((new_ms, new_os))
    elif new_fitness == fitness(memory[-1][0], memory[-1][1]):
        hd = hamming_distance(new_ms, memory[-1][0])
        if hd != 0:
            memory.pop()
            memory.append((new_ms, new_os))
    # 保持记忆库大小
    while len(memory) > memory_size:
        memory.pop()
    return memory

# 全局选择（GS）机器选择方法
def global_selection():
    machine_load = [0] * num_machines
    ms = []
    os = []
    remaining_operations = [(job, op) for job in range(num_jobs) for op in range(operations_per_job[job])]
    while remaining_operations:
        job, op = random.choice(remaining_operations)
        avail_machines = available_machines[sum(operations_per_job[:job]) + op]
        available_times = [(i, processing_times[job][op][i] + machine_load[i]) for i in avail_machines]
        best_machine = min(available_times, key=lambda x: x[1])[0]
        ms_index = avail_machines.index(best_machine)
        ms.append(ms_index)
        os.append(job)
        machine_load[best_machine] += processing_times[job][op][best_machine]
        remaining_operations.remove((job, op))
    return ms, os

# 局部选择（LS）机器选择方法
def local_selection():
    ms = []
    os = []
    for job in range(num_jobs):
        machine_load = [0] * num_machines
        for op in range(operations_per_job[job]):
            avail_machines = available_machines[sum(operations_per_job[:job]) + op]
            available_times = [(i, processing_times[job][op][i] + machine_load[i]) for i in avail_machines]
            best_machine = min(available_times, key=lambda x: x[1])[0]
            ms_index = avail_machines.index(best_machine)
            ms.append(ms_index)
            os.append(job)
            machine_load[best_machine] += processing_times[job][op][best_machine]
    return ms, os

# 随机选择（RS）机器选择方法
def random_selection():
    ms = []
    os = []
    for job in range(num_jobs):
        for op in range(operations_per_job[job]):
            avail_machines = available_machines[sum(operations_per_job[:job]) + op]
            ms_index = random.randint(0, len(avail_machines) - 1)
            ms.append(ms_index)
            os.append(job)
    return ms, os

# 生成初始种群，结合三种选择方法
def generate_population(pop_size, p_gs, p_ls, p_rs):
    population = []
    gs_count = int(pop_size * p_gs)
    ls_count = int(pop_size * p_ls)
    rs_count = pop_size - gs_count - ls_count
    for _ in range(gs_count):
        ms, os = global_selection()
        population.append((ms, os))
    for _ in range(ls_count):
        ms, os = local_selection()
        population.append((ms, os))
    for _ in range(rs_count):
        ms, os = random_selection()
        population.append((ms, os))
    return population

# 机器选择部分解码
def decode_machine_selection(ms):
    J_M = []
    T = []
    op_index = 0
    for job in range(num_jobs):
        job_machines = []
        job_times = []
        for op in range(operations_per_job[job]):
            avail_machines = available_machines[op_index]
            machine_index = ms[op_index]
            # 增加索引检查
            if machine_index < len(avail_machines):
                machine = avail_machines[machine_index]
                job_machines.append(machine)
                job_times.append(processing_times[job][op][machine])
            else:
                print(f"错误: 机器索引 {machine_index} 超出了可用机器列表 {avail_machines} 的范围。")
                return None, None
            op_index += 1
        J_M.append(job_machines)
        T.append(job_times)
    return J_M, T

# 工序排序部分解码（工序插入法）
def decode_operation_sequence(os, J_M, T):
    job_op_count = [0] * num_jobs
    machine_schedules = [[] for _ in range(num_machines)]
    job_finish_times = [[0] for _ in range(num_jobs)]

    for job in os:
        op = job_op_count[job]
        machine = J_M[job][op]
        processing_time = T[job][op]

        if op == 0:
            prev_finish_time = 0
        else:
            prev_finish_time = job_finish_times[job][op - 1]

        if len(machine_schedules[machine]) == 0:
            start_time = prev_finish_time
        else:
            idle_intervals = []
            last_end_time = 0
            for start, end in machine_schedules[machine]:
                if start > last_end_time:
                    idle_intervals.append((last_end_time, start))
                last_end_time = end
            if last_end_time < float('inf'):
                idle_intervals.append((last_end_time, float('inf')))

            found = False
            for TS, TE in idle_intervals:
                t_a = max(prev_finish_time, TS)
                if t_a + processing_time <= TE:
                    start_time = t_a
                    found = True
                    break
            if not found:
                last_machine_end_time = machine_schedules[machine][-1][1] if machine_schedules[machine] else 0
                start_time = max(prev_finish_time, last_machine_end_time)

        end_time = start_time + processing_time
        machine_schedules[machine].append((start_time, end_time))
        machine_schedules[machine].sort()
        job_finish_times[job].append(end_time)
        job_op_count[job] += 1

    makespan = max([max(finish_times) for finish_times in job_finish_times])
    return makespan

# 解码染色体，得到调度方案和适应度值
def decode_chromosome(ms, os):
    J_M, T = decode_machine_selection(ms)
    if J_M is None or T is None:
        return float('inf')
    makespan = decode_operation_sequence(os, J_M, T)
    return makespan

# 计算调度方案的完工时间作为适应度值
def fitness(ms, os):
    return decode_chromosome(ms, os)

# 锦标赛选择（修改为从记忆库和种群中选择）
def tournament_selection(population, memory, tournament_size):
    combined = population + memory
    tournament = random.sample(combined, tournament_size)
    best_fitness_value = float('inf')
    best_ms = None
    best_os = None
    for ms, os in tournament:
        schedule_fitness = fitness(ms, os)
        if schedule_fitness < best_fitness_value:
            best_fitness_value = schedule_fitness
            best_ms = ms
            best_os = os
    return best_ms, best_os

# 机器选择部分的均匀交叉
def machine_selection_uniform_crossover(parent1_ms, parent2_ms):
    T_a = len(parent1_ms)
    r = random.randint(1, T_a)
    positions = random.sample(range(T_a), r)
    child1_ms = [-1] * T_a
    child2_ms = [-1] * T_a
    # 复制选中位置的基因
    for pos in positions:
        child1_ms[pos] = parent1_ms[pos]
        child2_ms[pos] = parent2_ms[pos]
    # 复制剩余基因
    p1_remaining = [parent1_ms[i] for i in range(T_a) if i not in positions]
    p2_remaining = [parent2_ms[i] for i in range(T_a) if i not in positions]
    p1_index = 0
    p2_index = 0
    for i in range(T_a):
        if child1_ms[i] == -1:
            child1_ms[i] = p2_remaining[p1_index]
            p1_index += 1
        if child2_ms[i] == -1:
            child2_ms[i] = p1_remaining[p2_index]
            p2_index += 1
    return child1_ms, child2_ms

# 工序排序部分的改进POX交叉
def operation_order_pox_crossover(parent1_os, parent2_os):
    job_set = set(range(num_jobs))
    jobset1_size = random.randint(1, num_jobs - 1)
    jobset1 = set(random.sample(job_set, jobset1_size))
    jobset2 = job_set - jobset1
    child1_os = []
    child2_os = []
    # 复制属于jobset1的工件到child1，属于jobset2的工件到child2
    for job in parent1_os:
        if job in jobset1:
            child1_os.append(job)
        if job in jobset2:
            child2_os.append(job)
    # 复制属于jobset2的工件到child1，属于jobset1的工件到child2
    for job in parent2_os:
        if job in jobset2:
            index = 0
            while index < len(child1_os) and child1_os[index] in jobset1:
                index += 1
            child1_os.insert(index, job)
        if job in jobset1:
            index = 0
            while index < len(child2_os) and child2_os[index] in jobset2:
                index += 1
            child2_os.insert(index, job)
    return child1_os, child2_os

# 结合机器选择和工序排序的交叉操作
def crossover(parent1, parent2):
    parent1_ms, parent1_os = parent1
    parent2_ms, parent2_os = parent2
    child1_ms, child2_ms = machine_selection_uniform_crossover(parent1_ms, parent2_ms)
    child1_os, child2_os = operation_order_pox_crossover(parent1_os, parent2_os)
    return (child1_ms, child1_os), (child2_ms, child2_os)

# 机器选择部分的变异
def machine_selection_mutation(ms):
    r = random.randint(1, len(ms))
    positions = random.sample(range(len(ms)), r)
    for pos in positions:
        job_index = 0
        op_index = 0
        for job in range(num_jobs):
            for op in range(operations_per_job[job]):
                if job_index + op_index == pos:
                    avail_machines = available_machines[job_index + op_index]
                    best_machine_index = np.argmin([processing_times[job][op][i] for i in avail_machines])
                    ms[pos] = best_machine_index
                op_index += 1
            job_index += op_index
            op_index = 0
    return ms

# 工序排序部分的变异（邻域搜索变异）
def operation_sequence_mutation(ms, os):
    r = random.randint(1, len(os))
    positions = random.sample(range(len(os)), r)
    selected_genes = [os[pos] for pos in positions]
    best_fitness_value = float('inf')
    best_os = os.copy()
    for perm in permutations(selected_genes):
        new_os = os.copy()
        for i, pos in enumerate(positions):
            new_os[pos] = perm[i]
        new_fitness = fitness(ms, new_os)
        if new_fitness < best_fitness_value:
            best_fitness_value = new_fitness
            best_os = new_os
    return best_os

# 变异操作
def mutation(individual):
    ms, os = individual
    if random.random() < 0.5:
        ms = machine_selection_mutation(ms)
    else:
        os = operation_sequence_mutation(ms, os)
    return ms, os

# 遗传算法主函数（添加记忆库逻辑）
def genetic_algorithm(pop_size, num_generations, crossover_prob, mutation_prob,
                      tournament_size, p_gs, p_ls, p_rs, memory_size):
    population = generate_population(pop_size, p_gs, p_ls, p_rs)
    memory = init_memory(memory_size)

    for generation in range(num_generations):
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, memory, tournament_size)
            parent2 = tournament_selection(population, memory, tournament_size)
            if random.random() < crossover_prob:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            if random.random() < mutation_prob:
                child1 = mutation(child1)
            if random.random() < mutation_prob:
                child2 = mutation(child2)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

            # 更新记忆库
            for ms, os in [child1, child2]:
                memory = update_memory(memory, ms, os, memory_size)

        population = new_population

    # 最终从记忆库和种群中选最优
    best_fitness_value = float('inf')
    best_ms = None
    best_os = None
    combined = population + memory
    for ms, os in combined:
        schedule_fitness = fitness(ms, os)
        if schedule_fitness < best_fitness_value:
            best_fitness_value = schedule_fitness
            best_ms = ms
            best_os = os
    return best_ms, best_os, best_fitness_value

# 参数设置
pop_size = 50  # 种群规模
num_generations = 100  # 最大进化代数
crossover_prob = 0.8  # 交叉概率
mutation_prob = 0.01  # 变异概率
tournament_size = 5  # 锦标赛规模
p_gs = 0.3  # 全局选择（GS）的种群比例
p_ls = 0.3  # 局部选择（LS）的种群比例
p_rs = 0.4  # 随机选择（RS）的种群比例
memory_size = 10  # 记忆库大小

best_ms, best_os, best_fitness = genetic_algorithm(pop_size, num_generations, crossover_prob, mutation_prob,
                                                   tournament_size, p_gs, p_ls, p_rs, memory_size)
print("最优机器选择编码:", best_ms)
print("最优工序排序编码:", best_os)
print("最优适应度（完工时间）:", best_fitness)
    