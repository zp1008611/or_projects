import numpy as np
import random
import sys

# Constant for incompatible machines
INFINITE = 1000000

# MSOS整数编码
def msos_encoding():
    # 机器选择部分
    ms = []
    for job_ops in operation_machines:
        for op in job_ops:
            # 随机选择可选机器集中的一台机器
            ms.append(random.randint(1, len(op)))
    # 工序排序部分
    os = []
    for job_idx, num_ops in enumerate(job_operations):
        os.extend([job_idx + 1] * num_ops)
    random.shuffle(os)
    return ms, os

# 染色体解码
def decode_chromosome(ms, os):
    # 步骤1：机器选择部分解码
    J_M = []
    T = []
    job_idx = 0
    for num_ops in job_operations:
        job_machines = []
        job_times = []
        for op_idx in range(num_ops):
            machine_index = ms.pop(0) - 1
            machine, time = operation_machines[job_idx][op_idx][machine_index]
            job_machines.append(machine)
            job_times.append(time)
        J_M.append(job_machines)
        T.append(job_times)
        job_idx += 1

    # 步骤2：工序排序部分解码（工序插入法）
    machine_schedules = {i: [] for i in set([machine for job in J_M for machine in job])}
    job_finish_times = [0] * len(job_operations)
    for job_num in os:
        job_idx = job_num - 1
        for op_idx in range(len(J_M[job_idx])):
            machine = J_M[job_idx][op_idx]
            time = T[job_idx][op_idx]
            if not machine_schedules[machine] or op_idx == 0:
                if op_idx == 0:
                    start_time = 0
                else:
                    start_time = job_finish_times[job_idx]
            else:
                intervals = [(0, 0)] + [(end, next_start) for (_, end), (next_start, _) in zip(machine_schedules[machine], machine_schedules[machine][1:])] + [(machine_schedules[machine][-1][1], float('inf'))]
                valid_intervals = []
                for TS, TE in intervals:
                    t_a = max(job_finish_times[job_idx], TS)
                    if t_a + time <= TE:
                        valid_intervals.append((t_a, TE))
                if valid_intervals:
                    start_time = valid_intervals[0][0]
                else:
                    LM_i = machine_schedules[machine][-1][1]
                    start_time = max(job_finish_times[job_idx], LM_i)
            end_time = start_time + time
            machine_schedules[machine].append((start_time, end_time))
            job_finish_times[job_idx] = end_time
            break

    # 计算最大完工时间
    makespan = max([end for machine_schedule in machine_schedules.values() for _, end in machine_schedule])
    return makespan

# 初始化种群
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        ms, os = msos_encoding()
        makespan = decode_chromosome(ms.copy(), os.copy())
        population.append((ms, os, makespan))
    return population

# 交叉操作（机器选择部分均匀交叉）
def crossover(ms1, ms2):
    r = random.randint(1, T_a)
    indices = random.sample(range(T_a), r)
    c1 = [None] * T_a
    c2 = [None] * T_a
    for idx in indices:
        c1[idx] = ms1[idx]
        c2[idx] = ms2[idx]
    remaining1 = [gene for i, gene in enumerate(ms1) if i not in indices]
    remaining2 = [gene for i, gene in enumerate(ms2) if i not in indices]
    j1 = j2 = 0
    for i in range(T_a):
        if c1[i] is None:
            c1[i] = remaining2[j2]
            j2 += 1
        if c2[i] is None:
            c2[i] = remaining1[j1]
            j1 += 1
    return c1, c2

# 遗传算法主函数
def genetic_algorithm(pop_size, generations):
    population = initialize_population(pop_size)
    for _ in range(generations):
        # 选择父代
        parents = random.sample(population, 2)
        ms1, os1, _ = parents[0]
        ms2, os2, _ = parents[1]
        # 交叉操作
        new_ms1, new_ms2 = crossover(ms1, ms2)
        # 解码新个体
        makespan1 = decode_chromosome(new_ms1.copy(), os1.copy())
        makespan2 = decode_chromosome(new_ms2.copy(), os2.copy())
        # 替换种群中的个体
        population.sort(key=lambda x: x[2], reverse=True)
        population[-1] = (new_ms1, os1, makespan1)
        population[-2] = (new_ms2, os2, makespan2)
    # 返回最优解
    best_solution = min(population, key=lambda x: x[2])
    return best_solution






def read_instance(filename):
    with open(filename) as f:
        lines = f.readlines()

    first_line = lines[0].split()
    # Number of jobs
    nb_jobs = int(first_line[0])
    # Number of machines
    nb_machines = int(first_line[1])

    # Number of operations for each job
    nb_operations = [int(lines[j + 1].split()[0]) for j in range(nb_jobs)]

    # Number of tasks
    nb_tasks = sum(nb_operations[j] for j in range(nb_jobs))

    # Processing time for each task, for each machine
    task_processing_time = [[INFINITE for m in range(nb_machines)] for i in range(nb_tasks)]

    # For each job, for each operation, the corresponding task id
    job_operation_task = [[0 for o in range(nb_operations[j])] for j in range(nb_jobs)]

    id = 0
    for j in range(nb_jobs):
        line = lines[j + 1].split()
        tmp = 0
        for o in range(nb_operations[j]):
            nb_machines_operation = int(line[tmp + o + 1])
            for i in range(nb_machines_operation):
                machine = int(line[tmp + o + 2 * i + 2]) - 1
                time = int(line[tmp + o + 2 * i + 3])
                task_processing_time[id][machine] = time
            job_operation_task[j][o] = id
            id = id + 1
            tmp = tmp + 2 * nb_machines_operation

    # Trivial upper bound for the start times of the tasks
    max_start = sum(
        max(task_processing_time[i][m] for m in range(nb_machines) if task_processing_time[i][m] != INFINITE)
        for i in range(nb_tasks))

    return nb_jobs, nb_machines, nb_tasks, task_processing_time, job_operation_task, nb_operations, max_start





if __name__ == '__main__':
    instance_file = "./instances/Mk01.fjs"
    time_limit = 60
    nb_jobs, nb_machines, nb_tasks, task_processing_time, job_operation_task, nb_operations, max_start = read_instance(instance_file)
    print(nb_jobs, nb_machines, nb_tasks, task_processing_time, job_operation_task, nb_operations, max_start)
    # 运行遗传算法
    # pop_size = 50
    # generations = 100
    # best_ms, best_os, best_makespan = genetic_algorithm(pop_size, generations)
    # print(f"最优机器选择部分: {best_ms}")
    # print(f"最优工序排序部分: {best_os}")
    # print(f"最优最大完工时间: {best_makespan}")



