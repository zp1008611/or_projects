import coptpy as cp

# 创建一个线性规划模型
env = cp.Envr()
model = env.createModel("example_model")

# 创建变量并命名
x = model.addVar(name="x")
y = model.addVar(name="y")

# 设置目标函数
model.setObjective(2 * x + 3 * y, sense=cp.COPT.MAXIMIZE)

# 添加约束条件并命名
constraint1 = model.addConstr(x + y <= 10, name="1")
constraint2 = model.addConstr(2 * x + y <= 15, name="2")


# 保存模型为LP文件
try:
    model.write("example_model.lp")
    print("模型已成功保存为 example_model.lp")
except Exception as e:
    print(f"保存模型时出错: {e}")    


env = cp.Envr()
model = env.createModel("fractional")
model.read("example_model.lp")

for constr in model.getConstrs():
    print("yes")
    print(constr.name)