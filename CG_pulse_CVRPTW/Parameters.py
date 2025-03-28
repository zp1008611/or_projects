import cplex

class Parameters:
    capacity = 200.0

    class ColGen:
        abort = False
        zero_reduced_cost = -0.0001
        zero_reduced_cost_AbortColGen = -0.005
        subproblemTiLim = 5000
        subproblemObjVal = -1000
        M = 10000
        boundStep = 4

    @staticmethod
    def configure_copt(masterproblem):
        try:
            # 设置分支定界策略
            masterproblem.copt.parameters.mip.strategy.nodeselect.set(1)
            masterproblem.copt.parameters.mip.strategy.branch.set(1)
            
            # 显示选项
            masterproblem.copt.parameters.mip.display.set(2)
            masterproblem.copt.parameters.tune.display.set(1)
            masterproblem.copt.parameters.simplex.display.set(0)
            
            # 其他可能的参数配置
            # masterproblem.cplex.parameters.preprocessing.presolve.set(True)
            
        except copt.exceptions.CplexError as e:
            print(f"CPLEX配置错误: {e}")