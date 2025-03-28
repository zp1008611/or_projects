import coptpy as copt

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
            masterproblem.model.setParam(copt.COPT.Param.NodeSel, 1)  # 节点选择策略
            masterproblem.model.setParam(copt.COPT.Param.BranchStrat, 1)  # 分支策略
            
            # 显示选项
            masterproblem.model.setParam(copt.COPT.Param.MIPDisplay, 2)  # MIP显示级别
            masterproblem.model.setParam(copt.COPT.Param.LPMethod, 0)  # LP方法（单纯形法）
            
            # 求解限制
            masterproblem.model.setParam(copt.COPT.Param.TimeLimit, 3600)  # 时间限制：1小时
            masterproblem.model.setParam(copt.COPT.Param.RelGap, 1e-4)  # 相对间隙
            
            # 预处理设置
            masterproblem.model.setParam(copt.COPT.Param.Presolve, 1)  # 启用预处理
            
        except copt.CoptError as e:
            print(f"COPT配置错误: {e}")