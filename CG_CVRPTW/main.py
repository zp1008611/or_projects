# 假设 ColumnGen 类已经在某个模块中定义
from Algorithm import ColumnGen

def main():
    # 创建 ColumnGen 类的实例
    col_gen = ColumnGen("input/Solomon/50_customer/c101.txt")
    # 调用 runColumnGeneration 方法
    col_gen.runColumnGeneration()

if __name__ == "__main__":
    main()
    