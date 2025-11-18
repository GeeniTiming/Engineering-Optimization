import numpy as np
import sympy as sp
from typing import List, Tuple

class NewtonOptimizer:
    """牛顿法优化器（精确符号线搜索）"""
    
    def __init__(self, f_expr: sp.Expr, variables: List[sp.Symbol], epsilon: float = 0.1):
        self.f_expr = f_expr
        self.variables = variables
        self.epsilon = epsilon
        self.dim = len(variables)
        
        # 符号计算
        self.grad_expr = [sp.diff(f_expr, var) for var in variables]
        self.hessian_expr = sp.hessian(f_expr, variables)
        self.alpha_sym = sp.symbols('alpha')
        
        # 数值函数
        self.f_func = sp.lambdify(variables, f_expr, 'numpy')
        self.grad_func = sp.lambdify(variables, self.grad_expr, 'numpy')
        self.hessian_func = sp.lambdify(variables, self.hessian_expr, 'numpy')
    
    def exact_line_search(self, x: np.ndarray, d: np.ndarray) -> float:
        """精确线搜索：符号求解最优步长"""
        # 构建 φ(α) = f(x + αd)
        x_alpha = [x[i] + self.alpha_sym * d[i] for i in range(self.dim)]
        phi_alpha = self.f_expr.subs(dict(zip(self.variables, x_alpha)))
        
        # 求解 dφ/dα = 0
        dphi_dalpha = sp.diff(phi_alpha, self.alpha_sym)
        solutions = sp.solve(dphi_dalpha, self.alpha_sym)
        
        # 选择实数解
        for sol in solutions:
            if sol.is_real:
                return float(sol)
        
        raise ValueError("无实数解")
    
    def optimize(self, x0: np.ndarray, max_iter: int = 100) -> Tuple[np.ndarray, float]:
        """执行优化"""
        x = np.array(x0, dtype=float)
        
        print("阻尼牛顿法优化过程:")
        print(f"{'迭代':>3} {'x':>20} {'f(x)':>12} {'||∇f||':>12}")
        print("-" * 60)
        
        for k in range(max_iter):
            f_val = self.f_func(*x)
            grad = np.array(self.grad_func(*x))
            grad_norm = np.linalg.norm(grad)
            
            print(f"{k:3d} {str(x):>20} {f_val:12.6f} {grad_norm:12.6f}")
            
            if grad_norm < self.epsilon:
                print(f"\n收敛于第 {k} 次迭代")
                break
            
            # 计算牛顿方向
            H = np.array(self.hessian_func(*x), dtype=float)
            d = np.linalg.solve(H, -grad)
            
            # 精确线搜索
            alpha = self.exact_line_search(x, d)
            
            # 更新
            x = x + alpha * d
        
        return x, self.f_func(*x)

# 使用示例
if __name__ == "__main__":
    # 定义问题
    x1, x2 = sp.symbols('x1 x2')
    f = x1 + x2 ** 2 + x1 ** 4 + 2 * (x1 ** 2) * (x2 ** 2) + 8 * (x1 ** 2) * (x2 ** 6)
    print(f'f = {f}')
    # 创建优化器
    optimizer = NewtonOptimizer(f, [x1, x2], epsilon=0.1)
    
    # 执行优化
    x0 = np.array([1.0, 1.0])
    x_opt, f_opt = optimizer.optimize(x0)
    
    print(f"\n最优解: x* = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
    print(f"最优值: f(x*) = {f_opt:.6f}")
    # 代入一个点
    x_test = np.array([-0.6344, -0.0032])
    f_test = optimizer.f_func(*x_test)
    print(f"\n在点 x = ({x_test[0]:.6f}, {x_test[1]:.6f}) 处的函数值: f(x) = {f_test:.6f}")