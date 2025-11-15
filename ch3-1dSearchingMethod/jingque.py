import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from typing import Callable, Tuple, List, Optional, Union, Dict, Any
import warnings
import sympy as sp

class OneDimensionalSearchingMethod:
    def __init__(self, eps: float) -> None:
        self.eps = eps
        self.history = {}
    
    def plot(self, method_name: str, f: Callable, x_range: Tuple[float, float]):
        """
        使用 FuncAnimation 动态绘制搜索过程
        """
        if method_name not in self.history:
            print(f"No history data found for {method_name}")
            return

        history_data = self.history[method_name]
        
        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 确定x轴范围
        if x_range is None:
            if 'intervals' in history_data:
                intervals = history_data['intervals']
                x_min = min([min(interval) for interval in intervals]) - 0.5
                x_max = max([max(interval) for interval in intervals]) + 0.5
            elif 'points' in history_data:
                points = history_data['points']
                x_min = min(points) - 0.5
                x_max = max(points) + 0.5
            else:
                x_min, x_max = -2, 2
        else:
            x_min, x_max = x_range
        
        # 绘制函数曲线
        x_vals = np.linspace(x_min, x_max, 1000)
        y_vals = [f(x) for x in x_vals]
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 根据方法名称调用相应的动画函数
        if method_name == 'golden_splitting':
            self._animate_golden_splitting(ax, history_data, f, x_min, x_max)
        elif method_name == 'success_failure':
            self._animate_success_failure(ax, history_data, f, x_min, x_max)
        elif method_name == 'binary_splitting':
            self._animate_binary_splitting(ax, history_data, f, x_min, x_max)
        elif method_name == 'newton':
            self._animate_newton(ax, history_data, f, x_min, x_max)
        elif method_name == 'twice_interpolation':
            self._animate_twice_interpolation(ax, history_data, f, x_min, x_max)

    def _animate_golden_splitting(self, ax, history_data, f, x_min, x_max):
        """动态展示黄金分割法"""
        points = history_data['points']
        
        # 初始化动态元素
        current_scatter = ax.scatter([], [], c='red', s=80, zorder=5, label='Current Points')
        best_scatter = ax.scatter([], [], c='green', s=100, marker='*', zorder=6, label='Best Point')
        
        # 获取y轴范围用于绘制区间
        y_vals = [f(x) for x in np.linspace(x_min, x_max, 100)]
        y_min, y_max = min(y_vals), max(y_vals)
        interval_height = (y_max - y_min) * 0.1  # 区间显示高度
        
        # 使用Rectangle来显示区间
        interval_rect = Rectangle((0, y_min), 0, interval_height, 
                                alpha=0.2, color='yellow', label='Search Interval')
        ax.add_patch(interval_rect)
        
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        def update(frame):
            # 每4个点代表一次完整的迭代 (a, b, x1, x2)
            iter_num = frame // 4
            if iter_num * 4 + 3 >= len(points):
                return current_scatter, best_scatter, interval_rect, iteration_text
            
            # 获取当前迭代的四个关键点
            a = points[iter_num * 4]
            b = points[iter_num * 4 + 1]
            x1 = points[iter_num * 4 + 2]
            x2 = points[iter_num * 4 + 3]
            
            # 更新当前点
            current_x = [x1, x2]
            current_y = [f(x1), f(x2)]
            current_scatter.set_offsets(np.c_[current_x, current_y])
            
            # 更新搜索区间
            interval_rect.set_xy((a, y_min))
            interval_rect.set_width(b - a)
            
            # 找到当前最佳点
            all_points = points[:frame+1]
            all_y = [f(p) for p in all_points]
            if all_y:
                best_idx = np.argmin(all_y)
                best_x = all_points[best_idx]
                best_y = all_y[best_idx]
                best_scatter.set_offsets([[best_x, best_y]])
            
            iteration_text.set_text(f'Iteration: {iter_num + 1}\n'
                                  f'Interval: [{a:.3f}, {b:.3f}]\n'
                                  f'Length: {b-a:.3f}\n'
                                  f'x1: {x1:.3f}, x2: {x2:.3f}')
            
            return current_scatter, best_scatter, interval_rect, iteration_text

        # 创建动画
        frames = len(points) // 4
        anim = FuncAnimation(ax.figure, update, frames=frames, 
                            interval=1000, repeat_delay=2000, blit=False)
        
        ax.set_title('Golden Section Search - Live Animation', fontsize=14)
        plt.tight_layout()
        plt.show()

    def _animate_success_failure(self, ax, history_data, f, x_min, x_max):
        """动态展示成功失败法"""
        intervals = history_data['intervals']
        
        # 获取y轴范围
        y_vals = [f(x) for x in np.linspace(x_min, x_max, 100)]
        y_min, y_max = min(y_vals), max(y_vals)
        interval_height = (y_max - y_min) * 0.1
        
        # 初始化动态元素
        interval_rect = Rectangle((0, y_min), 0, interval_height, 
                                alpha=0.3, color='red', label='Current Interval')
        ax.add_patch(interval_rect)
        
        test_points = ax.scatter([], [], c='orange', s=60, zorder=5, label='Test Points')
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        def update(frame):
            if frame >= len(intervals):
                return interval_rect, test_points, iteration_text
            
            # 获取当前区间
            a, b = intervals[frame]
            
            # 更新区间显示
            interval_rect.set_xy((a, y_min))
            interval_rect.set_width(b - a)
            
            # 显示测试点（区间端点）
            test_x = [a, b]
            test_y = [f(a), f(b)]
            test_points.set_offsets(np.c_[test_x, test_y])
            
            # 计算区间中点作为当前估计的最优点
            mid_point = (a + b) / 2
            
            iteration_text.set_text(f'Iteration: {frame + 1}\n'
                                  f'Interval: [{a:.3f}, {b:.3f}]\n'
                                  f'Length: {b-a:.3f}\n'
                                  f'Midpoint: {mid_point:.3f}\n'
                                  f'f(mid): {f(mid_point):.3f}')
            
            return interval_rect, test_points, iteration_text

        # 创建动画
        anim = FuncAnimation(ax.figure, update, frames=len(intervals), 
                            interval=800, repeat_delay=2000, blit=False)
        
        ax.set_title('Success-Failure Method - Live Animation', fontsize=14)
        plt.tight_layout()
        plt.show()

    def _animate_binary_splitting(self, ax, history_data, f, x_min, x_max):
        """动态展示二分法"""
        intervals = history_data['intervals']
        
        # 获取y轴范围
        y_vals = [f(x) for x in np.linspace(x_min, x_max, 100)]
        y_min, y_max = min(y_vals), max(y_vals)
        interval_height = (y_max - y_min) * 0.1
        
        # 初始化动态元素
        interval_rect = Rectangle((0, y_min), 0, interval_height, 
                                alpha=0.3, color='purple', label='Current Interval')
        ax.add_patch(interval_rect)
        
        midpoint_line = ax.axvline(0, color='green', linestyle='--', alpha=0.7, label='Midpoint')
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        def update(frame):
            if frame >= len(intervals):
                return interval_rect, midpoint_line, iteration_text
            
            # 获取当前区间
            a, b = intervals[frame]
            mid_point = (a + b) / 2
            
            # 更新区间显示
            interval_rect.set_xy((a, y_min))
            interval_rect.set_width(b - a)
            
            # 更新中点线
            midpoint_line.set_xdata([mid_point])
            
            iteration_text.set_text(f'Iteration: {frame + 1}\n'
                                  f'Interval: [{a:.3f}, {b:.3f}]\n'
                                  f'Midpoint: {mid_point:.3f}\n'
                                  f'f(mid): {f(mid_point):.3f}\n'
                                  f'Interval Length: {b-a:.6f}')
            
            return interval_rect, midpoint_line, iteration_text

        # 创建动画
        anim = FuncAnimation(ax.figure, update, frames=len(intervals), 
                            interval=800, repeat_delay=2000, blit=False)
        
        ax.set_title('Binary Splitting Method - Live Animation', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _animate_newton(self, ax, history_data, f, x_min, x_max):
        """动态展示牛顿法"""
        points = history_data['points']
        
        # 获取y轴范围
        y_vals = [f(x) for x in np.linspace(x_min, x_max, 100)]
        y_min, y_max = min(y_vals), max(y_vals)
        interval_height = (y_max - y_min) * 0.1
        
        # 初始化动态元素
        point_line = ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Current Point')
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        def update(frame):
            if frame >= len(points):
                return point_line, iteration_text
            
            # 获取当前点
            x = points[frame]
            
            # 更新点线
            point_line.set_xdata([x])
            
            iteration_text.set_text(f'Iteration: {frame + 1}\n'
                                  f'Point: {x:.3f}\n'
                                  f'f(x): {f(x):.3f}')
            
            return point_line, iteration_text

        # 创建动画
        anim = FuncAnimation(ax.figure, update, frames=len(points), 
                            interval=800, repeat_delay=2000, blit=False)
        
        ax.set_title('Newton Method - Live Animation', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _animate_twice_interpolation(self, ax, history_data, f, x_min, x_max):
        """动态展示二次插值法"""
        points = history_data['points']
        triples = history_data['triples']
        # 获取y轴范围
        y_vals = [f(x) for x in np.linspace(x_min, x_max, 100)]
        y_min, y_max = min(y_vals), max(y_vals)
        interval_height = (y_max - y_min) * 0.1
        
        # 初始化动态元素
        current_scatter = ax.scatter([], [], c='red', s=80, zorder=5, label='Current Points')
        iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        def update(frame):
            if frame >= len(triples):
                return current_scatter, iteration_text
            
            # 获取当前三个点
            x1, x2, x3 = triples[frame]
            
            # 更新当前点
            current_x = [x1, x2, x3]
            current_x.sort()
            current_y = [f(x1), f(x2), f(x3)]
            current_scatter.set_offsets(np.c_[current_x, current_y])
            
            # current_scatter.set_color(['blue', 'red', 'blue'])
            # current_scatter.set_sizes([60, 100, 60])
            
            iteration_text.set_text(f'Iteration: {frame + 1}\n'
                                  f'Points: {x1:.3f}, {x2:.3f}, {x3:.3f}'
                                  f'\nf(x): {f(x1):.3f}, {f(x2):.3f}, {f(x3):.3f}'
                                  f'\nmin(f(x)): {min(current_y):.3f}')
            
            return current_scatter, iteration_text

        # 创建动画
        frames = len(triples)
        anim = FuncAnimation(ax.figure, update, frames=frames, 
                            interval=1000, repeat_delay=2000, blit=False)
        
        ax.set_title('Twice Interpolation Method - Live Animation', fontsize=14)
        plt.tight_layout()
        plt.show()

    # 原有的搜索方法保持不变
    def success_failure(self,
                        f: Callable,
                        a: float,
                        h: float,
                        max_iter: int = 1000
                        ) -> Tuple[float, float]:
        # 初始化历史记录
        self.history['success_failure'] = {
            'intervals': [],
            'final_interval': None
        }
        
        for i in range(max_iter):
            if f(a + h) < f(a):  # 前进运算
                if f(a + h) < f(a + 3 * h):
                    c = a
                    d = a + 3 * h
                    self.history['success_failure']['final_interval'] = (c, d)
                    self.history['success_failure']['intervals'].append((c, d))
                    return (c, d)
                a = a + h
            elif f(a) < f(a + h):  # 后退运算
                if f(a) < f(a - h / 4):
                    c = a - h / 4
                    d = a + h
                    self.history['success_failure']['final_interval'] = (c, d)
                    self.history['success_failure']['intervals'].append((c, d))
                    return (c, d)
                a = a - h / 4
            else:  # 无法确定方向
                raise ValueError("Choose an appropriate h.")
            
            # 记录当前区间
            current_interval = (a, a + 3 * h)
            self.history['success_failure']['intervals'].append(current_interval)
            h *= 2
            c = a
            d = a + h
        # 超过最大迭代次数
        self.history['success_failure']['final_interval'] = (c, d)
        warnings.warn("Maximum iterations reached without finding an interval.")
        return (c, d)
    
    def golden_splitting(self,
                         f: Callable,
                         a: float,
                         b: float,
                         ) -> float:
        assert a < b, "a must be less than b."
        
        # 初始化历史记录
        self.history['golden_splitting'] = {
            'points': [],
            'final_point': None
        }
        
        t = (np.sqrt(5) - 1) / 2
        x1 = a + (1 - t) * (b - a)
        x2 = a + t * (b - a)
        
        # 记录初始点
        self.history['golden_splitting']['points'].extend([a, b, x1, x2])
        
        while True:
            if f(x1) < f(x2):
                b = x2
                x2 = x1
                x1 = a + (1 - t) * (b - a)
            else:
                a = x1
                x1 = x2
                x2 = a + t * (b - a)
            
            # 记录当前点
            self.history['golden_splitting']['points'].extend([a, b, x1, x2])
            
            if abs(b - a) < self.eps:
                x_final = (a + b) / 2
                self.history['golden_splitting']['final_point'] = x_final
                self.history['golden_splitting']['points'].append(x_final)
                return x_final
    
    def binary_splitting(self,
                         df_dx: Callable,
                         a: float,
                         b: float,
                         ) -> float:
        # 初始化历史记录
        self.history['binary_splitting'] = {
            'intervals': [],
            'final_point': None
        }
        
        # 记录初始区间
        self.history['binary_splitting']['intervals'].append((a, b))
        
        while True:
            x0 = (a + b) / 2
            derivative = df_dx(x0)
            
            if derivative < 0:
                a = x0
            elif derivative > 0:
                b = x0
            else:
                self.history['binary_splitting']['final_point'] = x0
                return x0
            
            # 记录当前区间
            self.history['binary_splitting']['intervals'].append((a, b))
            
            if abs(b - a) < self.eps:
                x_final = (a + b) / 2
                self.history['binary_splitting']['final_point'] = x_final
                return x_final
    
    def newton(self,
               f: Callable,
               df_dx: Callable,
               d2f_dx2: Callable,
               x1: float,
               ):
        # 初始化历史记录
        self.history['newton'] = {
            'points': [x1],
            'final_point': None
        }
        while True:
            x2 = x1 - df_dx(x1) / d2f_dx2(x1)
            # 记录当前点
            self.history['newton']['points'].append(x2)
            if abs(x2 - x1) < self.eps:
                self.history['newton']['final_point'] = x2
                return x2
            x1 = x2
    
    def twice_interpolation(self,
                             f: Callable,
                             x1: float,
                             x2: float,
                             x3: float,
                             ):
        assert not f(x1) < f(x2) or f(x2) > f(x3), "不满足两头大中间小"
        self.history['twice_interpolation'] = {
            'points': sorted([x1, x2, x3]),
            'triples': [sorted([x1, x2, x3])],
            'final_point': None
        }
        while True:
            A = np.array([[1, x1, x1 ** 2], [1, x2, x2 ** 2], [1, x3, x3 ** 2]])
            b = np.array([f(x1), f(x2), f(x3)])
            X = np.linalg.solve(A, b)
            a0 = X[0]
            a1 = X[1]
            a2 = X[2]
            x = -a1 / (2 * a2)
            if abs(x - x2) < self.eps:
                final_point = x if f(x) < f(x2) else x2
                self.history['twice_interpolation']['final_point'] = final_point
                final_triple  = sorted([x1, x2, x])
                if final_triple not in self.history['twice_interpolation']['triples']:
                    self.history['twice_interpolation']['triples'].append(final_triple)
                if final_point not in self.history['twice_interpolation']['points']:
                    self.history['twice_interpolation']['points'].append(final_point)
                    self.history['twice_interpolation']['points'].sort()
                return final_point
            arr = [x1, x2, x3, x]
            arr.sort()
            if f(x2) > f(x):
                # 记录当前点
                index = arr.index(x)
                x2 = x
                x1 = arr[max(0, index - 1)]
                x3 = arr[min(len(arr) - 1, index + 1)]
            else:
                # 记录当前点
                index = arr.index(x2)
                x1 = arr[max(0, index - 1)]
                x3 = arr[min(len(arr) - 1, index + 1)]
            new_triple = sorted([x1, x2, x3])
            if new_triple not in self.history['twice_interpolation']['triples']:
                self.history['twice_interpolation']['triples'].append(new_triple)
            for point in arr:
                if point not in self.history['twice_interpolation']['points']:
                    self.history['twice_interpolation']['points'].append(point)
            self.history['twice_interpolation']['points'].sort()
    


def main():
    method = OneDimensionalSearchingMethod(eps=0.2)
    
    def f():
        x = sp.symbols('x')
        # fx_expr = x ** 3 - 2 * x + 1
        # fx_expr = x ** 4 - 4 * x ** 3 - 6 * x ** 2 - 16 * x + 4
        fx_expr = 3 * x ** 3 - 4 * x + 2
        # fx_expr = sp.sympify(input("请输入函数表达式："))
        fx = sp.lambdify(x, fx_expr, 'numpy')
        df_dx_expr = sp.diff(fx_expr, x)
        df_dx = sp.lambdify(x, df_dx_expr, 'numpy')
        d2f_dx2_expr = sp.diff(df_dx_expr, x)
        d2f_dx2 = sp.lambdify(x, d2f_dx2_expr, 'numpy')
        return fx, df_dx, d2f_dx2
    
    fx, df_dx, d2f_dx2 = f()
    
    print("Testing Success-Failure Method:")
    a = -1 / 2
    h = 1 / 2
    interval = method.success_failure(fx, a, h)
    print(f"Found interval: {interval}")
    # method.plot('success_failure', fx, x_range=(-2, 2))
    
    print("\nTesting Golden Section Search:")
    x_final_golden = method.golden_splitting(fx, 0, 2)
    print(f"Found minimum at x = {x_final_golden}, f(x) = {fx(x_final_golden)}")
    # method.plot('golden_splitting', fx, x_range=(0, 2))
    
    print("\nTesting Binary Search:")
    x_final_binary = method.binary_splitting(df_dx, 0, 2)
    print(f"Found minimum at x = {x_final_binary}, f(x) = {fx(x_final_binary)}")
    # method.plot('binary_splitting', fx, x_range=(0, 2))
    
    print("\nTesting Newton's Method:")
    x_final_newton = method.newton(fx, df_dx, d2f_dx2, 6)
    print(f"Found minimum at x = {x_final_newton}, f(x) = {fx(x_final_newton)}")
    # method.plot('newton', fx, x_range=(0, 5))
    
    print("\nTesting Twice Interpolation:")
    x_final_twice = method.twice_interpolation(fx, 0, 1, 3)
    print(f"Found minimum at x = {x_final_twice}, f(x) = {fx(x_final_twice)}")
    method.plot('twice_interpolation', fx, x_range=(0, 4))


if __name__ == '__main__':
    main()