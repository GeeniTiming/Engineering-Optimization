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
            if "intervals" in history_data:
                intervals = history_data["intervals"]
                x_min = min([min(interval) for interval in intervals]) - 0.5
                x_max = max([max(interval) for interval in intervals]) + 0.5
            elif "points" in history_data:
                points = history_data["points"]
                x_min = min(points) - 0.5
                x_max = max(points) + 0.5
            elif "trials" in history_data:
                trials = history_data["trials"]
                x_min = min(trials) - 0.5
                x_max = max(trials) + 0.5
            else:
                x_min, x_max = -2, 2
        else:
            x_min, x_max = x_range

        # 绘制函数曲线
        x_vals = np.linspace(x_min, x_max, 1000)
        y_vals = [f(x) for x in x_vals]
        ax.plot(x_vals, y_vals, "b-", linewidth=2, label="f(x)")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 根据方法名称调用相应的动画函数
        if method_name == "golden_splitting":
            self._animate_golden_splitting(ax, history_data, f, x_min, x_max)
        elif method_name == "success_failure":
            self._animate_success_failure(ax, history_data, f, x_min, x_max)
        elif method_name == "binary_splitting":
            self._animate_binary_splitting(ax, history_data, f, x_min, x_max)
        elif method_name == "newton":
            self._animate_newton(ax, history_data, f, x_min, x_max)
        elif method_name == "twice_interpolation":
            self._animate_twice_interpolation(ax, history_data, f, x_min, x_max)
        elif method_name in ("armijo_goldstein", "wolfe_powell"):
            self._animate_inexact_line_search(ax, history_data, f, x_min, x_max)

    def _animate_golden_splitting(self, ax, history_data, f, x_min, x_max):
        """动态展示黄金分割法"""
        points = history_data["points"]

        # 初始化动态元素
        current_scatter = ax.scatter(
            [], [], c="red", s=80, zorder=5, label="Current Points"
        )
        best_scatter = ax.scatter(
            [], [], c="green", s=100, marker="*", zorder=6, label="Best Point"
        )

        # 获取y轴范围用于绘制区间
        y_vals = [f(x) for x in np.linspace(x_min, x_max, 100)]
        y_min, y_max = min(y_vals), max(y_vals)
        interval_height = (y_max - y_min) * 0.1  # 区间显示高度

        # 使用Rectangle来显示区间
        interval_rect = Rectangle(
            (0, y_min),
            0,
            interval_height,
            alpha=0.2,
            color="yellow",
            label="Search Interval",
        )
        ax.add_patch(interval_rect)

        iteration_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

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
            all_points = points[: frame + 1]
            all_y = [f(p) for p in all_points]
            if all_y:
                best_idx = np.argmin(all_y)
                best_x = all_points[best_idx]
                best_y = all_y[best_idx]
                best_scatter.set_offsets([[best_x, best_y]])

            iteration_text.set_text(
                f"Iteration: {iter_num + 1}\n"
                f"Interval: [{a:.3f}, {b:.3f}]\n"
                f"Length: {b-a:.3f}\n"
                f"x1: {x1:.3f}, x2: {x2:.3f}"
            )

            return current_scatter, best_scatter, interval_rect, iteration_text

        # 创建动画
        frames = len(points) // 4
        anim = FuncAnimation(
            ax.figure,
            update,
            frames=frames,
            interval=1000,
            repeat_delay=2000,
            blit=False,
        )

        ax.set_title("Golden Section Search - Live Animation", fontsize=14)
        plt.tight_layout()
        plt.show()

    def _animate_success_failure(self, ax, history_data, f, x_min, x_max):
        """动态展示成功失败法"""
        intervals = history_data["intervals"]

        # 获取y轴范围
        y_vals = [f(x) for x in np.linspace(x_min, x_max, 100)]
        y_min, y_max = min(y_vals), max(y_vals)
        interval_height = (y_max - y_min) * 0.1

        # 初始化动态元素
        interval_rect = Rectangle(
            (0, y_min),
            0,
            interval_height,
            alpha=0.3,
            color="red",
            label="Current Interval",
        )
        ax.add_patch(interval_rect)

        test_points = ax.scatter(
            [], [], c="orange", s=60, zorder=5, label="Test Points"
        )
        iteration_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

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

            iteration_text.set_text(
                f"Iteration: {frame + 1}\n"
                f"Interval: [{a:.3f}, {b:.3f}]\n"
                f"Length: {b-a:.3f}\n"
                f"Midpoint: {mid_point:.3f}\n"
                f"f(mid): {f(mid_point):.3f}"
            )

            return interval_rect, test_points, iteration_text

        # 创建动画
        anim = FuncAnimation(
            ax.figure,
            update,
            frames=len(intervals),
            interval=800,
            repeat_delay=2000,
            blit=False,
        )

        ax.set_title("Success-Failure Method - Live Animation", fontsize=14)
        plt.tight_layout()
        plt.show()

    def _animate_binary_splitting(self, ax, history_data, f, x_min, x_max):
        """动态展示二分法"""
        intervals = history_data["intervals"]

        # 获取y轴范围
        y_vals = [f(x) for x in np.linspace(x_min, x_max, 100)]
        y_min, y_max = min(y_vals), max(y_vals)
        interval_height = (y_max - y_min) * 0.1

        # 初始化动态元素
        interval_rect = Rectangle(
            (0, y_min),
            0,
            interval_height,
            alpha=0.3,
            color="purple",
            label="Current Interval",
        )
        ax.add_patch(interval_rect)

        midpoint_line = ax.axvline(
            0, color="green", linestyle="--", alpha=0.7, label="Midpoint"
        )
        iteration_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

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

            iteration_text.set_text(
                f"Iteration: {frame + 1}\n"
                f"Interval: [{a:.3f}, {b:.3f}]\n"
                f"Midpoint: {mid_point:.3f}\n"
                f"f(mid): {f(mid_point):.3f}\n"
                f"Interval Length: {b-a:.6f}"
            )

            return interval_rect, midpoint_line, iteration_text

        # 创建动画
        anim = FuncAnimation(
            ax.figure,
            update,
            frames=len(intervals),
            interval=800,
            repeat_delay=2000,
            blit=False,
        )

        ax.set_title("Binary Splitting Method - Live Animation", fontsize=14)
        plt.tight_layout()
        plt.show()

    def _animate_newton(self, ax, history_data, f, x_min, x_max):
        """动态展示牛顿法"""
        points = history_data["points"]

        # 获取y轴范围
        y_vals = [f(x) for x in np.linspace(x_min, x_max, 100)]
        y_min, y_max = min(y_vals), max(y_vals)
        interval_height = (y_max - y_min) * 0.1

        # 初始化动态元素
        point_line = ax.axvline(
            0, color="red", linestyle="--", alpha=0.7, label="Current Point"
        )
        iteration_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        def update(frame):
            if frame >= len(points):
                return point_line, iteration_text

            # 获取当前点
            x = points[frame]

            # 更新点线
            point_line.set_xdata([x])

            iteration_text.set_text(
                f"Iteration: {frame + 1}\n" f"Point: {x:.3f}\n" f"f(x): {f(x):.3f}"
            )

            return point_line, iteration_text

        # 创建动画
        anim = FuncAnimation(
            ax.figure,
            update,
            frames=len(points),
            interval=800,
            repeat_delay=2000,
            blit=False,
        )

        ax.set_title("Newton Method - Live Animation", fontsize=14)
        plt.tight_layout()
        plt.show()

    def _animate_twice_interpolation(self, ax, history_data, f, x_min, x_max):
        """动态展示二次插值法"""
        points = history_data["points"]
        triples = history_data["triples"]
        # 获取y轴范围
        y_vals = [f(x) for x in np.linspace(x_min, x_max, 100)]
        y_min, y_max = min(y_vals), max(y_vals)
        interval_height = (y_max - y_min) * 0.1

        # 初始化动态元素
        current_scatter = ax.scatter(
            [], [], c="red", s=80, zorder=5, label="Current Points"
        )
        iteration_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

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

            iteration_text.set_text(
                f"Iteration: {frame + 1}\n"
                f"Points: {x1:.3f}, {x2:.3f}, {x3:.3f}"
                f"\nf(x): {f(x1):.3f}, {f(x2):.3f}, {f(x3):.3f}"
                f"\nmin(f(x)): {min(current_y):.3f}"
            )

            return current_scatter, iteration_text

        # 创建动画
        frames = len(triples)
        anim = FuncAnimation(
            ax.figure,
            update,
            frames=frames,
            interval=1000,
            repeat_delay=2000,
            blit=False,
        )

        ax.set_title("Twice Interpolation Method - Live Animation", fontsize=14)
        plt.tight_layout()
        plt.show()

    def _animate_inexact_line_search(self, ax, history_data, f, x_min, x_max):
        """动态展示非精确一维搜索(Armijo/Goldstein/Wolfe)"""
        trials = history_data["trials"]
        base_point = history_data["base_point"]
        direction = history_data["direction"]
        accepted_point = history_data.get("accepted")
        accepted_index = history_data.get("accepted_index", None)

        base_scatter = ax.scatter(
            [base_point], [f(base_point)], c="black", s=70, zorder=5, label="Base Point"
        )
        trial_scatter = ax.scatter(
            [], [], c="orange", s=70, zorder=6, label="Trial Point"
        )
        accepted_scatter = ax.scatter(
            [], [], c="green", s=120, marker="*", zorder=7, label="Accepted Point"
        )

        iteration_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        def update(frame):
            if frame >= len(trials):
                return trial_scatter, accepted_scatter, iteration_text

            x = trials[frame]
            trial_scatter.set_offsets([[x, f(x)]])

            if (
                accepted_point is not None
                and accepted_index is not None
                and frame >= accepted_index
            ):
                accepted_scatter.set_offsets([[accepted_point, f(accepted_point)]])

            if direction != 0:
                alpha = (x - base_point) / direction
            else:
                alpha = 0.0

            iteration_text.set_text(
                f"Iteration: {frame + 1}\n"
                f"Trial x: {x:.3f}\n"
                f"alpha: {alpha:.3f}\n"
                f"f(x): {f(x):.3f}"
            )

            return trial_scatter, accepted_scatter, iteration_text

        anim = FuncAnimation(
            ax.figure,
            update,
            frames=len(trials),
            interval=800,
            repeat_delay=2000,
            blit=False,
        )

        title = history_data.get("title", "Inexact Line Search - Live Animation")
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    def _line_search_setup(
        self, df_dx: Callable, x0: float, direction: Optional[float]
    ) -> Tuple[float, float]:
        if direction is None:
            direction = -df_dx(x0)
        g0 = df_dx(x0) * direction
        if g0 >= 0:
            raise ValueError("Direction is not a descent direction.")
        return direction, g0

    def armijo_goldstein(
        self,
        f: Callable,
        df_dx: Callable,
        x0: float,
        alpha0: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iter: int = 50,
        direction: Optional[float] = None,
    ) -> Tuple[float, float]:
        if not (0 < c1 < c2 < 1):
            raise ValueError("Require 0 < c1 < c2 < 1.")

        direction, g0 = self._line_search_setup(df_dx, x0, direction)
        phi0 = f(x0)

        self.history["armijo_goldstein"] = {
            "trials": [],
            "accepted": None,
            "accepted_index": None,
            "base_point": x0,
            "direction": direction,
            "title": "Armijo-Goldstein Line Search - Live Animation",
        }

        alpha_low = 0.0
        alpha_high = None
        alpha = alpha0

        for i in range(max_iter):
            x_trial = x0 + alpha * direction
            phi_a = f(x_trial)
            self.history["armijo_goldstein"]["trials"].append(x_trial)

            if phi_a > phi0 + c1 * alpha * g0:
                alpha_high = alpha
            elif phi_a < phi0 + c2 * alpha * g0:
                alpha_low = alpha
            else:
                self.history["armijo_goldstein"]["accepted"] = x_trial
                self.history["armijo_goldstein"]["accepted_index"] = (
                    len(self.history["armijo_goldstein"]["trials"]) - 1
                )
                return alpha, x_trial

            if alpha_high is None:
                alpha *= 2.0
            else:
                alpha = 0.5 * (alpha_low + alpha_high)

        x_final = x0 + alpha * direction
        self.history["armijo_goldstein"]["accepted"] = x_final
        self.history["armijo_goldstein"]["accepted_index"] = max_iter - 1
        return alpha, x_final

    def wolfe_powell(
        self,
        f: Callable,
        df_dx: Callable,
        x0: float,
        alpha0: float = 1.0,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iter: int = 50,
        direction: Optional[float] = None,
    ) -> Tuple[float, float]:
        if not (0 < c1 < c2 < 1):
            raise ValueError("Require 0 < c1 < c2 < 1.")

        direction, g0 = self._line_search_setup(df_dx, x0, direction)
        phi0 = f(x0)

        self.history["wolfe_powell"] = {
            "trials": [],
            "accepted": None,
            "accepted_index": None,
            "base_point": x0,
            "direction": direction,
            "title": "Wolfe-Powell Line Search - Live Animation",
        }

        def phi(alpha):
            return f(x0 + alpha * direction)

        def dphi(alpha):
            return df_dx(x0 + alpha * direction) * direction

        def zoom(alpha_lo: float, alpha_hi: float) -> Tuple[float, float]:
            if alpha_hi < alpha_lo:
                alpha_lo, alpha_hi = alpha_hi, alpha_lo
            phi_lo = phi(alpha_lo)
            for _ in range(max_iter):
                alpha_j = 0.5 * (alpha_lo + alpha_hi)
                x_trial = x0 + alpha_j * direction
                self.history["wolfe_powell"]["trials"].append(x_trial)
                phi_j = phi(alpha_j)

                if (phi_j > phi0 + c1 * alpha_j * g0) or (phi_j >= phi_lo):
                    alpha_hi = alpha_j
                else:
                    dphi_j = dphi(alpha_j)
                    if abs(dphi_j) <= -c2 * g0:
                        return alpha_j, x_trial
                    if dphi_j * (alpha_hi - alpha_lo) >= 0:
                        alpha_hi = alpha_lo
                    alpha_lo = alpha_j
                    phi_lo = phi_j

            x_final = x0 + alpha_j * direction
            return alpha_j, x_final

        alpha_prev = 0.0
        phi_prev = phi0
        alpha = alpha0

        for i in range(max_iter):
            x_trial = x0 + alpha * direction
            self.history["wolfe_powell"]["trials"].append(x_trial)
            phi_a = phi(alpha)

            if (phi_a > phi0 + c1 * alpha * g0) or (i > 0 and phi_a >= phi_prev):
                alpha_star, x_star = zoom(alpha_prev, alpha)
                self.history["wolfe_powell"]["accepted"] = x_star
                self.history["wolfe_powell"]["accepted_index"] = (
                    len(self.history["wolfe_powell"]["trials"]) - 1
                )
                return alpha_star, x_star

            dphi_a = dphi(alpha)
            if abs(dphi_a) <= -c2 * g0:
                self.history["wolfe_powell"]["accepted"] = x_trial
                self.history["wolfe_powell"]["accepted_index"] = (
                    len(self.history["wolfe_powell"]["trials"]) - 1
                )
                return alpha, x_trial

            if dphi_a >= 0:
                alpha_star, x_star = zoom(alpha, alpha_prev)
                self.history["wolfe_powell"]["accepted"] = x_star
                self.history["wolfe_powell"]["accepted_index"] = (
                    len(self.history["wolfe_powell"]["trials"]) - 1
                )
                return alpha_star, x_star

            alpha_prev = alpha
            phi_prev = phi_a
            alpha *= 2.0

        x_final = x0 + alpha * direction
        self.history["wolfe_powell"]["accepted"] = x_final
        self.history["wolfe_powell"]["accepted_index"] = max_iter - 1
        return alpha, x_final

    # 原有的搜索方法保持不变
    def success_failure(
        self, f: Callable, a: float, h: float, max_iter: int = 1000
    ) -> Tuple[float, float]:
        # 初始化历史记录
        self.history["success_failure"] = {"intervals": [], "final_interval": None}

        for i in range(max_iter):
            if f(a + h) < f(a):  # 前进运算
                if f(a + h) < f(a + 3 * h):
                    c = a
                    d = a + 3 * h
                    self.history["success_failure"]["final_interval"] = (c, d)
                    self.history["success_failure"]["intervals"].append((c, d))
                    return (c, d)
                a = a + h
            elif f(a) < f(a + h):  # 后退运算
                if f(a) < f(a - h / 4):
                    c = a - h / 4
                    d = a + h
                    self.history["success_failure"]["final_interval"] = (c, d)
                    self.history["success_failure"]["intervals"].append((c, d))
                    return (c, d)
                a = a - h / 4
            else:  # 无法确定方向
                raise ValueError("Choose an appropriate h.")

            # 记录当前区间
            current_interval = (a, a + 3 * h)
            self.history["success_failure"]["intervals"].append(current_interval)
            h *= 2
            c = a
            d = a + h
        # 超过最大迭代次数
        self.history["success_failure"]["final_interval"] = (c, d)
        warnings.warn("Maximum iterations reached without finding an interval.")
        return (c, d)

    def golden_splitting(
        self,
        f: Callable,
        a: float,
        b: float,
    ) -> float:
        assert a < b, "a must be less than b."

        # 初始化历史记录
        self.history["golden_splitting"] = {"points": [], "final_point": None}

        t = (np.sqrt(5) - 1) / 2
        x1 = a + (1 - t) * (b - a)
        x2 = a + t * (b - a)

        # 记录初始点
        print(f"Initial points: a={a:.3f}, b={b:.3f}, x1={x1:.3f}, x2={x2:.3f}")
        self.history["golden_splitting"]["points"].extend([a, b, x1, x2])

        while True:
            if f(x1) < f(x2):
                b = x2
                x2 = x1
                x1 = a + (1 - t) * (b - a)
                print(
                    f"Step {len(self.history['golden_splitting']['points']) // 4}: "
                    f"a={a:.3f}, b={b:.3f}, x1={x1:.3f}, x2={x2:.3f}"
                )
            else:
                a = x1
                x1 = x2
                x2 = a + t * (b - a)
                print(
                    f"Step {len(self.history['golden_splitting']['points']) // 4}: "
                    f"a={a:.3f}, b={b:.3f}, x1={x1:.3f}, x2={x2:.3f}"
                )

            # 记录当前点
            self.history["golden_splitting"]["points"].extend([a, b, x1, x2])

            if abs(b - a) < self.eps:
                x_final = (a + b) / 2
                self.history["golden_splitting"]["final_point"] = x_final
                self.history["golden_splitting"]["points"].append(x_final)
                return x_final

    def binary_splitting(
        self,
        df_dx: Callable,
        a: float,
        b: float,
    ) -> float:
        # 初始化历史记录
        self.history["binary_splitting"] = {"intervals": [], "final_point": None}

        # 记录初始区间
        print(f"Initial interval: a={a:.3f}, b={b:.3f}")
        self.history["binary_splitting"]["intervals"].append((a, b))

        while True:
            x0 = (a + b) / 2
            derivative = df_dx(x0)

            if derivative < 0:
                a = x0
            elif derivative > 0:
                b = x0
            else:
                self.history["binary_splitting"]["final_point"] = x0
                return x0

            # 记录当前区间
            print(
                f"Step {len(self.history['binary_splitting']['intervals'])}: "
                f"a={a:.3f}, b={b:.3f}, x0={x0:.3f}, df_dx={derivative:.3f}"
            )
            self.history["binary_splitting"]["intervals"].append((a, b))

            if abs(b - a) < self.eps:
                x_final = (a + b) / 2
                self.history["binary_splitting"]["final_point"] = x_final
                return x_final

    def newton(
        self,
        f: Callable,
        df_dx: Callable,
        d2f_dx2: Callable,
        x1: float,
    ):
        # 初始化历史记录
        self.history["newton"] = {"points": [x1], "final_point": None}
        while True:
            x2 = x1 - df_dx(x1) / d2f_dx2(x1)
            # 记录当前点
            self.history["newton"]["points"].append(x2)
            print(
                f"Step {len(self.history['newton']['points']) - 1}: "
                f"x1={x1:.3f}, x2={x2:.3f}, df_dx(x2)={df_dx(x2):.3f}"
            )
            if df_dx(x2) < self.eps:
                self.history["newton"]["final_point"] = x2
                return x2
            x1 = x2

    def twice_interpolation(
        self,
        f: Callable,
        x1: float,
        x2: float,
        x3: float,
    ):
        assert not f(x1) < f(x2) or f(x2) > f(x3), "不满足两头大中间小"
        self.history["twice_interpolation"] = {
            "points": sorted([x1, x2, x3]),
            "triples": [sorted([x1, x2, x3])],
            "final_point": None,
        }
        while True:
            A = np.array([[1, x1, x1**2], [1, x2, x2**2], [1, x3, x3**2]])
            b = np.array([f(x1), f(x2), f(x3)])
            X = np.linalg.solve(A, b)
            a0 = X[0]
            a1 = X[1]
            a2 = X[2]
            x = -a1 / (2 * a2)
            if abs(x - x2) < self.eps:
                final_point = x if f(x) < f(x2) else x2
                self.history["twice_interpolation"]["final_point"] = final_point
                final_triple = sorted([x1, x2, x])
                if final_triple not in self.history["twice_interpolation"]["triples"]:
                    self.history["twice_interpolation"]["triples"].append(final_triple)
                if final_point not in self.history["twice_interpolation"]["points"]:
                    self.history["twice_interpolation"]["points"].append(final_point)
                    self.history["twice_interpolation"]["points"].sort()
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
            if new_triple not in self.history["twice_interpolation"]["triples"]:
                self.history["twice_interpolation"]["triples"].append(new_triple)
            for point in arr:
                if point not in self.history["twice_interpolation"]["points"]:
                    self.history["twice_interpolation"]["points"].append(point)
            self.history["twice_interpolation"]["points"].sort()


def test_success_failure(
    method: OneDimensionalSearchingMethod,
    fx: Callable,
):
    print("Testing Success-Failure Method:")
    a = -1 / 2
    h = 1 / 2
    interval = method.success_failure(fx, a, h)
    print(f"Found interval: {interval}")
    # method.plot('success_failure', fx, x_range=(-2, 2))


def test_golden_splitting(
    method: OneDimensionalSearchingMethod,
    fx: Callable,
):
    print("Testing Golden Section Search:")
    x_final_golden = method.golden_splitting(fx, 0, 2)
    print(f"Found minimum at x = {x_final_golden}, f(x) = {fx(x_final_golden)}")
    # method.plot('golden_splitting', fx, x_range=(0, 2))


def test_binary_splitting(
    method: OneDimensionalSearchingMethod,
    fx: Callable,
    df_dx: Callable,
):
    print("Testing Binary Search:")
    x_final_binary = method.binary_splitting(df_dx, 0, 2)
    print(f"Found minimum at x = {x_final_binary}, f(x) = {fx(x_final_binary)}")
    # method.plot('binary_splitting', fx, x_range=(0, 2))


def test_newton(
    method: OneDimensionalSearchingMethod,
    fx: Callable,
    df_dx: Callable,
    d2f_dx2: Callable,
    x1: float,
):
    print("Testing Newton's Method:")
    x_final_newton = method.newton(fx, df_dx, d2f_dx2, x1)
    print(f"Found minimum at x = {x_final_newton}, f(x) = {fx(x_final_newton)}")
    # method.plot('newton', fx, x_range=(0, 5))


def test_twice_interpolation(
    method: OneDimensionalSearchingMethod,
    fx: Callable,
):
    print("Testing Twice Interpolation:")
    x_final_twice = method.twice_interpolation(fx, 0, 1, 3)
    print(f"Found minimum at x = {x_final_twice}, f(x) = {fx(x_final_twice)}")
    method.plot("twice_interpolation", fx, x_range=(0, 4))


def main():
    method = OneDimensionalSearchingMethod(eps=1e-2)

    def f():
        x = sp.symbols("x")
        # fx_expr = x ** 3 - 2 * x + 1
        # fx_expr = x ** 4 - 4 * x ** 3 - 6 * x ** 2 - 16 * x + 4
        fx_expr = 3 * x**3 - 4 * x + 2
        # fx_expr = sp.sympify(input("请输入函数表达式："))
        fx = sp.lambdify(x, fx_expr, "numpy")
        df_dx_expr = sp.diff(fx_expr, x)
        df_dx = sp.lambdify(x, df_dx_expr, "numpy")
        d2f_dx2_expr = sp.diff(df_dx_expr, x)
        d2f_dx2 = sp.lambdify(x, d2f_dx2_expr, "numpy")
        print(f"Function: {fx_expr}")
        return fx, df_dx, d2f_dx2

    fx, df_dx, d2f_dx2 = f()

    # test_success_failure(method, fx)

    # test_golden_splitting(method, fx)

    # test_binary_splitting(method, fx, df_dx)

    # test_newton(method, fx, df_dx, d2f_dx2, 6)

    test_twice_interpolation(method, fx)


if __name__ == "__main__":
    main()
