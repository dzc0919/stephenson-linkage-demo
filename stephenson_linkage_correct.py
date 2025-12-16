import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import tkinter.font as tkfont
import math
import numpy as np

class StephensonIIILinkage:
    def __init__(self, root):
        self.root = root
        self.root.title("Stephenson III 六杆机构 (nv=2 标准模型)")

        # 统一色板
        self.colors = {
            'bg': '#f3f6fb',
            'panel': '#ffffff',
            'border': '#d7dce3',
            'text': '#1f2933',
            'muted': '#6b7280',
            'accent': '#1f7a8c',
            'accent2': '#f59e0b'
        }

        # 统一字体设置，优先使用现代无衬线/等宽字体
        sans_family = self._choose_family(['Segoe UI', 'Helvetica', 'Arial'])
        mono_family = self._choose_family(['JetBrains Mono', 'Cascadia Mono', 'Consolas', 'Courier New'])
        self.fonts = {
            'title': tkfont.Font(family=sans_family, size=16, weight='bold'),
            'subtitle': tkfont.Font(family=sans_family, size=12, weight='bold'),
            'body': tkfont.Font(family=sans_family, size=11),
            'body_bold': tkfont.Font(family=sans_family, size=11, weight='bold'),
            'caption': tkfont.Font(family=sans_family, size=10),
            'mono': tkfont.Font(family=mono_family, size=10),
            'mono_bold': tkfont.Font(family=mono_family, size=10, weight='bold'),
        }

        # 设置 ttk 默认字体风格
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        self.root.configure(bg=self.colors['bg'])
        style.configure("TLabel", font=self.fonts['body'], background=self.colors['bg'], foreground=self.colors['text'])
        style.configure("Muted.TLabel", font=self.fonts['caption'], background=self.colors['bg'], foreground=self.colors['muted'])
        style.configure("Panel.TLabel", font=self.fonts['body'], background=self.colors['panel'], foreground=self.colors['text'])
        style.configure("PanelMuted.TLabel", font=self.fonts['caption'], background=self.colors['panel'], foreground=self.colors['muted'])
        style.configure("TButton", font=self.fonts['body'], padding=6,
                        background=self.colors['panel'], borderwidth=1, relief='flat')
        style.map("TButton",
                  background=[("active", self.colors['accent'])],
                  foreground=[("active", "white")])
        style.configure("TLabelframe", background=self.colors['panel'], bordercolor=self.colors['border'])
        style.configure("TLabelframe.Label", font=self.fonts['body_bold'], background=self.colors['panel'], foreground=self.colors['text'])
        style.configure("TFrame", background=self.colors['bg'])
        style.configure("Main.TFrame", background=self.colors['bg'])
        style.configure("Panel.TFrame", background=self.colors['panel'])
        style.configure("Panel.TLabelframe", background=self.colors['panel'])
        style.configure("TScale", font=self.fonts['body'])

        # --- 1. 定义机构尺寸 (Design Parameters: x) ---
        # 这些就是你在论文 Matrix A 中需要求偏导的 x 变量
        self.params = {
            # Loop 1: 四杆机构基础 (A-B-C-D)
            'L_AB': 100,  # x1: 曲柄 (Crank)
            'L_BC': 220,  # x2: 耦合器边1 (Coupler Side 1)
            'L_CD': 200,  # x3: 摇杆 (Rocker)
            'L_AD': 300,  # x4: 机架水平距离 (Ground 1)

            # Coupler Triangle: 刚性三角形 BCE
            # 注意: E点的位置由三角形的边长确定
            'L_BE': 140,  # x5: 耦合器边2 (Coupler Side 2)
            'L_CE': 120,  # x6: 耦合器边3 (Coupler Side 3) - 用于确定E的刚性位置

            # Loop 2: 附加二杆组 (E-F-G)
            'L_EF': 240,  # x7: 连杆 (Link 5)
            'L_FG': 220,  # x8: 摇杆 (Link 6)

            # Ground Points (机架坐标)
            'Ax': 200, 'Ay': 400,  # A点 (原点)
            'Dx': 500, 'Dy': 400,  # D点 (A点右侧300处, 对应 L_AD)
            'Gx': 650, 'Gy': 400   # G点 (第二回路的接地点)
        }

        # --- 初始化绘图界面 ---
        self.setup_ui()

        # --- 初始状态 ---
        self.theta2 = math.radians(60)  # 输入角 (Input Angle)
        self.animating = False
        self.draw_mechanism()

    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10", style="Main.TFrame")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 画布
        self.canvas = tk.Canvas(main_frame, width=1000, height=600, bg='#f9fafb',
                              highlightthickness=1, highlightbackground=self.colors['border'])
        self.canvas.grid(row=0, column=0, columnspan=2, pady=10)

        # 控件区域
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10", style="Panel.TLabelframe")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # 按钮
        ttk.Button(control_frame, text="自动旋转 (Auto)", command=self.toggle_animation).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="重置 (Reset)", command=self.reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="计算矩阵 (Jacobian)", command=self.show_matrices_window).pack(side=tk.LEFT, padx=5)

        # 角度滑块
        ttk.Label(control_frame, text="输入角 \u03B82:", style="Panel.TLabel").pack(side=tk.LEFT, padx=10)
        self.slider = ttk.Scale(control_frame, from_=0, to=360, command=self.on_slider_change)
        self.slider.set(60)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 信息标签
        self.info_label = ttk.Label(control_frame, text="状态: 就绪", style="Panel.TLabel", foreground="blue")
        self.info_label.pack(side=tk.RIGHT, padx=10)

    def calculate_circle_intersection(self, p1, r1, p2, r2, side_selector=+1):
        """
        计算两圆交点 (核心几何算法)
        p1, p2: 圆心 {'x':, 'y':}
        r1, r2: 半径
        side_selector: +1 或 -1，用于选择两个交点中的哪一个
        """
        d2 = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
        d = math.sqrt(d2)

        if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
            return None # 无法构成三角形/无交点

        a = (r1**2 - r2**2 + d2) / (2 * d)
        h = math.sqrt(max(0, r1**2 - a**2))

        x2 = p1['x'] + a * (p2['x'] - p1['x']) / d
        y2 = p1['y'] + a * (p2['y'] - p1['y']) / d

        x3 = x2 + side_selector * h * (p2['y'] - p1['y']) / d
        y3 = y2 - side_selector * h * (p2['x'] - p1['x']) / d

        return {'x': x3, 'y': y3}

    def solve_kinematics(self, theta_input):
        """
        运动学求解 (Kinematic Solver)
        这里对应论文中的 vector loop equations
        """
        p = self.params

        # 1. 定义固定点 (Ground Points)
        A = {'x': p['Ax'], 'y': p['Ay']}
        D = {'x': p['Dx'], 'y': p['Dy']}
        G = {'x': p['Gx'], 'y': p['Gy']}

        # 2. 计算 Loop 1: A-B-C-D
        # 计算 B 点 (由输入角 theta2 驱动)
        # Bx = Ax + L_AB * cos(theta2)
        B = {
            'x': A['x'] + p['L_AB'] * math.cos(theta_input),
            'y': A['y'] - p['L_AB'] * math.sin(theta_input) # Canvas Y轴向下，故减
        }

        # 计算 C 点 (B点和D点的圆交点)
        # 这是一个四杆机构的闭环约束
        C = self.calculate_circle_intersection(B, p['L_BC'], D, p['L_CD'], side_selector=-1)

        if not C: return None # 机构卡死或不可达

        # 3. 计算 E 点 (刚性三角形 BCE)
        # E 是基于 B 和 C 的刚性点，由 L_BE 和 L_CE 确定
        # 这对应 Matrix A 中的耦合器刚性约束
        E = self.calculate_circle_intersection(B, p['L_BE'], C, p['L_CE'], side_selector=+1)

        if not E: return None

        # 4. 计算 Loop 2: G-F-E (或 E-F-G)
        # 我们已知 E 和 G，需要找中间点 F
        # F 是 E点(半径L_EF) 和 G点(半径L_FG) 的交点
        F = self.calculate_circle_intersection(E, p['L_EF'], G, p['L_FG'], side_selector=+1)

        if not F: return None

        return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G}

    def draw_mechanism(self):
        self.canvas.delete("all")

        # 获取当前位置
        pos = self.solve_kinematics(self.theta2)

        if pos is None:
            if hasattr(self, 'info_label'):
                self.info_label.config(text="错误: 机构无法到达此位置 (几何约束限制)", foreground="red")
            return

        if hasattr(self, 'info_label'):
            self.info_label.config(text=f"角度: {math.degrees(self.theta2):.1f}° | 状态: 正常", foreground="blue")

        # --- 绘图 (Drawing) ---

        # 1. 绘制机架 (Ground) - 灰色虚线
        self.canvas.create_line(pos['A']['x'], pos['A']['y'], pos['G']['x'], pos['G']['y'], 
                              fill='gray', dash=(4, 4), width=2)
        self.draw_joint(pos['A'], 'A (Fixed)', 'black')
        self.draw_joint(pos['D'], 'D (Fixed)', 'black')
        self.draw_joint(pos['G'], 'G (Fixed)', 'black')

        # 2. 绘制 Loop 1 (四杆部分)
        # 曲柄 AB
        self.canvas.create_line(pos['A']['x'], pos['A']['y'], pos['B']['x'], pos['B']['y'], 
                              fill='blue', width=5, tags="link")
        self.draw_text_mid(pos['A'], pos['B'], "x1 (AB)")

        # 摇杆 CD
        self.canvas.create_line(pos['C']['x'], pos['C']['y'], pos['D']['x'], pos['D']['y'], 
                              fill='purple', width=5, tags="link")
        self.draw_text_mid(pos['C'], pos['D'], "x3 (CD)")

        # 3. 绘制 耦合器三角形 BCE (Coupler)
        # 这是一个刚性体，填充绿色表示
        self.canvas.create_polygon(pos['B']['x'], pos['B']['y'],
                                 pos['C']['x'], pos['C']['y'],
                                 pos['E']['x'], pos['E']['y'],
                                 fill='lightgreen', outline='green', width=2)
        # 绘制三角形边框
        self.canvas.create_line(pos['B']['x'], pos['B']['y'], pos['C']['x'], pos['C']['y'], fill='green', width=3)
        self.canvas.create_line(pos['B']['x'], pos['B']['y'], pos['E']['x'], pos['E']['y'], fill='green', width=3)
        self.canvas.create_line(pos['C']['x'], pos['C']['y'], pos['E']['x'], pos['E']['y'], fill='green', width=3)

        # 标注耦合器尺寸
        self.draw_text_mid(pos['B'], pos['C'], "x2")
        self.draw_text_mid(pos['B'], pos['E'], "x5")

        # 4. 绘制 Loop 2 (附加二杆组)
        # 连杆 EF
        self.canvas.create_line(pos['E']['x'], pos['E']['y'], pos['F']['x'], pos['F']['y'], 
                              fill='orange', width=5, tags="link")
        self.draw_text_mid(pos['E'], pos['F'], "x7 (EF)")

        # 摇杆 FG
        self.canvas.create_line(pos['F']['x'], pos['F']['y'], pos['G']['x'], pos['G']['y'], 
                              fill='brown', width=5, tags="link")
        self.draw_text_mid(pos['F'], pos['G'], "x8 (FG)")

        # 5. 绘制所有关节 (Joints)
        for name in ['B', 'C', 'E', 'F']:
            self.draw_joint(pos[name], name, 'red')

        # 6. 辅助标注 (针对作业)
        self.canvas.create_text(50, 30, text="Stephenson III Linkage",
                               font=self.fonts['title'], anchor='w', fill=self.colors['text'])
        self.canvas.create_text(50, 60, text="Loop 1: A-B-C-D (四杆)",
                               font=self.fonts['caption'], anchor='w', fill='blue')
        self.canvas.create_text(50, 80, text="Loop 2: G-F-E (二杆组)",
                               font=self.fonts['caption'], anchor='w', fill='brown')
        self.canvas.create_text(50, 100, text="nv = 2 (双回路)",
                               font=self.fonts['caption'], anchor='w', fill=self.colors['muted'])

    def draw_joint(self, p, label, color):
        r = 6
        self.canvas.create_oval(p['x']-r, p['y']-r, p['x']+r, p['y']+r, fill='white', outline=color, width=2)
        self.canvas.create_text(p['x']+10, p['y']-10, text=label, font=self.fonts['body_bold'], fill='#333')

    def draw_text_mid(self, p1, p2, text):
        mx = (p1['x'] + p2['x']) / 2
        my = (p1['y'] + p2['y']) / 2
        self.canvas.create_text(mx, my, text=text, font=self.fonts['caption'], fill='#666')

    # --- 交互与动画逻辑 ---
    def on_slider_change(self, val):
        self.theta2 = math.radians(float(val))
        self.draw_mechanism()

    def toggle_animation(self):
        self.animating = not self.animating
        if self.animating:
            self.animate()

    def animate(self):
        if not self.animating: return

        current_deg = math.degrees(self.theta2)
        new_deg = (current_deg + 2) % 360
        self.theta2 = math.radians(new_deg)
        self.slider.set(new_deg)

        self.draw_mechanism()
        self.root.after(50, self.animate)

    def reset(self):
        self.animating = False
        self.slider.set(60)
        self.on_slider_change(60)

    # ===== 矩阵计算功能 =====

    def solve_angles(self, theta_input):
        """
        计算所有关键角度 (转换为数学坐标系)

        返回:
            dict: {
                'theta2': θ₂ (输入角),
                'theta3': θ₃ (角BC),
                'theta4': θ₄ (角CD),
                'theta5': θ₅ (耦合器角 = θ₃ + α),
                'theta6': θ₆ (角EF),
                'theta7': θ₇ (角FG),
                'alpha': α (耦合器三角形内角)
            }
            或 None (机构不可达)
        """
        # 获取点位置
        pos = self.solve_kinematics(theta_input)
        if pos is None:
            return None

        p = self.params

        # 坐标系转换：Canvas (Y向下) → Math (Y向上)
        # 使用 atan2(-dy, dx) 来转换

        # θ₃: B到C的角度
        dx_BC = pos['C']['x'] - pos['B']['x']
        dy_BC = pos['C']['y'] - pos['B']['y']
        theta3 = math.atan2(-dy_BC, dx_BC)  # Canvas → Math

        # θ₄: D到C的角度
        dx_DC = pos['C']['x'] - pos['D']['x']
        dy_DC = pos['C']['y'] - pos['D']['y']
        theta4 = math.atan2(-dy_DC, dx_DC)

        # α: 耦合器三角形内角 (使用余弦定理，更稳定)
        L_BC = p['L_BC']
        L_BE = p['L_BE']
        L_CE = p['L_CE']
        cos_alpha = (L_BC**2 + L_BE**2 - L_CE**2) / (2 * L_BC * L_BE)
        # 限制范围避免数值误差
        cos_alpha = max(-1, min(1, cos_alpha))
        alpha = math.acos(cos_alpha)

        # θ₅: E点相对于绝对坐标系的角度
        # E在BCE三角形中，θ₅ = θ₃ + α
        theta5 = theta3 + alpha

        # θ₆: E到F的角度
        dx_EF = pos['F']['x'] - pos['E']['x']
        dy_EF = pos['F']['y'] - pos['E']['y']
        theta6 = math.atan2(-dy_EF, dx_EF)

        # θ₇: G到F的角度
        dx_GF = pos['F']['x'] - pos['G']['x']
        dy_GF = pos['F']['y'] - pos['G']['y']
        theta7 = math.atan2(-dy_GF, dx_GF)

        return {
            'theta2': theta_input,
            'theta3': theta3,
            'theta4': theta4,
            'theta5': theta5,
            'theta6': theta6,
            'theta7': theta7,
            'alpha': alpha
        }

    def calculate_matrix_A(self, theta_input):
        """
        计算 Matrix A: ∂Φ/∂x (4×8)

        约束方程对设计参数的偏导数
        x = [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈]
          = [L_AB, L_BC, L_CD, L_AD, L_BE, L_CE, L_EF, L_FG]

        返回:
            np.ndarray (4×8) 或 None
        """
        angles = self.solve_angles(theta_input)
        if angles is None:
            return None

        p = self.params

        # 提取角度
        theta2 = angles['theta2']
        theta3 = angles['theta3']
        theta4 = angles['theta4']
        theta5 = angles['theta5']
        theta6 = angles['theta6']
        theta7 = angles['theta7']

        # 初始化 Matrix A (4×8)
        A = np.zeros((4, 8))

        # Loop 1 - X方向 (Φ₁): x₁cos(θ₂) + x₂cos(θ₃) - x₃cos(θ₄) - x₄ = 0
        A[0, 0] = math.cos(theta2)   # ∂Φ₁/∂x₁
        A[0, 1] = math.cos(theta3)   # ∂Φ₁/∂x₂
        A[0, 2] = -math.cos(theta4)  # ∂Φ₁/∂x₃
        A[0, 3] = -1                 # ∂Φ₁/∂x₄

        # Loop 1 - Y方向 (Φ₂): x₁sin(θ₂) + x₂sin(θ₃) - x₃sin(θ₄) = 0
        A[1, 0] = math.sin(theta2)
        A[1, 1] = math.sin(theta3)
        A[1, 2] = -math.sin(theta4)
        # A[1, 3] = 0 (已是零)

        # Loop 2 - X方向 (Φ₃): x₅cos(θ₅) + x₇cos(θ₆) - x₈cos(θ₇) - ΔxGE = 0
        # 注意: x₂和x₆会影响α，进而影响θ₅，这里简化处理
        A[2, 4] = math.cos(theta5)   # ∂Φ₃/∂x₅
        A[2, 6] = math.cos(theta6)   # ∂Φ₃/∂x₇
        A[2, 7] = -math.cos(theta7)  # ∂Φ₃/∂x₈

        # Loop 2 - Y方向 (Φ₄): x₅sin(θ₅) + x₇sin(θ₆) - x₈sin(θ₇) - ΔyGE = 0
        A[3, 4] = math.sin(theta5)
        A[3, 6] = math.sin(theta6)
        A[3, 7] = -math.sin(theta7)

        return A

    def calculate_matrix_B(self, theta_input):
        """
        计算 Matrix B: ∂Φ/∂u (4×4)

        约束方程对角度变量的偏导数
        u = [θ₃, θ₄, θ₆, θ₇] (角度变量)

        理论依据:
        - Loop 1 (Φ₁, Φ₂): 仅依赖 θ₃, θ₄
        - Loop 2 (Φ₃, Φ₄): 依赖 θ₃(通过θ₅=θ₃+α), θ₆, θ₇
        - 关键: θ₅ = θ₃ + α，所以 ∂θ₅/∂θ₃ = 1

        返回:
            np.ndarray (4×4) 或 None
        """
        angles = self.solve_angles(theta_input)
        if angles is None:
            return None

        p = self.params

        # 提取角度
        theta3 = angles['theta3']
        theta4 = angles['theta4']
        theta5 = angles['theta5']  # = θ₃ + α
        theta6 = angles['theta6']
        theta7 = angles['theta7']

        # 提取杆长
        x2 = p['L_BC']  # 耦合器边1
        x3 = p['L_CD']  # 摇杆
        x5 = p['L_BE']  # 耦合器边2
        x7 = p['L_EF']  # 连杆
        x8 = p['L_FG']  # 摇杆

        # 初始化 Matrix B (4×4)
        B = np.zeros((4, 4))

        # ===== Loop 1 约束 (仅依赖 θ₃, θ₄) =====

        # Φ₁ (Loop 1 X): x₁cos(θ₂) + x₂cos(θ₃) - x₃cos(θ₄) - x₄ = 0
        B[0, 0] = -x2 * math.sin(theta3)  # ∂Φ₁/∂θ₃
        B[0, 1] = x3 * math.sin(theta4)   # ∂Φ₁/∂θ₄
        B[0, 2] = 0                       # ∂Φ₁/∂θ₆
        B[0, 3] = 0                       # ∂Φ₁/∂θ₇

        # Φ₂ (Loop 1 Y): x₁sin(θ₂) + x₂sin(θ₃) - x₃sin(θ₄) = 0
        B[1, 0] = x2 * math.cos(theta3)   # ∂Φ₂/∂θ₃
        B[1, 1] = -x3 * math.cos(theta4)  # ∂Φ₂/∂θ₄
        B[1, 2] = 0
        B[1, 3] = 0

        # ===== Loop 2 约束 (依赖 θ₃, θ₆, θ₇) =====
        # 注意: θ₅ = θ₃ + α, 所以 ∂θ₅/∂θ₃ = 1

        # Φ₃ (Loop 2 X): x₅cos(θ₅) + x₇cos(θ₆) - x₈cos(θ₇) - ΔxGE = 0
        B[2, 0] = -x5 * math.sin(theta5)  # ∂Φ₃/∂θ₃ (通过θ₅)
        B[2, 1] = 0                       # ∂Φ₃/∂θ₄ (Loop2不依赖θ₄)
        B[2, 2] = -x7 * math.sin(theta6)  # ∂Φ₃/∂θ₆
        B[2, 3] = x8 * math.sin(theta7)   # ∂Φ₃/∂θ₇

        # Φ₄ (Loop 2 Y): x₅sin(θ₅) + x₇sin(θ₆) - x₈sin(θ₇) - ΔyGE = 0
        B[3, 0] = x5 * math.cos(theta5)   # ∂Φ₄/∂θ₃ (通过θ₅)
        B[3, 1] = 0                       # ∂Φ₄/∂θ₄
        B[3, 2] = x7 * math.cos(theta6)   # ∂Φ₄/∂θ₆
        B[3, 3] = -x8 * math.cos(theta7)  # ∂Φ₄/∂θ₇

        return B

    def _create_matrix_table(self, parent, matrix, row_labels, col_labels):
        """
        创建矩阵表格显示（辅助方法）

        参数:
            parent: 父容器
            matrix: numpy数组
            row_labels: 行标签列表
            col_labels: 列标签列表
        """
        table_frame = ttk.Frame(parent, style="Panel.TFrame")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 表头 - 左上角空白
        ttk.Label(table_frame, text="", width=12, style="Panel.TLabel").grid(row=0, column=0, padx=2, pady=2)

        # 列标签
        for j, col_label in enumerate(col_labels):
            label = ttk.Label(table_frame, text=col_label,
                             font=self.fonts['mono_bold'],
                             width=10, anchor='center',
                             style="Panel.TLabel")
            label.grid(row=0, column=j+1, padx=2, pady=2)

        # 数据行
        for i in range(matrix.shape[0]):
            # 行标签
            row_label = ttk.Label(table_frame, text=row_labels[i],
                                 font=self.fonts['mono_bold'],
                                 width=12, anchor='w',
                                 style="Panel.TLabel")
            row_label.grid(row=i+1, column=0, padx=2, pady=2)

            # 数据单元格
            for j in range(matrix.shape[1]):
                value = matrix[i, j]

                # 格式化数值
                if abs(value) < 1e-10:
                    text = "0.000"
                    fg = 'gray'
                elif abs(value) < 0.01 or abs(value) > 1000:
                    text = f"{value:.2e}"  # 科学记数法
                    fg = 'blue' if value > 0 else 'red'
                else:
                    text = f"{value:.3f}"
                    fg = 'blue' if value > 0 else 'red'

                # 创建单元格标签
                cell_label = tk.Label(table_frame, text=text,
                                     foreground=fg,
                                     background='#f8fafc',
                                     font=self.fonts['mono'],
                                     width=10, anchor='center',
                                     relief='solid', borderwidth=1)
                cell_label.grid(row=i+1, column=j+1, padx=1, pady=1)

    def show_matrices_window(self):
        """
        显示矩阵计算结果的弹窗
        """
        # 计算矩阵
        matrix_A = self.calculate_matrix_A(self.theta2)
        matrix_B = self.calculate_matrix_B(self.theta2)
        angles = self.solve_angles(self.theta2)

        # 错误处理
        if matrix_A is None or matrix_B is None or angles is None:
            messagebox.showerror("错误",
                               "无法计算矩阵：机构在当前位置不可达\n"
                               "请调整输入角度后重试")
            return

        # 创建弹窗
        window = Toplevel(self.root)
        window.title(f"雅可比矩阵 - 输入角: {math.degrees(self.theta2):.1f}°")
        window.geometry("900x700")
        window.resizable(True, True)
        window.configure(bg=self.colors['bg'])

        # 主容器
        main_container = ttk.Frame(window, padding="10", style="Main.TFrame")
        main_container.pack(fill=tk.BOTH, expand=True)

        # 标题信息
        info_frame = ttk.Frame(main_container, style="Main.TFrame")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(info_frame,
                               text="Stephenson III 机构 - 雅可比矩阵分析",
                               font=self.fonts['title'])
        title_label.pack()

        # 角度信息
        angle_info = (f"当前配置: θ₂={math.degrees(angles['theta2']):.1f}°, "
                     f"θ₃={math.degrees(angles['theta3']):.1f}°, "
                     f"θ₄={math.degrees(angles['theta4']):.1f}°, "
                     f"θ₅={math.degrees(angles['theta5']):.1f}°, "
                     f"θ₆={math.degrees(angles['theta6']):.1f}°, "
                     f"θ₇={math.degrees(angles['theta7']):.1f}°")

        angle_label = ttk.Label(info_frame, text=angle_info,
                               font=self.fonts['mono'], foreground='blue')
        angle_label.pack(pady=5)

        # 创建滚动容器
        canvas_scroll = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical",
                                 command=canvas_scroll.yview)
        scrollable_frame = ttk.Frame(canvas_scroll)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )

        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        # Matrix A
        frame_A = ttk.LabelFrame(scrollable_frame,
                                text="Matrix A: ∂Φ/∂x (约束方程对设计参数的偏导)",
                                padding="10",
                                style="Panel.TLabelframe")
        frame_A.pack(fill=tk.BOTH, expand=True, pady=5)

        desc_A = ttk.Label(frame_A,
                          text="设计参数: x = [x₁(AB), x₂(BC), x₃(CD), x₄(AD), x₅(BE), x₆(CE), x₇(EF), x₈(FG)]",
                          font=self.fonts['mono'], foreground='gray', style="Panel.TLabel")
        desc_A.pack(anchor='w', pady=(0, 5))

        row_labels_A = ['Φ₁ (Loop1-X)', 'Φ₂ (Loop1-Y)',
                       'Φ₃ (Loop2-X)', 'Φ₄ (Loop2-Y)']
        col_labels_A = ['x₁(AB)', 'x₂(BC)', 'x₃(CD)', 'x₄(AD)',
                       'x₅(BE)', 'x₆(CE)', 'x₇(EF)', 'x₈(FG)']

        self._create_matrix_table(frame_A, matrix_A, row_labels_A, col_labels_A)

        # Matrix B
        frame_B = ttk.LabelFrame(scrollable_frame,
                                text="Matrix B: ∂Φ/∂u (约束方程对角度变量的偏导)",
                                padding="10",
                                style="Panel.TLabelframe")
        frame_B.pack(fill=tk.BOTH, expand=True, pady=5)

        desc_B = ttk.Label(frame_B,
                          text="角度变量: u = [θ₃, θ₄, θ₆, θ₇]  (注意: θ₅=θ₃+α 为刚性耦合)",
                          font=self.fonts['mono'], foreground='gray', style="Panel.TLabel")
        desc_B.pack(anchor='w', pady=(0, 5))

        row_labels_B = ['Φ₁ (Loop1-X)', 'Φ₂ (Loop1-Y)',
                       'Φ₃ (Loop2-X)', 'Φ₄ (Loop2-Y)']
        col_labels_B = ['θ₃', 'θ₄', 'θ₆', 'θ₇']

        self._create_matrix_table(frame_B, matrix_B, row_labels_B, col_labels_B)

        # 布局滚动组件
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 关闭按钮
        close_btn = ttk.Button(main_container, text="关闭",
                              command=window.destroy)
        close_btn.pack(pady=10)

        # 检查奇异性
        det_B = np.linalg.det(matrix_B)
        if abs(det_B) < 1e-6:
            warning_label = ttk.Label(info_frame,
                                     text=f"⚠️ 警告: 机构接近奇异位置 (det(B) = {det_B:.2e})",
                                     font=self.fonts['body_bold'],
                                     foreground='red')
            warning_label.pack(pady=5)

    def _choose_family(self, candidates):
        """Pick the first available font family from the candidate list."""
        if not hasattr(self, '_font_families'):
            self._font_families = set(tkfont.families())
        for name in candidates:
            if name in self._font_families:
                return name
        return 'TkDefaultFont'

if __name__ == "__main__":
    root = tk.Tk()
    app = StephensonIIILinkage(root)
    root.mainloop()
