import numpy as np
import math



def bit_reverse(k, log2n):
    """计算 k 在 log2n 位下的位反转"""
    rev = 0
    n = k
    for i in range(log2n):
        if n & 1:
            rev |= (1 << (log2n - 1 - i))
        n >>= 1
    return rev

def CFFT(AR_in, AI_in, N, ISIGN):
    """
    参数:
        AR_in, AI_in: 输入实部和虚部数组（list 或 np.array），长度 N=2^M
        N: 数组长度，必须为 2 的幂
        ISIGN: 1 表示正变换 (FFT)， -1 表示逆变换 (IFFT)
    
    返回:
        AR_out, AI_out: 变换后的实部和虚部新数组（不修改原输入）
    """
    assert N > 0 and (N & (N - 1)) == 0, "N 必须是 2 的整数次幂"
    log2n = int(math.log2(N))
    
    # 复制输入，避免修改原数据
    AR = np.array(AR_in, dtype=float).copy()
    AI = np.array(AI_in, dtype=float).copy()
    
    # 第一步：bit-reversal 重排
    for i in range(N):
        j = bit_reverse(i, log2n)
        if j > i:
            AR[i], AR[j] = AR[j], AR[i]
            AI[i], AI[j] = AI[j], AI[i]
    
    # 第二步：蝶形运算（使用递推旋转因子，提高精度）
    for s in range(1, log2n + 1):
        m = 1 << s          # m = 2^s
        mh = m // 2         # m/2
        
        # 本阶段基本旋转因子 wm = exp(-2πi * ISIGN / m)
        theta_m = -2.0 * np.pi * ISIGN / m
        wm_r = np.cos(theta_m)
        wm_i = np.sin(theta_m)
        
        for k in range(0, N, m):
            # 每个块从旋转因子 1+0j 开始
            wr = 1.0
            wi = 0.0
            for jj in range(mh):
                i1 = k + jj
                i2 = i1 + mh
                
                # 旋转下半部分
                tr = wr * AR[i2] - wi * AI[i2]
                ti = wr * AI[i2] + wi * AR[i2]
                
                # 蝶形更新
                AR[i2] = AR[i1] - tr
                AI[i2] = AI[i1] - ti
                AR[i1] = AR[i1] + tr
                AI[i1] = AI[i1] + ti
                
                # 递推更新旋转因子：(wr + i wi) *= wm
                temp_r = wr * wm_r - wi * wm_i
                temp_i = wr * wm_i + wi * wm_r
                wr = temp_r
                wi = temp_i
    
    # 逆变换需要归一化 1/N（正变换不归一化）
    if ISIGN == -1:
        AR /= N
        AI /= N
    
    return AR, AI

def compute_bessel_Jn(n, z, N):
    """
    使用 FFT 方法计算第一类贝塞尔函数 Jn(z)
    n: 阶数 (n = 0, 1, ..., N-1)
    z: 参数
    N: 采样点数，必须为 2 的幂
    """
    AR = np.zeros(N)
    AI = np.zeros(N)
    
    for j in range(N):
        theta_j = 2.0 * np.pi * j / N
        inner = z * np.cos(theta_j)
        AR[j] = np.cos(inner)      # 对应公式 (2)
        AI[j] = np.sin(inner)      # 对应公式 (3)
    
    # 执行逆 FFT (ISIGN = -1)
    AR_out, AI_out = CFFT(AR, AI, N, 1)
    
    # 提取 Jn(z)：注意因子 sqrt(N) 来自 Δθ = 2π/N 和 1/N 归一化
    # 详细推导见作业提示公式 (1)
    # Jn = np.sqrt(N) * AR_out[n]
    z0 = complex (AR_out[n] , -1 * AI_out[n])
    III = complex(0,1) 
    z = z0 * (III**n) / N
    Jn = z.real
    return Jn





def find_min_M_greater_than_n0_plus_1(z: float) -> int:
    """
    对浮点数 z 向上取整得到 n_0 = ceil(z)
    然后返回严格大于 (n_0 + 1) 的最小 2 的幂 M = 2^n
    """
    if z <= 0:
        raise ValueError("z 应为正数，以确保结果有意义")
    
    n_0 = math.ceil(z)          # 向上取整得到 n_0
    target = n_0 + 1             # 需要严格大于的目标值
    
    # 计算严格大于 target 的最小 2 的幂
    # target.bit_length() 给出 target 的二进制位数
    # 1 << length 就是 2^length，正好是大于等于 2^(bit_length-1) * 2 的值
    # 当 target 是 2 的幂时，它会自动进位到下一个
    M = 1 << target.bit_length()
    
    return M

# ====================== 主程序与测试 ======================
if __name__ == "__main__":
    print("=== FFT 功能验证（单位冲激测试）===\n")
    N_test = 16
    AR_test = [1.0 if i == 0 else 0.0 for i in range(N_test)]
    AI_test = [0.0] * N_test
    
    AR_fft, AI_fft = CFFT(AR_test, AI_test, N_test, 1)
    print("正变换后实部", [round(x, 6) for x in AR_fft])
    print("正变换后虚部", [round(x, 6) for x in AI_fft])
    
    # 逆变换恢复
    AR_recover, AI_recover = CFFT(AR_fft, AI_fft, N_test, -1)
    print("逆变换恢复实部（应接近原冲激）：", [round(x, 6) for x in AR_recover])
    
    print("\n=== 使用 FFT 计算贝塞尔函数 J_n(z) ===\n")

    while True:
        try:
            user_input = input("请输入一个非负浮点数: ")
            z = float(user_input)
        
            if z < 0:
                print("错误：输入的数不能为负数，请重新输入！")
                continue
        
        # 输入合法，跳出循环
            print(f"您输入的非负浮点数是: {z}")
            break
        
        except ValueError:
            print("错误：输入无效，请输入一个有效的数字！")
    
    while True:
        try:
            user_input = input("请输入一个非负整数: ")
            num = int(user_input)
        
            if num < 0:
                print("错误：输入的数不能为负数，请重新输入！")
                continue
        
            # 输入合法，跳出循环
            print(f"您输入的非负整数是: {num}")
            break
        
        except ValueError:
            print("错误：输入无效，请输入一个有效的整数！")


    # 可选：使用 scipy 精确值进行对比（提交作业时可注释掉）
    from scipy.special import jv
    
    #for z in zs:
    print(f"\nz = {z}")
    # 根据 z 大小选择合适的 N（N 越大精度越高）

    M = find_min_M_greater_than_n0_plus_1((z+num))

    Ns = [M << 6, M << 8 , M << 10]
        
    for N in Ns:
        print(f" Terms: N = {N:5d} : ", end="")
        print("\n")
        
        Jn_approx = compute_bessel_Jn(num, z, N)
        print(f"J_{num}({z:f}) ≈ {Jn_approx: .10e}", end="   ")
        print("\n")
                
        exact = jv(num, z)
        err = abs(Jn_approx - exact)
        print(f"误差(参考scipy库的标准值 {err:.3e})", end="   ")
        print("\n")
    print("\n")