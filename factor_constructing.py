import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import combinations  # 添加缺失的导入

class FactorMiner:
    def __init__(self, data):
        self.df = data.dropna()  # 剔除缺失值
        self.factors = pd.DataFrame()
        
    def _select_low_corr_metrics(self):
        """选择相关性低于0.5的指标组合"""
        metrics = ['pe', 'pb', 'eps', 'gpr', 'npr', 'rev_yoy', 'profit_yoy',
                  'float_share', 'total_assets', 'liquid_assets']
        corr_threshold = 0.5
        
        # 遍历所有三元组合
        for combo in combinations(metrics, 3):
            sub_df = self.df[list(combo)]
            corr_matrix = sub_df.corr().abs()
            if (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1) == 1).max().max() < corr_threshold):
                return combo
        raise ValueError("未找到符合条件的低相关性指标组合")
    
    def build_factors(self):
        """构建三个低相关性因子"""
        try:
            # 选择低相关性指标组合
            metric_names = self._select_low_corr_metrics()
            self.factors = self.df[list(metric_names)].rename(columns={
                metric_names[0]: 'Factor1',
                metric_names[1]: 'Factor2',
                metric_names[2]: 'Factor3'
            })
        except:
            # 备选组合：使用固定的低相关指标
            self.factors = self.df[['pb', 'npr', 'float_share']].rename(columns={
                'pb': 'Factor1',
                'npr': 'Factor2',
                'float_share': 'Factor3'
            })
    
    def calculate_corr_matrix(self):
        """计算相关系数矩阵"""
        return self.factors.corr().round(2)
    
    def calculate_factor_ic(self, target_col='pct_chg'):
        """计算因子IC值（信息系数）"""
        if target_col not in self.df.columns:
            return None
            
        ic_values = {}
        for factor in self.factors.columns:
            ic, _ = pearsonr(self.factors[factor], self.df[target_col])
            ic_values[factor] = ic
        return pd.Series(ic_values).round(3)
    
    def plot_factor_performance(self, save_path=None):
        """绘制因子分布及业绩图表"""
        plt.figure(figsize=(4, 9), dpi=200)
        
        # 子图1
        plt.subplot(3, 1, 1)
        sns.histplot(self.factors['Factor1'], kde=True, color='blue', bins=30)
        plt.title('Factor1 ({}'.format(self.factors.columns[0].upper())+')', fontsize=3)
        plt.xlabel('数值', fontsize=3)
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)
        
        # 子图2
        plt.subplot(3, 1, 2)
        sns.histplot(self.factors['Factor2'], kde=True, color='green', bins=30)
        plt.title('Factor2 ({}'.format(self.factors.columns[1].upper())+')', fontsize=3)
        plt.xlabel('数值', fontsize=3)
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)
        
        # 子图3
        plt.subplot(3, 1, 3)
        sns.histplot(self.factors['Factor3'], kde=True, color='red', bins=30)
        plt.title('Factor3 ({}'.format(self.factors.columns[2].upper())+')', fontsize=3)
        plt.xlabel('数值', fontsize=3)
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# 数据预处理
df = pd.read_csv('output.csv')
df['liquid_assets/total_assets'] = df['liquid_assets'] / df['total_assets']  # 添加流动比率指标

# 实例化并运行
miner = FactorMiner(df)
miner.build_factors()

# 输出相关系数矩阵
print("因子相关系数矩阵：")
print(miner.calculate_corr_matrix())

# 输出因子IC值（与涨跌幅的相关性）
ic_values = miner.calculate_factor_ic()
if ic_values is not None:
    print("\n因子IC值（与次日涨跌幅相关性）：")
    print(ic_values)

# 绘制因子图表
miner.plot_factor_performance('factor_performance.png')