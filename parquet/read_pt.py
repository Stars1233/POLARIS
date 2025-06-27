import pdb

import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet("qwen3-4b-stage2.parquet")

# # 筛选出满足 difficulty > 0.9 的所有样本
# df_candidates = df[df["difficulty"] > 0.9]

# # 判断满足条件的样本是否至少有54个
# if len(df_candidates) >= 54:
#     # 可选择两种方式：
    
#     # 方式1：按顺序取前54个样本
#     indices_to_remove = df_candidates.head(54).index

#     # 方式2：随机抽取54个样本（取消注释下面代码并注释掉方式1）
#     # indices_to_remove = df_candidates.sample(n=54, random_state=42).index

#     # 删除这54个样本
#     df_filtered = df.drop(indices_to_remove)
# else:
#     print(f"满足条件的样本不足54个，只有 {len(df_candidates)} 个，将全部删除。")
#     df_filtered = df.drop(df_candidates.index)

df_filtered = df.drop(columns="difficulty")

# 将过滤后的数据保存到新的 Parquet 文件
# pdb.set_trace()
df_filtered.to_parquet("qwen3-4b-s2.parquet")
print("处理完成，新的数据文件已保存为 filtered_data.parquet")

