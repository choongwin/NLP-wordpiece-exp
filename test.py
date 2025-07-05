import pyarrow.parquet as pq

# 读取为 PyArrow Table
table = pq.read_table(r"C:\Users\choon\OneDrive\Desktop\fnlp25_hw3_2200094605\HoC\test.parquet")

# 转换为 Pandas DataFrame（可选）
df = table.to_pandas()

print(df)