import pandas as pd
df=pd.read_csv("pokemon.csv")

id_column = "#"  #要重新編號的欄位名稱
start_from = 1   #編號起始值

#依序重新編號
df[id_column] = range(start_from, start_from + len(df))

#覆蓋回原本檔案：
df.to_csv("pokemon.csv", index=False)
print("已重新編號並輸出為 pokemon.csv")

unique_count = df.duplicated().sum()
print(f"共有 {unique_count} 筆資料是重複的")

print(f"資料共有 {len(df)} 列")
