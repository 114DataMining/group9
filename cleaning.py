import pandas as pd
df = pd.read_csv("pokemon.csv")
miscount =  df.isnull().sum()
print(f"各欄位缺失值數量：")
print(f"{miscount}")

#使用字串 'None' 來填補 Type 2 欄位中的所有缺失值 (NaN)
df['Type 2'] = df['Type 2'].fillna('NULL')
df.to_csv("pokemon.csv", index=False)

#再次檢查 Type 2 欄位是否還有缺失值
print("Type 2欄位處理後的缺失值數量：")
print(df['Type 2'].isnull().sum())

