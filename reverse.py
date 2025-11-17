import pandas as pd
df = pd.read_csv('pokemon.csv')

#列出不同類別數量
num_classes= df['Type 1'].nunique()
print("屬性1數量:", num_classes)

# 列出所有不同類別
classes = df['Type 1'].unique()
print("屬性1類別名稱:", classes)

# 生成編號映射
classes_mapping = {t: i+1 for i, t in enumerate(classes)}

# 覆蓋原欄位
df['Type 1'] = df['Type 1'].map(classes_mapping)
df['Type 2'] = df['Type 2'].fillna('NULL')
print(df)
df.to_csv("pokemon.csv", index=False)
print("CSV 已覆蓋完成！")

