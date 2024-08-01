from sklearn.model_selection import train_test_split
import pyarrow.feather as feather

"""Train-Test Split (done once)"""
df = feather.read_feather('../data/raw/home_assignment.feather')
print(df.shape)
df = df.drop_duplicates()
df = df.dropna(subset=['TLJYWBE'])

df, df_test, target, target_test = train_test_split(df, df['TLJYWBE'], test_size=0.2,
                                                    random_state=42)

df.to_feather('../data/raw/train_data.csv')
df_test.to_feather('../data/raw/test_data.csv')
print(df.shape)
print(df_test.shape)
print("Done")