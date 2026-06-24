import pandas as pd

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\intra_oral_ml\data\masks\combined_with_masks.csv")
print(f"Shape: {df.shape}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")
print(f"\nmask_path null: {df['mask_path'].isna().sum()} / {len(df)}")
print(f"\nSample mask_path:\n{df['mask_path'].head(3).tolist()}")