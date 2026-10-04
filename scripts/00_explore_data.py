from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame

print("Shape:", df.shape)
print("Class names:", data.target_names)
print("Number of classes:", df["target"].nunique())
print("Class proportions:")
print(df["target"].value_counts(normalize=True))