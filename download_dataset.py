import kagglehub

path = kagglehub.dataset_download(
    "andrewmvd/face-mask-detection"
)

print("Dataset downloaded to:")
print(path)