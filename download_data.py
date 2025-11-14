import kagglehub
import os

def download_celeba():
    """
    Downloads the CelebA dataset using kagglehub and returns the path.
    """
    print("Downloading CelebA dataset...")
    # Make sure your kaggle.json is in ~/.kaggle/ or set KAGGLE_CONFIG_DIR
    # os.environ["KAGGLE_CONFIG_DIR"] = "/path/to/your/kaggle/json"
    
    dataset_path = kagglehub.dataset_download('jessicali9530/celeba-dataset')
    
    print(f"Dataset downloaded and extracted to: {dataset_path}")
    return dataset_path

if __name__ == "__main__":
    data_path = download_celeba()
    print("\nData download complete.")
    print(f"Dataset is located at: {data_path}")
    print("You can now run 'python main.py' to start training.")