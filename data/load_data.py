import pandas as pd

def load_dataset(train_path, test_path):
    """
    Load and combine training and testing datasets.

    Parameters:
        train_path (str): Path to the training data file.
        test_path (str): Path to the testing data file.

    Returns:
        pd.DataFrame: Combined and cleaned dataset.
    """
    print("Loading datasets...")
    
    
    column_names = [
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"
    ]

    train_data = pd.read_csv(
        train_path,
        names=column_names,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?"
    )
    print("Training data loaded. Shape:", train_data.shape)
    print(train_data.head())

    test_data = pd.read_csv(
        test_path,
        names=column_names,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?",
        skiprows=1
    )
    print("Testing data loaded. Shape:", test_data.shape)
    print(test_data.head())

    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    print("Combined data. Shape:", combined_data.shape)

    combined_data.dropna(inplace=True)
    print("Data after dropping missing values. Shape:", combined_data.shape)

    return combined_data