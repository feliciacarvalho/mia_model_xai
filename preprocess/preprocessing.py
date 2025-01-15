import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, data_size=10000, random_state=7):
    """
    Preprocess the dataset: encode target, one-hot encode categorical features, and scale features.

    Parameters:
        data (pd.DataFrame): Combined dataset.
        data_size (int): Number of samples to use for training/testing the target model.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Processed x_train, x_test, y_train, y_test, and the scaler used.
    """
    print("Starting preprocessing...")

    
    data['Target'] = data['Target'].map({"<=50K": 0, ">50K": 1})
    y = data['Target'].values
    data.drop('Target', axis=1, inplace=True)
    print("Target variable encoded. Distribution:")
    print(pd.value_counts(pd.Series(y)))

    
    categorical_features = data.select_dtypes(include=['object']).columns
    print("Categorical features detected:", list(categorical_features))
    data = pd.get_dummies(data, columns=categorical_features)
    print("Data after one-hot encoding. Shape:", data.shape)

    
    scaler = MinMaxScaler()
    x = scaler.fit_transform(data.astype('float32'))
    print("Features scaled using MinMaxScaler.")

    # Shuffle the indices and split manually
    np.random.seed(random_state)
    sh = np.arange(x.shape[0])
    np.random.shuffle(sh)
    print("Shuffled indices:", sh[:10]) 

    # Train-test split based on data_size
    x_train = x[sh[:data_size]]
    y_train = y[sh[:data_size]]
    x_test = x[sh[data_size:data_size * 2]]
    y_test = y[sh[data_size:data_size * 2]]

    print("Data split into training and testing sets.")
    print("Training data shape:", x_train.shape, ", Testing data shape:", x_test.shape)

    return x_train, x_test, y_train, y_test, scaler
