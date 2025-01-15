import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

def train_target_model(x, y, results_dir, data_size=10000, nh=5, lrate=0.001, decay=1e-7, 
                       batch_size=32, epochs=100, seed=7):
    """
    Train the target model with the specified parameters.

    Parameters:
        x (np.ndarray): Input features.
        y (np.ndarray): Target labels.
        results_dir (str): Directory to save the model and results.
        data_size (int): Number of training samples for the target model.
        nh (int): Number of hidden layers.
        lrate (float): Learning rate.
        decay (float): Learning rate decay.
        batch_size (int): Batch size.
        epochs (int): Number of epochs for training.
        seed (int): Random seed for reproducibility.

    Returns:
        model_target (Sequential): Trained target model.
        history (History): Training history of the target model.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Shuffle and split the data
    sh = np.arange(x.shape[0])
    np.random.shuffle(sh)
    print("Shuffled indices:", sh[:10])  # Debugging output

    # Split data into training and testing sets
    xtr_target = x[sh[:data_size]]
    ytr_target = y[sh[:data_size]]
    xts_target = x[sh[data_size:data_size * 2]]
    yts_target = y[sh[data_size:data_size * 2]]

   
    K.clear_session()
    model_target = Sequential([
        Dense(nh, input_shape=(x.shape[1],), activation='sigmoid', name='hidden'),
        Dense(1, activation='sigmoid', name='output')
    ])
    opt = Adam(learning_rate=lrate, decay=decay)
    model_target.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

   
    print("Training Target Model...")
    print(model_target.summary())
    history = model_target.fit(
        xtr_target, ytr_target,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(xts_target, yts_target),
        shuffle=True,
        verbose=1
    )

    print("\n\nTarget Model Training Complete")
    print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    
    model_target_name = os.path.join(results_dir, f'UCI_Adult_target_{data_size}.h5')
    model_target.save(model_target_name)
    print(f"Target Model saved at {model_target_name}")

    # Generate predictions for attack model data
    ytemp_tr_target = model_target.predict(xtr_target)
    ytemp_ts_target = model_target.predict(xts_target)
    xts_att = np.vstack((ytemp_tr_target, ytemp_ts_target))
    yts_att = np.zeros((2 * data_size, 1))
    yts_att[data_size:] = 1
    xts_att_truelabels = np.vstack((ytr_target.reshape(-1, 1), yts_target.reshape(-1, 1)))

    # Save attack model test data
    xts_att_dict = {
        'xts_att': xts_att,
        'yts_att': yts_att,
        'xts_att_truelabels': xts_att_truelabels
    }
    np.save(os.path.join(results_dir, f'att_test_data_{data_size}.npy'), xts_att_dict)

    # Save target model data representation
    target_rep = np.zeros((1, x.shape[0]))
    target_rep[0, :] = sh
    np.save(os.path.join(results_dir, f'data_adult_target_{data_size}.npy'), target_rep)

    
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Target Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'target_model_training.png'))
    plt.close()

    return model_target, history
