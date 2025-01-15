import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def train_shadow_models(x, y, num_models, results_dir, data_size=10000, nh=5, lrate=0.001, decay=1e-7, 
                        batch_size=32, epochs=100, seed=7):
    """
    Train multiple shadow models for membership inference.

    Parameters:
        x (np.ndarray): Input features.
        y (np.ndarray): Target labels.
        num_models (int): Number of shadow models to train.
        results_dir (str): Directory to save the shadow models and results.
        data_size (int): Number of training samples for each shadow model.
        nh (int): Number of hidden layers.
        lrate (float): Learning rate.
        decay (float): Learning rate decay.
        batch_size (int): Batch size.
        epochs (int): Number of epochs for training.
        seed (int): Random seed for reproducibility.

    Returns:
        shadow_models (list): List of trained shadow models.
        shadow_histories (list): Training histories of the shadow models.
    """
    np.random.seed(seed)
    sh = np.arange(x.shape[0])
    sh = shuffle(sh, random_state=seed)
    
    shadow_rep = np.zeros((num_models, x.shape[0]))
    shadow_models = []
    shadow_histories = []

    os.makedirs(results_dir, exist_ok=True)

    print(f"Training {num_models} shadow models...")

    for i in range(num_models):
        print(f"\nTraining Shadow Model {i + 1}/{num_models}...")
        sh = shuffle(sh, random_state=seed + i)
        shadow_rep[i, :] = sh

        # Split data for the current shadow model
        xtr_shadow = x[sh[:data_size]]
        ytr_shadow = y[sh[:data_size]]
        xts_shadow = x[sh[data_size:2 * data_size]]
        yts_shadow = y[sh[data_size:2 * data_size]]

        
        model_shadow = Sequential([
            Dense(nh, input_shape=(x.shape[1],), activation='sigmoid', name='hidden'),
            Dense(1, activation='sigmoid', name='output')
        ])
        opt = Adam(learning_rate=lrate, decay=decay)
        model_shadow.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        print(model_shadow.summary())
        history = model_shadow.fit(
            xtr_shadow, ytr_shadow,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(xts_shadow, yts_shadow),
            shuffle=True,
            verbose=1
        )

        print(f"Shadow Model {i + 1} Training Complete")
        print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

        
        model_shadow_name = os.path.join(results_dir, f'shadow_model_{i + 1}_{data_size}.h5')
        model_shadow.save(model_shadow_name)
        print(f"Shadow Model {i + 1} saved at {model_shadow_name}")

        
        shadow_models.append(model_shadow)
        shadow_histories.append(history)

        # Generate predictions for attack model data
        ytemp_tr_shadow = model_shadow.predict(xtr_shadow)
        ytemp_ts_shadow = model_shadow.predict(xts_shadow)

        if i == 0:
            xtr_att = np.vstack((ytemp_tr_shadow, ytemp_ts_shadow))
            ytr_att = np.zeros((2 * data_size, 1))
            ytr_att[data_size:2 * data_size] = 1
            xtr_att_truelabels = np.hstack((ytr_shadow, yts_shadow))
        else:
            xtr_att = np.vstack((xtr_att, ytemp_tr_shadow, ytemp_ts_shadow))
            ytr_att = np.vstack((ytr_att, np.zeros((data_size, 1)), np.ones((data_size, 1))))
            xtr_att_truelabels = np.vstack((xtr_att_truelabels, np.hstack((ytr_shadow, yts_shadow))))

    # Save attack model training data
    attack_data = {
        'xtr_att': xtr_att,
        'ytr_att': ytr_att,
        'xtr_att_truelabels': xtr_att_truelabels
    }
    np.save(os.path.join(results_dir, f'att_train_data_{data_size}.npy'), attack_data)
    np.save(os.path.join(results_dir, f'shadow_rep_{data_size}.npy'), shadow_rep)

   
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(shadow_histories):
        plt.plot(history.history['accuracy'], label=f'Shadow Model {i + 1} Train Accuracy')
        plt.plot(history.history['val_accuracy'], label=f'Shadow Model {i + 1} Validation Accuracy')

    mean_train_acc = np.mean([h.history['accuracy'] for h in shadow_histories], axis=0)
    mean_val_acc = np.mean([h.history['val_accuracy'] for h in shadow_histories], axis=0)
    plt.plot(mean_train_acc, label='Mean Train Accuracy', linestyle='--', color='blue')
    plt.plot(mean_val_acc, label='Mean Validation Accuracy', linestyle='--', color='orange')

    plt.title('Shadow Models Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'shadow_models_training.png'))
    plt.close()

    return shadow_models, shadow_histories
