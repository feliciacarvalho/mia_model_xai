# membership_inference_attack/main.py

from data.load_data import load_dataset
from preprocess.preprocessing import preprocess_data
from models.target_model import train_target_model
from models.shadow_model import train_shadow_models
from models.attack_model import train_attack_model
# from xai.shap_analysis import explain_model
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


data_file = os.path.join(DATA_DIR, 'adult.data')
test_file = os.path.join(DATA_DIR, 'adult.test')
data = load_dataset(data_file, test_file)

x_train, x_test, y_train, y_test, scaler = preprocess_data(data)

# Treina modelo alvo
model_target, hist_target = train_target_model(x_train, y_train, x_test, y_test, RESULTS_DIR)

# Treina modelos shadow
shadow_models, shadow_histories = train_shadow_models(x_train, y_train, num_models=5, results_dir=RESULTS_DIR)

# Treina modelo de ataque
attack_model, attack_history = train_attack_model(shadow_models, x_train, y_train, x_test, y_test, RESULTS_DIR)

# explain_model(attack_model, x_train, x_test, RESULTS_DIR)
 