import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

# ======================
# Seeds for reproducibility
# (ensures consistent results across runs)
# ======================
tf.random.set_seed(42)
np.random.seed(42)

# ======================
# Paths and Parameters
# ======================
PLOTS_DIR = Path("/workspace/Matilde-IA/plots")
IMAGE_DIR = Path("/workspace/Matilde-IA/train/image")
LABEL_DIR = Path("/workspace/Matilde-IA/train/label")

IMG_SIZE = (224, 224)         # image size required by EfficientNet
BATCH_SIZE = 8
EPOCHS = 20                   # training epochs per AL iteration
QUERIES_PER_IT = 30           # number of most uncertain samples to query each iteration
INITIAL_SIZE = 0.1            # fraction of dataset initially labeled
MAX_ITERATIONS = 50           # maximum iterations to avoid infinite loops
F1_LIMIT = 3                  # stop if F1 hasn't improved for 3 iterations

# ======================
# Load Images + Labels
# ======================
print("Loading data...")
image_files = sorted(list(IMAGE_DIR.glob("*.png")))
label_files = sorted(list(LABEL_DIR.glob("*.png")))

X = []
y = []

def augment_image(img):
    # Applies augmentations to help balance the dataset
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img

for img_path, label_path in zip(image_files, label_files):
    # -------- IMAGE ---------
    img = tf.io.read_file(str(img_path))
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.grayscale_to_rgb(img)    # EfficientNet requires 3 channels
    img = preprocess_input(img)             # EfficientNet standard normalization
    
    # -------- LABEL ---------
    mask = tf.io.read_file(str(label_path))
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)

    # If sum > 0, we treat the mask as positive
    label = tf.cast(tf.reduce_sum(mask) > 0, tf.float32)
    
    # Apply augmentation only for positive samples
    if label == 1.0:
        img = augment_image(img)
    
    X.append(img)
    y.append(float(label))

X = np.stack(X)
y = np.array(y)
print(f"Total dataset shape: {X.shape}")

# ======================
# Data Split
# ======================
# Hold-out test set (15%)
X_remain, X_test, y_remain, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Validation set (~10% of total)
X_pool_full, X_val, y_pool_full, y_val = train_test_split(
    X_remain, y_remain, test_size=0.12, stratify=y_remain, random_state=42
)

# Initial labeled/unlabeled pools for active learning
X_labeled_init, X_unlabeled_init, y_labeled_init, y_unlabeled_init = train_test_split(
    X_pool_full, y_pool_full, train_size=INITIAL_SIZE, stratify=y_pool_full, random_state=42
)

print(f"\n--- Data Split Summary ---")
print(f"Test Set: {len(X_test)}")
print(f"Validation Set: {len(X_val)}")
print(f"Initial Labeled Train: {len(X_labeled_init)}")
print(f"Unlabeled Pool: {len(X_unlabeled_init)}")

# TensorFlow datasets for validation and test sets
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# ======================
# Model Builder: EfficientNetB0
# ======================
def build_efficientnetb0(input_shape=(224,224,3)):
    # Load EfficientNet with ImageNet weights
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Freeze BatchNorm layers and other layers to retain pretrained features
    for layer in base_model.layers:
        if 'bn' in layer.name:
            layer.trainable = False
        else:
            layer.trainable = False
    # Last 50 convolutional layers trainable
    for layer in base_model.layers[-50:]:
        if 'bn' not in layer.name:
            layer.trainable = True
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)

    # Convert feature maps to a single vector per image
    x = layers.GlobalAveragePooling2D()(x)

    # To reduce overfitting and help generalization
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)    # predicts probability of positive class
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ======================
# Entropy Utility
# ======================
def calculate_entropy(probs):
    epsilon = 1e-7
    probs = np.clip(probs, epsilon, 1 - epsilon)
    entropy = - (probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
    return entropy

# ======================
# Least Confidence Utility
# ======================
def least_confidence_sampling(probs, k):
    scores = 1 - np.abs(probs - 0.5)
    return np.argsort(scores)[-k:]

# ======================
# Active Learning Loop
# iteratively train model, evaluate, and query most uncertain samples
# ======================
strategies = ["entropy", "least_confidence", "random"]
f1_history_all = {}
loss_history_all = {}
final_train_data = {}

for SAMPLING_STRATEGY in strategies:
    print(f"\n===== Strategy: {SAMPLING_STRATEGY} =====")
    
    X_labeled = X_labeled_init.copy()
    y_labeled = y_labeled_init.copy()
    X_unlabeled = X_unlabeled_init.copy()
    y_unlabeled = y_unlabeled_init.copy()
    
    f1_history = []
    labeled_sizes = []
    loss_history = []

    best_f1 = 0
    f1_wait = 0
    iteration = 0

    while iteration < MAX_ITERATIONS and len(X_unlabeled) > 0:
        print(f"\nIteration {iteration} - Labeled: {len(X_labeled)}")
        
        # Compute class weights to handle imbalance
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_labeled), y=y_labeled)
        cw_dict = {int(c): float(w) for c, w in zip(np.unique(y_labeled), weights)}
        
        train_ds = tf.data.Dataset.from_tensor_slices((X_labeled, y_labeled)).shuffle(500).batch(BATCH_SIZE)
        
        tf.keras.backend.clear_session()       # reset session before new model
        model = build_efficientnetb0()
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=cw_dict, callbacks=[es], verbose=0)
        
        # Track losses
        loss_history.append({'loss': history.history['loss'], 'val_loss': history.history['val_loss']})
        
        # Evaluate F1
        y_pred_val = (model.predict(val_ds) > 0.5).astype("int32").flatten()
        f1_current = f1_score(y_val, y_pred_val)
        f1_history.append(f1_current)
        labeled_sizes.append(len(X_labeled))
        print(f"Val Loss: {history.history['val_loss'][-1]:.4f} | F1: {f1_current:.4f}")

        if f1_current > best_f1 + 1e-4:
            best_f1 = f1_current
            f1_wait = 0
        else:
            f1_wait += 1

        if f1_wait >= F1_LIMIT:
            print(f"Stopping AL: F1 did not improve for {F1_LIMIT} iterations.")
            break
        
        if len(X_unlabeled) == 0:
            break
        
        # Query samples
        pool_ds = tf.data.Dataset.from_tensor_slices(X_unlabeled).batch(BATCH_SIZE)
        probs_pool = model.predict(pool_ds).flatten()
        
        if SAMPLING_STRATEGY == "entropy":
            uncertainties = calculate_entropy(probs_pool)
            query_indices = np.argsort(uncertainties)[-QUERIES_PER_IT:]

        elif SAMPLING_STRATEGY == "least_confidence":
            query_indices = least_confidence_sampling(probs_pool, QUERIES_PER_IT)
        else:
            query_indices = np.random.choice(len(X_unlabeled), size=QUERIES_PER_IT, replace=False)
        
        # Add queried samples to labeled set and remove from unlabeled pool
        new_X = X_unlabeled[query_indices]
        new_y = y_unlabeled[query_indices]
        X_labeled = np.concatenate([X_labeled, new_X], axis=0)
        y_labeled = np.concatenate([y_labeled, new_y], axis=0)
        X_unlabeled = np.delete(X_unlabeled, query_indices, axis=0)
        y_unlabeled = np.delete(y_unlabeled, query_indices, axis=0)

        iteration += 1
    
    f1_history_all[SAMPLING_STRATEGY] = (labeled_sizes, f1_history)
    loss_history_all[SAMPLING_STRATEGY] = loss_history
    final_train_data[SAMPLING_STRATEGY] = (X_labeled.copy(), y_labeled.copy())

# ======================
# Plot F1 score evolution over iterations
# ======================
plt.figure(figsize=(8,5))
colors = {'entropy':'forestgreen', 'least_confidence':'royalblue', 'random':'darkorange'}
markers = {'entropy':'o', 'least_confidence':'s', 'random':'^'}
for strat in strategies:
    plt.plot(range(len(f1_history_all[strat][1])), f1_history_all[strat][1], marker=markers[strat], color=colors[strat],
        linewidth=2, label=strat.replace('_',' ').title())

plt.title("F1 Score Evolution over AL Iterations")
plt.xlabel("AL Iteration")
plt.ylabel("F1-Score")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR,"classification_al_hitl_f1_iterations.png"), dpi=300)
plt.close()

# ======================
# Plot Comparative Learning Curves
# ======================
plt.figure(figsize=(10,6))
colors = {'entropy':'forestgreen', 'least_confidence':'royalblue', 'random':'darkorange'}
markers = {'entropy':'o', 'least_confidence':'s', 'random':'^'}

for strat in strategies:
    sizes, f1s = f1_history_all[strat]
    plt.plot(sizes, f1s, marker=markers[strat], color=colors[strat], linewidth=2, label=strat.replace('_',' ').title())

plt.xlabel("Number of Labeled Samples")
plt.ylabel("F1 Score")
plt.title("Active Learning Strategy Comparison")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "learning_curve_comparison.png"), dpi=300)
plt.close()
print("Comparative learning curve plot saved.")

# ======================
# Plot Training and validation loss over AL cycles for all strategies
# ======================
colors = {'entropy':'forestgreen', 'least_confidence':'royalblue', 'random':'darkorange'}

fig, axes = plt.subplots(1, 3, figsize=(20,6), sharey=True)

for idx, strat in enumerate(strategies):
    all_train_losses = []
    all_val_losses = []
    for data in loss_history_all[strat]:
        all_train_losses.extend(data['loss'])
        all_val_losses.extend(data['val_loss'])
    
    axes[idx].plot(all_train_losses, label='Train Loss', color='royalblue', linewidth=2)
    axes[idx].plot(all_val_losses, label='Val Loss', color='darkorange', linewidth=2)
    axes[idx].set_title(strat.replace('_',' ').title())
    axes[idx].set_xlabel("Cumulative Epochs")
    if idx == 0:
        axes[idx].set_ylabel("Binary Crossentropy")
    axes[idx].grid(True, linestyle=':', alpha=0.6)
    axes[idx].legend()

plt.suptitle("Training & Validation Loss across AL cycles (All Strategies)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(PLOTS_DIR,"classification_al_hitl_train_val_loss_all_strategies.png"), dpi=300)
plt.close()

# ======================
# Final Training
# ======================
history_final_all = {}
f1_final_all = {}

for strat in strategies:
    print(f"\n=== Final Training for {strat} ===")
    
    X_final_train, y_final_train = final_train_data[strat]
    
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_final_train, y_final_train, test_size=0.1, stratify=y_final_train, random_state=42
    )

    train_ds_final = tf.data.Dataset.from_tensor_slices((X_train_final, y_train_final)).shuffle(500).batch(BATCH_SIZE)
    val_ds_final = tf.data.Dataset.from_tensor_slices((X_val_final, y_val_final)).batch(BATCH_SIZE)

    tf.keras.backend.clear_session()
    model_final = build_efficientnetb0()
    weights_final = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_final), y=y_train_final)
    cw_dict_final = {int(c): float(w) for c, w in zip(np.unique(y_train_final), weights_final)}

    es_final = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history_final = model_final.fit(train_ds_final, validation_data=val_ds_final, epochs=EPOCHS, class_weight=cw_dict_final, verbose=1)

    # Final evaluation on test set
    y_pred_final = (model_final.predict(test_ds) > 0.5).astype("int32").flatten()
    print("Classification Report:\n", classification_report(y_test, y_pred_final))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))

    history_final_all[strat] = history_final.history
    f1_final_all[strat] = f1_score(y_test, y_pred_final)

# ======================
# Final Training Plots
# ======================
fig, axes = plt.subplots(1, 3, figsize=(20,6), sharey=True)

for idx, strat in enumerate(strategies):
    all_train_losses = history_final_all[strat]['loss']
    all_val_losses = history_final_all[strat]['val_loss']
    
    axes[idx].plot(all_train_losses, label='Train Loss', color='royalblue', linewidth=2)
    axes[idx].plot(all_val_losses, label='Val Loss', color='darkorange', linewidth=2)
    axes[idx].set_title(strat.replace('_',' ').title())
    axes[idx].set_xlabel("Epochs")
    if idx == 0:
        axes[idx].set_ylabel("Binary Crossentropy")
    axes[idx].grid(True, linestyle=':', alpha=0.6)
    axes[idx].legend()

plt.suptitle("Final Training Loss (All Strategies)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(PLOTS_DIR,"classification_al_hitl_final_training_loss_all_strategies.png"), dpi=300)
plt.close()

# ======================
# Print F1 final for each strategy
# ======================
for strat in strategies:
    print(f"Final Test F1-Score ({strat}): {f1_final_all[strat]:.4f}")
