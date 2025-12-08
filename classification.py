import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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

IMG_SIZE = (224, 224)   # image size required by EfficientNet
BATCH_SIZE = 8
EPOCHS = 20             # epochs used for cross-validation
FINAL_EPOCHS = 200      # final training
N_SPLITS = 5            # number of folds in cross-validation

# ======================
# Load Images + Labels
# ======================
print("Loading and preprocessing images...")
image_files = sorted(list(IMAGE_DIR.glob("*.png")))
label_files = sorted(list(LABEL_DIR.glob("*.png")))

X = []
y = []

def augment_image(img):
    """Applies augmentations to help balance the dataset."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img

for img_path, label_path in zip(image_files, label_files):
    # -------- IMAGE PROCESSING ---------
    img = tf.io.read_file(str(img_path))
    img = tf.image.decode_image(img, channels=1, expand_animations=False) 
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.grayscale_to_rgb(img)     # EfficientNet requires 3 channels
    img = preprocess_input(img)              # EfficientNet standard normalization

    # -------- LABEL PROCESSING ---------
    mask = tf.io.read_file(str(label_path))
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    
    # If sum > 0, we treat the mask as positive
    label = tf.cast(tf.reduce_sum(mask) > 0, tf.float32)   

    # Apply augmentation only to positive samples
    if label == 1.0:
        img = augment_image(img)

    X.append(img)
    y.append(float(label))

X = np.stack(X)
y = np.array(y)
print(f"Data loaded successfully. Output shape: {X.shape}")

# ======================
# Initial Train/Test Split
# (15% reserved for final test)
# ======================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# ======================
# Model Builder: EfficientNetB0
# ======================
def build_efficientnetb0(input_shape=(224,224,3)):
    # Load EfficientNet with ImageNet weights
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
   
    # Freeze most layers, keep last 50 trainable
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
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
# Phase 1: 5-Fold Cross-Validation
# ======================
print("\n=== Phase 1: 5-Fold Cross-Validation ===")
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
val_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
    print(f"\n--- Fold {fold+1} ---")

    X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]

    # Class weights because of the unbalanced data
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    class_weights_fold = {int(c): float(w) for c, w in zip(np.unique(y_tr), weights)}

    train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(500).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

    model = build_efficientnetb0()
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              class_weight=class_weights_fold, callbacks=[es], verbose=0)

    # Predictions and F1 score
    y_val_pred = (model.predict(val_ds) > 0.5).astype("int32").flatten()
    f1 = f1_score(y_val, y_val_pred)
    print(f"Fold {fold+1} F1-score (Best Model): {f1:.4f}")
    val_f1_scores.append(f1)

# Compute mean and std of CV results
f1_scores_array = np.array(val_f1_scores)
mean_f1 = np.mean(f1_scores_array)
std_f1 = np.std(f1_scores_array)

print("\nCross-Validation Results:")
print(f"Average F1-score: {mean_f1:.4f} ± {std_f1:.4f}")

# ======================
# Plot CV F1-scores
# ======================
folds = np.arange(1, N_SPLITS+1)
plt.figure(figsize=(8,5))
plt.plot(folds, f1_scores_array, marker='o', label='Fold F1-Score', color='blue')
plt.fill_between(folds, f1_scores_array - std_f1, f1_scores_array + std_f1, color='orange', alpha=0.2, label='±1 Std')
plt.hlines(mean_f1, 1, N_SPLITS, colors='red', linestyles='--', label=f'Mean F1={mean_f1:.4f}')
plt.title('5-Fold Cross-Validation F1-Scores')
plt.xlabel('Fold')
plt.ylabel('F1-Score')
plt.xticks(folds)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(PLOTS_DIR,"f1_scores_classification.png"), dpi=300)
plt.close()


# ======================
# Final Training 
# ======================
print("\n=== Phase 2: Training Final Model (90% Train / 10% Val) ===")
X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.10,
    stratify=y_trainval,
    random_state=42
)
print(f"Final Train samples: {len(X_final_train)} | Final Validation samples: {len(X_final_val)}")

weights_final = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_final_train), y=y_final_train
)
class_weights_final = {int(c): float(w) for c, w in zip(np.unique(y_final_train), weights_final)}
print("Class weights for final training:", class_weights_final)

final_train_ds = tf.data.Dataset.from_tensor_slices((X_final_train, y_final_train))\
    .shuffle(1000).batch(BATCH_SIZE)
final_val_ds = tf.data.Dataset.from_tensor_slices((X_final_val, y_final_val))\
    .batch(BATCH_SIZE)

final_model = build_efficientnetb0()
es_final = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)

history = final_model.fit(
    final_train_ds,
    validation_data=final_val_ds,
    epochs=FINAL_EPOCHS,
    class_weight=class_weights_final,
    callbacks=[es_final]
)

# ======================
# Plot Final Training Curves
# ======================
plt.figure(figsize=(10, 6))
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(train_loss) + 1)

plt.plot(epochs_range, train_loss, label='Training Loss', color='blue', linestyle='-')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange', linestyle='-')
plt.title('Final Model: Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (Binary Crossentropy)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(PLOTS_DIR,"final_training_validation_loss.png"), dpi=300)
plt.close()


# ======================
# Final Test Evaluation
# ======================
print("\n=== Phase 4: Evaluating on Held-out Test Set ===")
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
y_test_pred = (final_model.predict(test_ds) > 0.5).astype("int32").flatten()

print(f"\nFinal Test F1-score: {f1_score(y_test, y_test_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))


