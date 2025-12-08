import os
import math
import shutil
import numpy as np
import random
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt

# ===========================
# Seeds for reproducibility
# (ensures consistent results across runs)
# ===========================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===========================
# Paths and Parameters
# ===========================
PLOTS_DIR = Path("/workspace/Matilde-IA/plots")
IMAGE_DIR = Path("/workspace/Matilde-IA/train/image")
MASK_DIR  = Path("/workspace/Matilde-IA/train/label")

SEG_IMG_SIZE = 256             # image size required for segmentation
BATCH_SIZE = 8
EPOCHS_PER_CYCLE = 20          # epochs per active learning cycle
MAX_ITERATIONS = 7             # maximum iterations
SAMPLES_PER_CYCLE = 30         # number of most uncertain samples to query each iteration
INITIAL_SIZE = 0.1             # fraction of dataset initially labeled
ROI_CROP = True                # whether to crop images around the region of interest (mask)

# ===========================
# METRICS AND LOSSES
# ===========================
# Dice coefficient (overlap measure)
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.*intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Dice loss (1 - Dice coefficient)
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Combined BCE + Dice loss for stable segmentation training
def bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

# Intersection over Union (IoU) metric
def iou_coef(y_true, y_pred, smooth=1e-6): 
    ndims = len(y_true.shape)

    if ndims == 4:        # batch: (batch, H, W, C)
        axis = [1, 2, 3]
    elif ndims == 3:      # single image: (H, W, C)
        axis = [0, 1, 2]
    else:
        raise ValueError(f"Unsupported input shape {y_true.shape}, expected 3D or 4D tensor")
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=axis)
    union = tf.keras.backend.sum(y_true, axis=axis) + tf.keras.backend.sum(y_pred, axis=axis) - intersection
    return tf.keras.backend.mean((intersection + smooth) / (union + smooth))

# Accuracy, Sensitivity, Specificity
def compute_pixel_metrics(y_true, y_pred):
    y_true = y_true.astype(np.uint8)
    y_pred = (y_pred > 0.5).astype(np.uint8)

    TP = np.sum((y_true==1) & (y_pred==1))
    TN = np.sum((y_true==0) & (y_pred==0))
    FP = np.sum((y_true==0) & (y_pred==1))
    FN = np.sum((y_true==1) & (y_pred==0))

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-7)
    sens = TP / (TP + FN + 1e-7)
    spec = TN / (TN + FP + 1e-7)

    return acc, sens, spec

# ============================
# DATA LOADING
# ============================
def load_image_and_mask_paths(image_dir, mask_dir):
    image_files = sorted(list(image_dir.glob("*.png")))
    mask_files  = sorted(list(mask_dir.glob("*.png")))
    assert len(image_files) == len(mask_files)
    return image_files, mask_files

# Read mask and convert to binary (0 or 1)
def read_mask_binary(path):
    m = tf.io.read_file(str(path))
    m = tf.image.decode_png(m, channels=1)
    m = tf.cast(m > 0, tf.float32)
    return m

# ============================
# FILTER POSITIVE SAMPLES
# ============================
image_files, mask_files = load_image_and_mask_paths(IMAGE_DIR, MASK_DIR)
positive_images, positive_masks = [], []

# Keep only images with a positive mask
print("Filtering dataset for positive masks...")
for img_p, m_p in zip(image_files, mask_files):
    m = read_mask_binary(m_p)
    if tf.reduce_sum(m) > 0:
        positive_images.append(str(img_p))
        positive_masks.append(str(m_p))

print(f"Total Positive Images: {len(positive_images)}")
if len(positive_images) == 0:
    raise RuntimeError("No positive images found.")

# ============================
# ROI CROP FUNCTION
# ============================
def roi_crop_numpy(img_u8, mask_u8, crop_size=SEG_IMG_SIZE):
    # Crop around the centroid of the mask for more focused learning
    h, w = img_u8.shape[:2]
    ys, xs = np.where(mask_u8.squeeze() > 0)
    
    # Fallback:: if mask is empty, resize the full image
    if len(xs) == 0:
        img_r = tf.image.resize(img_u8, (crop_size, crop_size)).numpy()
        mask_r = tf.image.resize(mask_u8, (crop_size, crop_size), method='nearest').numpy()
        return img_r, mask_r

    cx, cy = int(xs.mean()), int(ys.mean())
    x1 = max(0, min(cx - crop_size//2, w - crop_size))
    y1 = max(0, min(cy - crop_size//2, h - crop_size))
    return img_u8[y1:y1+crop_size, x1:x1+crop_size], mask_u8[y1:y1+crop_size, x1:x1+crop_size]

# ============================
# DATA GENERATOR (TRAIN + VAL)
# ============================
def generator(img_paths, mask_paths, batch=4, augment=False, to_3ch=False, roi_crop=ROI_CROP):
    n = len(img_paths)
    indices = np.arange(n)
    while True:
        np.random.shuffle(indices)
        for start in range(0, n, batch):
            batch_idx = indices[start:start+batch]
            Xb, Yb = [], []
            for i in batch_idx:

                img = tf.io.read_file(img_paths[i])
                img = tf.image.decode_png(img, channels=1)
                mask = tf.io.read_file(mask_paths[i])
                mask = tf.image.decode_png(mask, channels=1)

                img_np = img.numpy()
                mask_np = (mask.numpy() > 0).astype(np.uint8)

                # Crop or resize
                if roi_crop:
                    img_crop, mask_crop = roi_crop_numpy(img_np, mask_np)
                else:
                    img_crop = tf.image.resize(img, (SEG_IMG_SIZE, SEG_IMG_SIZE)).numpy()
                    mask_crop = tf.image.resize(mask, (SEG_IMG_SIZE, SEG_IMG_SIZE), method='nearest').numpy()

                # Normalize
                img_f = img_crop.astype(np.float32)/255.0
                mask_f = (mask_crop > 0).astype(np.float32)

                # Optional augmentation
                if augment:
                    if np.random.rand() < 0.5:
                        img_f = np.fliplr(img_f)
                        mask_f = np.fliplr(mask_f)
                    if np.random.rand() < 0.3:
                        img_f = np.flipud(img_f)
                        mask_f = np.flipud(mask_f)
                    if np.random.rand() < 0.5:
                        angle = np.random.uniform(-10, 10)
                        img_f = tfa.image.rotate(img_f, angle*np.pi/180, interpolation='BILINEAR').numpy()
                        mask_f = tfa.image.rotate(mask_f, angle*np.pi/180, interpolation='NEAREST').numpy()
                    if np.random.rand() < 0.4:
                        img_f = tf.image.random_brightness(img_f, 0.12).numpy()
                    if np.random.rand() < 0.4:
                        img_f = tf.image.random_contrast(img_f, 0.85, 1.15).numpy()
                        img_f = np.clip(img_f, 0, 1)
                
                # Convert to 3 channels if needed
                if to_3ch:
                    img_f = np.repeat(img_f, 3, axis=-1)

                Xb.append(img_f)
                Yb.append(mask_f)

            yield np.stack(Xb), np.stack(Yb)

# =========================
# U-NET (VGG16)
# =========================
# Convolutional block
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    return x

# Up-sampling block with skip connections
def up_block(x, skip, filters):
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

# U-Net with VGG16 encoder
def build_unet_vgg16(input_shape=(SEG_IMG_SIZE, SEG_IMG_SIZE, 3)):
    inp = layers.Input(shape=input_shape)
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=inp)

    for layer in vgg.layers:
        layer.trainable = False

    c1 = vgg.get_layer("block1_conv2").output
    c2 = vgg.get_layer("block2_conv2").output
    c3 = vgg.get_layer("block3_conv3").output
    c4 = vgg.get_layer("block4_conv3").output
    bridge = vgg.get_layer("block5_conv3").output

    u1 = up_block(bridge, c4, 512)
    u2 = up_block(u1, c3, 256)
    u3 = up_block(u2, c2, 128)
    u4 = up_block(u3, c1, 64)
    o = layers.Conv2D(32, 3, activation='relu', padding='same')(u4)
    out = layers.Conv2D(1,1,activation='sigmoid')(o)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_coef]
    )
    return model

# =======================
# DATA SPLITS
# =======================
X_all = positive_images
Y_all = positive_masks

# Split positive images into train/test (15% test)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, Y_all, test_size=0.15, random_state=42
)

# Split train+val into initial pool and validation set (12% for validation)
X_pool_full, X_val, y_pool_full, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.12, random_state=42
)

# Initial labeled/unlabeled pools for active learning
X_labeled_init, X_unlabeled_init, y_labeled_init, y_unlabeled_init = train_test_split(
    X_pool_full, y_pool_full, train_size=INITIAL_SIZE, random_state=42
)

print(f"\n--- Data Split Summary ---")
print(f"Test Set: {len(X_test)}")
print(f"Validation Set: {len(X_val)}")
print(f"Initial Labeled Train: {len(X_labeled_init)}")
print(f"Unlabeled Pool: {len(X_unlabeled_init)}")

# =========================
# EVALUATION
# =========================
def full_metrics(model, X_paths, y_paths):
    dices, ious = [], []
    accs, senss, specs = [], [], []

    # Loop through each image-mask pair
    for img_path, mask_path in zip(X_paths, y_paths):

        # ====== PREPROCESS IMAGE ======
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, (SEG_IMG_SIZE, SEG_IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.repeat(img, 3, axis=-1)       # Repeat the single channel to 3 channels (needed for VGG16 encoder)
        img = tf.expand_dims(img, 0)

        # ====== PREDICT ======
        # Predict the mask using the model
        pred = model.predict(img, verbose=0)[0]
        pred_bin = (pred > 0.5).astype(np.uint8)

        # ====== PREPROCESS MASK ======
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, (SEG_IMG_SIZE, SEG_IMG_SIZE), method='nearest')
        mask = tf.cast(mask > 0, tf.uint8)
        mask = mask.numpy()

        # ====== DICE & IoU ======
        # Compute Dice coefficient and IoU between ground truth and prediction
        dices.append(dice_coef(mask.astype(np.float32), pred_bin.astype(np.float32)).numpy())
        ious.append(iou_coef(mask.astype(np.float32), pred_bin.astype(np.float32)).numpy())

        # ====== ACCURACY, SENSITIVITY, SPECIFICITY ======
        TP = np.sum((mask==1) & (pred_bin==1))
        TN = np.sum((mask==0) & (pred_bin==0))
        FP = np.sum((mask==0) & (pred_bin==1))
        FN = np.sum((mask==1) & (pred_bin==0))

        acc = (TP + TN) / (TP + TN + FP + FN + 1e-7)
        sens = TP / (TP + FN + 1e-7)
        spec = TN / (TN + FP + 1e-7)

        accs.append(acc)
        senss.append(sens)
        specs.append(spec)

    return {
        "Dice": float(np.mean(dices)),
        "IoU": float(np.mean(ious)),
        "Accuracy": float(np.mean(accs)),
        "Sensitivity": float(np.mean(senss)),
        "Specificity": float(np.mean(specs))
    }

# ======================
# Entropy Utility
# ======================
def calculate_entropy(probs):
    epsilon = 1e-7
    probs = np.clip(probs, epsilon, 1 - epsilon)
    entropies = - (probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
    return np.mean(entropies, axis=(1,2,3))

# ======================
# Least Confidence Utility
# ======================
def least_confidence_sampling(probs, k):
    scores = np.mean(1 - np.abs(probs - 0.5), axis=(1,2,3))
    return np.argsort(scores)[-k:]

# =======================
# PREPROCESS IMAGE FOR AL POOL
# Prepare images for prediction in the unlabeled pool
# =======================
def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, (SEG_IMG_SIZE, SEG_IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.repeat(img, 3, axis=-1)
    return img

# ========================
# ACTIVE LEARNING LOOP
# iteratively train model, evaluate, and query most uncertain samples
# ========================
strategies = ["entropy", "least_confidence", "random"]
al_history_dict = {}
al_metrics_dict = {}
final_train_data = {}
final_models = {}

# Preprocess image and mask for training
def preprocess_for_training(img_path, mask_path):
    # Image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, (SEG_IMG_SIZE, SEG_IMG_SIZE))
    img = tf.cast(img, tf.float32)/255.0
    img = tf.repeat(img, 3, axis=-1)

    # Mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (SEG_IMG_SIZE, SEG_IMG_SIZE), method='nearest')
    mask = tf.cast(mask > 0, tf.float32)

    return img, mask

# Prepare validation dataset
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(lambda x, y: preprocess_for_training(x, y))
val_ds = val_ds.batch(BATCH_SIZE)

# Active Learning Main Loop
for SAMPLING_STRATEGY in strategies:
    print(f"\n===== Strategy: {SAMPLING_STRATEGY} =====")

    # Initialize labeled and unlabeled datasets for current strategy
    X_labeled = X_labeled_init.copy()
    y_labeled = y_labeled_init.copy()
    X_unlabeled = X_unlabeled_init.copy()
    y_unlabeled = y_unlabeled_init.copy()

    history_cycles = []
    metrics_cycles = []

    dice_best = -1
    dice_wait = 0
    DICE_LIMIT = 3     # Stop if Dice does not improve for this many iterations

    iteration = 0
    while iteration < MAX_ITERATIONS and len(X_unlabeled) > 0:
        print(f"\nIteration {iteration} - Labeled: {len(X_labeled)}")
        
        # Create training dataset
        train_ds = tf.data.Dataset.from_tensor_slices((X_labeled, y_labeled))
        train_ds = train_ds.map(lambda x, y: preprocess_for_training(x, y))
        train_ds = train_ds.shuffle(500).batch(BATCH_SIZE)
        
        # Clear previous model session
        tf.keras.backend.clear_session()
        model = build_unet_vgg16() 
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        
        # Train the model
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PER_CYCLE, callbacks=[es], verbose=0)
        history_cycles.append(history.history)
        
        # Evaluate metrics on validation set
        metrics = full_metrics(model, X_val, y_val)
        metrics['num_labeled_samples'] = len(X_labeled)
        metrics_cycles.append(metrics)

        print(f"Iteration {iteration} metrics: Dice={metrics['Dice']:.4f}, IoU={metrics['IoU']:.4f}, Loss={history.history['loss'][-1]:.4f}")
        
        # Early stopping
        if metrics['Dice'] > dice_best + 1e-4:
            dice_best = metrics['Dice']
            dice_wait = 0
        else:
            dice_wait += 1

        if dice_wait >= DICE_LIMIT:
            print(f"Stopping AL: Dice did not improve for {DICE_LIMIT} iterations.")
            break
    
        if len(X_unlabeled) == 0:
            break
        
        # Query samples
        pool_ds = tf.data.Dataset.from_tensor_slices(X_unlabeled)
        pool_ds = pool_ds.map(preprocess_image).batch(BATCH_SIZE)
        probs_pool_masks = model.predict(pool_ds)

        n_query = min(SAMPLES_PER_CYCLE, len(X_unlabeled))

        # Select samples according to sampling strategy
        if SAMPLING_STRATEGY == "entropy":
            uncertainties = calculate_entropy(probs_pool_masks)
            query_indices = np.argsort(uncertainties)[-n_query:]
        elif SAMPLING_STRATEGY == "least_confidence":
            query_indices = least_confidence_sampling(probs_pool_masks, n_query)
        else:  # random
            query_indices = np.random.choice(len(X_unlabeled), size=n_query, replace=False)
        
        # Add queried samples to labeled set and remove from unlabeled set
        new_X = [X_unlabeled[i] for i in query_indices]
        new_y = [y_unlabeled[i] for i in query_indices]
        X_labeled.extend([X_unlabeled[i] for i in query_indices])
        y_labeled.extend([y_unlabeled[i] for i in query_indices])
        for idx in sorted(query_indices, reverse=True):
            X_unlabeled.pop(idx)
            y_unlabeled.pop(idx)

        iteration += 1

    al_history_dict[SAMPLING_STRATEGY] = history_cycles
    al_metrics_dict[SAMPLING_STRATEGY] = metrics_cycles
    final_train_data[SAMPLING_STRATEGY] = (X_labeled.copy(), y_labeled.copy())
    final_models[SAMPLING_STRATEGY] = model

# ============================
# METRICS AFTER ACTIVE LEARNING PER STRATEGY
# ============================
for strategy in strategies:
    print(f"\n=== Final metrics for strategy {strategy} ===")
    metrics_after_AL = full_metrics(final_models[strategy], X_test, y_test)
    
    for k,v in metrics_after_AL.items():
        print(f"{k}: {v:.4f}")

# ========================
# Plot Dice/IoU vs number of labeled samples
# ========================
plt.figure(figsize=(12,5))

# Subplot 1: Dice
plt.subplot(1,2,1)
for strategy in strategies:
    labeled_counts = [m['num_labeled_samples'] for m in al_metrics_dict[strategy]]
    dices = [m['Dice'] for m in al_metrics_dict[strategy]]
    plt.plot(labeled_counts, dices, marker='o', label=strategy)
plt.xlabel("Number of Labeled Samples")
plt.ylabel("Dice Score")
plt.title("Dice vs Number of Labeled Samples")
plt.grid(True)
plt.legend()

# Subplot 2: IoU
plt.subplot(1,2,2)
for strategy in strategies:
    labeled_counts = [m['num_labeled_samples'] for m in al_metrics_dict[strategy]]
    ious = [m['IoU'] for m in al_metrics_dict[strategy]]
    plt.plot(labeled_counts, ious, marker='o', label=strategy)
plt.xlabel("Number of Labeled Samples")
plt.ylabel("IoU Score")
plt.title("IoU vs Number of Labeled Samples")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR,"segmentation_dice_iou_vs_samples_AL.png"), dpi=300)
plt.close()

# ===========================
# ACTIVE LEARNING - PLOT
# ===========================
metrics = ["loss", "dice_coef", "iou_coef"]
titles = ["Loss", "Dice", "IoU"]

for metric, title in zip(metrics, titles):
    plt.figure(figsize=(15,5))
    for i, strategy in enumerate(strategies):

        plt.subplot(1,3,i+1)

        all_cycles = al_history_dict[strategy]
        
        min_epochs = min(len(cycle[metric]) for cycle in all_cycles)
        cycles_trimmed = [cycle[metric][:min_epochs] for cycle in all_cycles]
        mean_metric = np.mean(cycles_trimmed, axis=0)
        
        plt.plot(mean_metric, label="mean over cycles")
        
        val_metric = "val_" + metric
        if val_metric in all_cycles[0]:
            cycles_trimmed_val = [cycle[val_metric][:min_epochs] for cycle in all_cycles]
            mean_val = np.mean(cycles_trimmed_val, axis=0)
            plt.plot(mean_val, label="val mean over cycles")
        
        plt.title(f"{strategy} - {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.grid(True)
        plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,f"segmentation_AL_{title}_by_strategy.png"), dpi=300)
    plt.close()

# ==========================
# FINAL TRAINING
# ==========================
history_final_all = {}
final_metrics_all = {}

for strat in strategies:
    print(f"\n=== Final Training for {strat} ===")
    
    X_final_train, y_final_train = final_train_data[strat]

    # Split 90/10
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_final_train, y_final_train, test_size=0.1, random_state=42
    )

    train_ds_final = ( tf.data.Dataset.from_tensor_slices((X_train_final, y_train_final)) .map(lambda x, y: preprocess_for_training(x, y)) .shuffle(500) .batch(BATCH_SIZE) )
    val_ds_final = ( tf.data.Dataset.from_tensor_slices((X_val_final, y_val_final)) .map(lambda x, y: preprocess_for_training(x, y)) .batch(BATCH_SIZE) )

    tf.keras.backend.clear_session()
    model_final = build_unet_vgg16()

    es_final = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history_final = model_final.fit(
        train_ds_final,
        validation_data=val_ds_final,
        epochs=EPOCHS_PER_CYCLE + 5,
        callbacks=[es_final],
        verbose=1
    )
    history_final_all[strat] = history_final.history
    metrics = full_metrics(model_final, X_test, y_test)
    final_metrics_all[strat] = metrics

# ==========================
# FINAL TRAINING PLOTS
# ==========================
for strat in strategies:
    history = history_final_all[strat]
    
    plt.figure(figsize=(18,5))
    
    # Loss
    plt.subplot(1,3,1)
    plt.plot(history['loss'], label='train_loss')
    plt.plot(history.get('val_loss', []), label='val_loss')
    plt.title(f'{strat} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Dice
    plt.subplot(1,3,2)
    plt.plot(history['dice_coef'], label='train_dice')
    plt.plot(history.get('val_dice_coef', []), label='val_dice')
    plt.title(f'{strat} - Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.grid(True)
    
    # IoU
    plt.subplot(1,3,3)
    plt.plot(history['iou_coef'], label='train_iou')
    plt.plot(history.get('val_iou_coef', []), label='val_iou')
    plt.title(f'{strat} - IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,f'segmentation_final_training_{strat}.png'), dpi=300)
    plt.close()

# ========================
# FINAL METRICS PER STRATEGY
# ========================
print("\n=== FINAL TRAINING METRICS PER STRATEGY ===")
for strat in strategies:
    print(f"\n--- Strategy: {strat} ---")
    metrics = final_metrics_all[strat]
    print(f"Dice Coefficient: {metrics['Dice']:.4f}")
    print(f"IoU: {metrics['IoU']:.4f}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f"Specificity: {metrics['Specificity']:.4f}")
