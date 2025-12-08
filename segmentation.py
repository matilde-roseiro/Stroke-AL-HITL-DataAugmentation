import os
import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16, ResNet50
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt

# ======================
# Seeds for reproducibility
# (ensures consistent results across runs)
# ======================
tf.random.set_seed(42)
np.random.seed(42)

# ========================
# Paths and Parameters
# ========================
IMAGE_DIR = Path("/workspace/Matilde-IA/train/image")
MASK_DIR  = Path("/workspace/Matilde-IA/train/label")

SEG_IMG_SIZE = 256    # image size required for segmentation
BATCH_SIZE = 16
EPOCHS = 25           # number of training epochs
ROI_CROP = True       # whether to crop images around the region of interest (mask)

# ========================
# METRICS AND LOSSES
# ========================
# Dice coefficient (overlap measure)
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) +
                                           tf.keras.backend.sum(y_pred_f) + smooth)

# Dice loss (1 - Dice coefficient)
def dice_loss_tf(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Combined BCE + Dice loss for stable segmentation training
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dl = dice_loss_tf(y_true, y_pred)
    return 0.5 * bce + 0.5 * dl

# Intersection over Union (IoU) metric
def iou_coef(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1,2,3])
    union = (tf.reduce_sum(y_true, axis=[1,2,3]) +
             tf.reduce_sum(y_pred, axis=[1,2,3]) - intersection)
    return tf.reduce_mean((intersection + smooth) / (union + smooth))

# Sensitivity / Recall
def sensitivity(y_true, y_pred):
    tp = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return tp / (possible + tf.keras.backend.epsilon())

# Specificity
def specificity(y_true, y_pred):
    y_true_n = 1 - y_true
    y_pred_n = 1 - y_pred
    tn = tf.reduce_sum(tf.round(tf.clip_by_value(y_true_n * y_pred_n, 0, 1)))
    possible_n = tf.reduce_sum(tf.round(tf.clip_by_value(y_true_n, 0, 1)))
    return tn / (possible_n + tf.keras.backend.epsilon())

# Binary accuracy
def accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)

METRICS_LIST = [dice_coef, iou_coef, accuracy, sensitivity, specificity]

# ========================
# DATA LOADING HELPERS
# ========================
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

# ========================
# FILTER POSITIVE SAMPLES
# ========================
image_files, mask_files = load_image_and_mask_paths(IMAGE_DIR, MASK_DIR)
positive_images, positive_masks = [], []

# Keep only images with a positive mask
print("Filtering dataset...")
for img_p, m_p in zip(image_files, mask_files):
    m = read_mask_binary(m_p)
    if tf.reduce_sum(m) > 0:
        positive_images.append(str(img_p))
        positive_masks.append(str(m_p))

print(f"Positive samples: {len(positive_images)} / {len(image_files)}")
if len(positive_images) == 0:
    raise RuntimeError("No positive images found.")

# ========================
# ROI CROP FUNCTION
# ========================
def roi_crop_numpy(img_u8, mask_u8, crop_size=SEG_IMG_SIZE):
    # Crop around the centroid of the mask for more focused learning
    h, w = img_u8.shape[:2]
    ys, xs = np.where(mask_u8.squeeze() > 0)
    if len(xs) == 0:
        # Fallback: center crop if mask is empty
        y0 = max(0, (h - crop_size)//2)
        x0 = max(0, (w - crop_size)//2)
        return img_u8[y0:y0+crop_size, x0:x0+crop_size, :], mask_u8[y0:y0+crop_size, x0:x0+crop_size, :]

    cx = int(xs.mean())
    cy = int(ys.mean())
    x1 = max(0, min(cx - crop_size//2, w - crop_size))
    y1 = max(0, min(cy - crop_size//2, h - crop_size))
    return img_u8[y1:y1+crop_size, x1:x1+crop_size, :], mask_u8[y1:y1+crop_size, x1:x1+crop_size, :]

# ========================
# DATA GENERATOR (TRAIN + VAL)
# ========================
def generator(img_paths, mask_paths, batch=4, augment=False, to_3ch=False, roi_crop=ROI_CROP):
    n = len(img_paths)
    while True:
        idxs = np.random.permutation(n)
        for start in range(0, n, batch):
            batch_idx = idxs[start:start+batch]
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
                img_f = img_crop.astype(np.float32) / 255.0
                mask_f = mask_crop.astype(np.float32)

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
                    if img_f.ndim == 2:
                        img_f = np.expand_dims(img_f, -1)
                    img_f = np.repeat(img_f, 3, axis=-1)

                # Ensure channel dimension exists
                if img_f.ndim == 2:
                    img_f = np.expand_dims(img_f, -1)
                if mask_f.ndim == 2:
                    mask_f = np.expand_dims(mask_f, -1)

                Xb.append(img_f)
                Yb.append(mask_f)

            yield np.stack(Xb), np.stack(Yb)

# ========================
# U-NET BASE MODELS
# ========================
# Base convolutional block
def conv_block(x, filters):
    x = layers.Conv2D(filters,3,activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters,3,activation='relu',padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

# Up-sampling block with skip connections
def up_block(x, skip, filters):
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

# Base U-Net architecture
def build_unet_base(input_shape=(SEG_IMG_SIZE, SEG_IMG_SIZE,1)):
    inp = layers.Input(shape=input_shape)
    c1 = conv_block(inp, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64);  p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256); p4 = layers.MaxPooling2D()(c4)
    bn = conv_block(p4, 512)

    u5 = up_block(bn, c4, 256)
    u6 = up_block(u5, c3, 128)
    u7 = up_block(u6, c2, 64)
    u8 = up_block(u7, c1, 32)

    out = layers.Conv2D(1,1,activation='sigmoid')(u8)
    model = models.Model(inp,out)
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss=bce_dice_loss, metrics=METRICS_LIST)
    return model

# U-Net with VGG16 encoder
def build_unet_vgg16(input_shape=(SEG_IMG_SIZE, SEG_IMG_SIZE,3)):
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

    o = layers.Conv2D(32,3,activation='relu',padding='same')(u4)
    out = layers.Conv2D(1,1,activation='sigmoid')(o)

    model = models.Model(inp,out)
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss=bce_dice_loss, metrics=METRICS_LIST)
    return model

# U-Net with ResNet50 encoder
def build_unet_resnet50(input_shape=(SEG_IMG_SIZE, SEG_IMG_SIZE,3)):
    inp = layers.Input(shape=input_shape)
    res = ResNet50(weights='imagenet', include_top=False, input_tensor=inp)
    for layer in res.layers:
        layer.trainable = False

    c1 = res.get_layer("conv1_relu").output
    c2 = res.get_layer("conv2_block3_out").output
    c3 = res.get_layer("conv3_block4_out").output
    c4 = res.get_layer("conv4_block6_out").output
    bridge = res.get_layer("conv5_block3_out").output

    u1 = up_block(bridge, c4, 512)
    u2 = up_block(u1, c3, 256)
    u3 = up_block(u2, c2, 128)
    u4 = up_block(u3, c1, 64)

    o = layers.UpSampling2D()(u4)
    o = conv_block(o, 32)
    out = layers.Conv2D(1,1,activation='sigmoid')(o)

    model = models.Model(inp,out)
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss=bce_dice_loss, metrics=METRICS_LIST)
    return model

# ========================
# TRAIN / VALIDATION SPLIT
# ========================
# Split positive images into train/test (15% test)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    positive_images, positive_masks, test_size=0.15, random_state=42
)

# Split train into train/validation (10% of train as val)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.10, random_state=42
)

print(f"Final splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# Compute number of steps per epoch for training/validation generators
steps_train = math.ceil(len(X_train)/BATCH_SIZE)
steps_val   = math.ceil(len(X_val)/BATCH_SIZE)

# ========================
# CALLBACKS
# ========================
# Reduce learning rate when validation loss plateaus
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3,
    min_lr=1e-7, verbose=1
)

# Early stopping to prevent overfitting, restore best weights
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=7,
    restore_best_weights=True, verbose=1
)

my_callbacks = [reduce_lr, early_stop]

# ========================
# TRAINING MODELS
# ========================
trained_models = {}      # dictionary to store trained models
histories = {}           # dictionary to store training history

# --- Base U-Net ---
tf.keras.backend.clear_session()        # clear previous models from memory
print("\n--- Training Base U-Net ---")
model_base = build_unet_base()
hist_base = model_base.fit(
    generator(X_train, y_train, BATCH_SIZE, augment=True, to_3ch=False),
    validation_data=generator(X_val, y_val, BATCH_SIZE, augment=False, to_3ch=False),
    steps_per_epoch=steps_train,
    validation_steps=steps_val,
    epochs=EPOCHS,
    callbacks=my_callbacks,
    verbose=1
)
trained_models["Base"] = model_base
histories["Base"] = hist_base

# --- VGG16 U-Net ---
tf.keras.backend.clear_session()
print("\n--- Training VGG16 U-Net ---")
model_vgg = build_unet_vgg16()
hist_vgg = model_vgg.fit(
    generator(X_train, y_train, BATCH_SIZE, augment=True, to_3ch=True),             # 3 channels for VGG16
    validation_data=generator(X_val, y_val, BATCH_SIZE, augment=False, to_3ch=True),
    steps_per_epoch=steps_train,
    validation_steps=steps_val,
    epochs=EPOCHS,
    callbacks=my_callbacks,
    verbose=1
)
trained_models["VGG16"] = model_vgg
histories["VGG16"] = hist_vgg

# --- ResNet50 U-Net ---
tf.keras.backend.clear_session()
print("\n--- Training ResNet50 U-Net ---")
model_res = build_unet_resnet50()
hist_res = model_res.fit(
    generator(X_train, y_train, BATCH_SIZE, augment=True, to_3ch=True),              # ResNet50 expects 3 channels
    validation_data=generator(X_val, y_val, BATCH_SIZE, augment=False, to_3ch=True),
    steps_per_epoch=steps_train,
    validation_steps=steps_val,
    epochs=EPOCHS,
    callbacks=my_callbacks,
    verbose=1
)
trained_models["ResNet50"] = model_res
histories["ResNet50"] = hist_res

# ========================
# PLOT TRAINING CURVES
# Plot loss and metrics over epochs for each model
# ========================
def plot_final_training_graphs(histories):
    metrics_to_plot = ["loss", "dice_coef", "iou_coef"]
    for name, hist in histories.items():
        plt.figure(figsize=(18,5))
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(1,3,i+1)
            plt.plot(hist.history[metric], label="Train")
            plt.plot(hist.history[f"val_{metric}"], label="Val", linestyle="--")
            plt.title(f"{name} U-Net: {metric} vs Epoch")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()

plot_final_training_graphs(histories)

# ========================
# FINAL DETERMINISTIC EVALUATION
# ========================
# Load and preprocess images into arrays for evaluation
def load_and_preprocess_list(img_paths, mask_paths, to_3ch=False,
                             roi_crop=ROI_CROP, seg_size=SEG_IMG_SIZE):

    X_list, Y_list = [], []

    for ip, mp in zip(img_paths, mask_paths):
        img = tf.io.read_file(ip)
        img = tf.image.decode_png(img, channels=1)

        mask = tf.io.read_file(mp)
        mask = tf.image.decode_png(mask, channels=1)

        img_np = img.numpy()
        mask_np = (mask.numpy() > 0).astype(np.uint8)

        # ROI crop or resize
        if roi_crop:
            img_crop, mask_crop = roi_crop_numpy(img_np, mask_np, crop_size=seg_size)
        else:
            img_crop = tf.image.resize(img, (seg_size, seg_size)).numpy()
            mask_crop = tf.image.resize(mask, (seg_size, seg_size), method='nearest').numpy()

        # Normalize 
        img_f = img_crop.astype(np.float32) / 255.0
        mask_f = (mask_crop.astype(np.float32) > 0.5).astype(np.float32)

        # Ensure channel dimension exists
        if img_f.ndim == 2:
            img_f = np.expand_dims(img_f, -1)
        if mask_f.ndim == 2:
            mask_f = np.expand_dims(mask_f, -1)

        # Convert to 3 channels if needed
        if to_3ch:
            img_f = np.repeat(img_f, 3, axis=-1)

        X_list.append(img_f)
        Y_list.append(mask_f)

    X_arr = np.stack(X_list, axis=0)
    Y_arr = np.stack(Y_list, axis=0)
    return X_arr, Y_arr

# Compute metrics
def compute_numpy_metrics(y_true, y_pred_bin):
    eps = 1e-6
    y_true_f = y_true.reshape(y_true.shape[0], -1)
    y_pred_f = y_pred_bin.reshape(y_pred_bin.shape[0], -1)

    intersection = (y_true_f * y_pred_f).sum(axis=1)
    sum_true = y_true_f.sum(axis=1)
    sum_pred = y_pred_f.sum(axis=1)

    dice = (2 * intersection + eps) / (sum_true + sum_pred + eps)
    union = sum_true + sum_pred - intersection
    iou = (intersection + eps) / (union + eps)

    acc = ((y_true_f == y_pred_f).sum(axis=1) / y_true_f.shape[1]).astype(np.float32)

    tp = intersection
    sens = (tp + eps) / (sum_true + eps)

    tn = ((1 - y_true_f) * (1 - y_pred_f)).sum(axis=1)
    sum_neg = (1 - y_true_f).sum(axis=1)
    spec = (tn + eps) / (sum_neg + eps)

    return {
        "Dice": dice.mean(),
        "IoU": iou.mean(),
        "Accuracy": acc.mean(),
        "Sensitivity": sens.mean(),
        "Specificity": spec.mean()
    }

# Evaluate all trained models
print("\n=== Final Deterministic Evaluation ===")

X_test_1ch, y_test_arr = load_and_preprocess_list(X_test, y_test, to_3ch=False)
X_test_3ch = np.repeat(X_test_1ch, 3, axis=-1)          # duplicate channels for pretrained encoders

for name, model in trained_models.items():
    print(f"\n{name} U-Net:")

    X_in = X_test_1ch if name == "Base" else X_test_3ch

    y_pred = model.predict(X_in, batch_size=BATCH_SIZE)
    y_pred_bin = (y_pred > 0.5).astype(np.float32)

    results = compute_numpy_metrics(y_test_arr, y_pred_bin)

    print("Dice: {:.4f} | IoU: {:.4f} | Accuracy: {:.4f} | Sensitivity: {:.4f} | Specificity: {:.4f}".format(
        results["Dice"], results["IoU"], results["Accuracy"],
        results["Sensitivity"], results["Specificity"]
    ))
