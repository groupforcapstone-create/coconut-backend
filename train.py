# ==========================================
# PHASE 1: ENVIRONMENT SETUP & IMPORTS
# ==========================================
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, applications
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile

print("✅ Phase 1: Libraries imported successfully.")

# ==========================================
# PHASE 2: LOCAL PATHS & CONFIGURATION
# ==========================================
BASE_DIR         = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATASET_PATH     = os.path.join(BASE_DIR, "Dataset6")
OUTPUT_DIR       = os.path.join(BASE_DIR, "Coconut_Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# CONFIGURATION SETTINGS (50 TOTAL EPOCHS | NO EARLY STOPPING)
# -------------------------------------------------------------
USE_MOBILENET_V3 = False 

IMG_SIZE         = (160, 160) 
BATCH_SIZE       = 32          
EPOCHS_HEAD      = 15  # Phase 1 Warmup Duration        
EPOCHS_FINE      = 35  # Phase 2 Fine-Tuning Duration (Total = 50 Epochs)       
SEED             = 42

FINE_TUNE_AT     = 120 if USE_MOBILENET_V3 else 100 

MODEL_SAVE_PATH  = os.path.join(OUTPUT_DIR, "coconut_model_best.keras")
TFLITE_SAVE_PATH = os.path.join(OUTPUT_DIR, "coconut_variety_model.tflite")
CURVES_SAVE_PATH = os.path.join(OUTPUT_DIR, "coconuts_performance.png")
CM_SAVE_PATH     = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
LABELS_SAVE_PATH = os.path.join(OUTPUT_DIR, "coconut_labels.txt")

print(f"✅ Phase 2: Configuration set. Target Backbone: {'MobileNetV3Small' if USE_MOBILENET_V3 else 'MobileNetV2'}")

# ==========================================
# PHASE 3: DIRECTML LOCAL HARDWARE VERIFICATION
# ==========================================
print("\n=== SYSTEM HARDWARE CHECK ===")
print("TF version :", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
try:
    import tensorflow_directml_plugin
    dml_active = True
except ImportError:
    dml_active = False

if not gpus and not dml_active:
    print("ℹ️ Note: No local GPU detected. Pipeline will process via CPU hardware.")
else:
    print("✅ GPU Acceleration Active via Windows DirectML/NVIDIA Layer!")

# ==========================================
# PHASE 4: LOCAL DIRECTORY SYNCHRONIZATION
# ==========================================
if not os.path.exists(DATASET_PATH):
    print(f"🚨 ERROR: Dataset folder not found at: {DATASET_PATH}")
    print("Please make sure your 'Dataset6' folder is placed in the same folder as this script.")
    sys.exit(1)

OLD_NONERICE_PATH = os.path.join(DATASET_PATH, "Nonerice")
NEW_NOTCOCONUT_PATH = os.path.join(DATASET_PATH, "NotCoconut")

if os.path.exists(OLD_NONERICE_PATH):
    try:
        os.rename(OLD_NONERICE_PATH, NEW_NOTCOCONUT_PATH)
        print("🔄 Mapping active: Renamed local 'Nonerice' folder to 'NotCoconut'.")
    except Exception as e:
        print(f"⚠️ Notification: Folder rename bypassed ({e}).")
else:
    print("ℹ️ Check: 'Nonerice' folder already mapped or named as 'NotCoconut'.")

# ==========================================
# PHASE 5: PRE-TRAINING SCAN — DATASET CLEANUP
# ==========================================
print("\n🔍 Scanning local dataset directories for corrupt files...")
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None 

removed_count = 0
for root, dirs, files in os.walk(DATASET_PATH):
    for fname in files:
        if fname.startswith('.') or not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')): 
            continue
        fpath = os.path.join(root, fname)
        try:
            with Image.open(fpath) as img:
                img.load()
        except Exception:
            try:
                os.remove(fpath)
                removed_count += 1
            except Exception:
                pass
print(f"✅ Phase 5: Scan complete. Isolated/Removed {removed_count} unreadable asset files.")

# ==========================================
# PHASE 6: DATASET PIPELINE LOADING
# ==========================================
print("\n📊 Loading local images into optimized Keras tensors...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.3,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

temp_val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.3,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

CLASS_NAMES = train_dataset.class_names
NUM_CLASSES = len(CLASS_NAMES)
print("✅ Phase 6: Local data loader active. Target classes:", CLASS_NAMES)

# ==========================================
# PHASE 7: DATASET DIVISION (VALIDATION / TEST SPLIT)
# ==========================================
val_batches_cardinality = tf.data.experimental.cardinality(temp_val_dataset)

if val_batches_cardinality == tf.data.UNKNOWN_CARDINALITY:
    val_batches = 0
else:
    val_batches = val_batches_cardinality.numpy()

if val_batches <= 1:
    test_dataset = temp_val_dataset
    val_dataset = temp_val_dataset
else:
    test_batches_count = val_batches // 2
    test_dataset = temp_val_dataset.take(test_batches_count)
    val_dataset = temp_val_dataset.skip(test_batches_count)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
print(f"✅ Phase 7: Verification data split processed into distinct Validation and Test segments.")

# ==========================================
# PHASE 8: AUTOMATED CLASS WEIGHT COMPUTATION
# ==========================================
print("\nBalancing data weight parameters across uneven variety counts...")
y_list = []
for _, labels in train_dataset:
    y_list.extend(labels.numpy())
y_train = np.array(y_list)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print("✅ Phase 8: Class imbalance coefficient calculations completed.")

# ==========================================
# PHASE 9: BUILD PRE-TRAINED BACKBONE NETWORK
# ==========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1), 
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
], name="data_augmentation")

if USE_MOBILENET_V3:
    base_model = applications.MobileNetV3Small(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
else:
    base_model = applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')

base_model.trainable = False  

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./127.5, offset=-1)(x) 
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x) 

outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = tf.keras.Model(inputs, outputs)
print("✅ Phase 9: Network topology architecture prepared.")

# ==========================================
# PHASE 10: PHASE 1 — HEAD WARMUP (15 STRICT EPOCHS)
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# EarlyStopping completely removed to ensure all 15 epochs execute
callbacks_phase1 = []

print(f"\n🚀 PHASE 1: Running classification warmup loop (Epochs: {EPOCHS_HEAD} | LR=0.0001)...")
history_phase1 = model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=EPOCHS_HEAD, 
    callbacks=callbacks_phase1,
    class_weight=class_weight_dict
)
print("✅ Phase 10: Top classifier layers warmed up.")

# ==========================================
# PHASE 11: PHASE 2 — FINE-TUNING (35 STRICT EPOCHS)
# ==========================================
print("\n🧠 Unfreezing deep structural features for weight fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# EarlyStopping removed here as well. Model saves best validation weights naturally.
callbacks_phase2 = [
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001, verbose=1),
]

print(f"\n🚀 PHASE 2: Launching deep variety calibration (Epochs: {EPOCHS_FINE} | Total Combined Target = 50)...")
history_phase2 = model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=(EPOCHS_HEAD + EPOCHS_FINE), 
    initial_epoch=EPOCHS_HEAD, 
    callbacks=callbacks_phase2, 
    class_weight=class_weight_dict
)
print("✅ Phase 11: Deep training execution loop completed.")

# ==========================================
# PHASE 12: UNBIASED TEST EVALUATION
# ==========================================
print("\n📊 Computing statistical score validations on clean test images...")
test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
print(f"\n✅ Phase 12 Complete. Absolute General Accuracy Score: {test_acc * 100:.2f}%")

# ==========================================
# PHASE 13: CHART GENERATION — PROGRESSION CURVES
# ==========================================
print("\n📈 Exporting metric training progression curves...")
acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
loss = history_phase1.history['loss'] + history_phase2.history['loss']
val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(acc, label='Train Accuracy', color='#1f77b4', linewidth=2)
ax1.plot(val_acc, label='Validation Accuracy', color='#ff7f0e', linestyle='--', linewidth=2)
ax1.axvline(x=EPOCHS_HEAD - 1, color='gray', linestyle=':', label='Fine-Tuning Threshold')
ax1.set_title('Accuracy Progression Profile', fontsize=12, fontweight='bold')
ax1.set_xlabel('Combined Operational Epochs')
ax1.set_ylabel('Accuracy Value Scale')
ax1.legend(loc='lower right')

ax2.plot(loss, label='Train Loss', color='#d62728', linewidth=2)
ax2.plot(val_loss, label='Validation Loss', color='#2ca02c', linestyle='--', linewidth=2)
ax2.axvline(x=EPOCHS_HEAD - 1, color='gray', linestyle=':', label='Fine-Tuning Threshold')
ax2.set_title('Loss Entropy Profile Across Lifespan', fontsize=12, fontweight='bold')
ax2.set_xlabel('Combined Operational Epochs')
ax2.set_ylabel('Loss Value Scale')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(CURVES_SAVE_PATH, dpi=300)
print(f"✅ Phase 13: Technical metrics plot saved to: {CURVES_SAVE_PATH}")
plt.show() 

# ==========================================
# PHASE 14: ANALYTICS — CONFUSION MATRIX VISUALIZATION
# ==========================================
print("\n🗺️ Formatting classification matrix metrics...")
y_true, y_pred = [], []
for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\n=== COMPLETE STRUCTURAL CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES))

plt.figure(figsize=(9, 7))
sns.heatmap(
    confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))), 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=CLASS_NAMES, 
    yticklabels=CLASS_NAMES
)
plt.title('Analytical Predictions Confusion Matrix Alignment', fontsize=12, fontweight='bold')
plt.ylabel('Actual Category Label')
plt.xlabel('Predicted System Category Assignation')
plt.tight_layout()
plt.savefig(CM_SAVE_PATH, dpi=300)
print(f"✅ Phase 14: Confusion matrix heatmap saved to: {CM_SAVE_PATH}")
plt.show() 

# ==========================================
# PHASE 15: INT8 QUANTIZATION EXPORT PROTOCOL (TFLITE)
# ==========================================
print("\n📦 Initializing hardware deployment compiler pipeline for Full INT8...")
def rep_data_gen():
    for images, _ in test_dataset.take(5):
        for img in images: 
            yield [tf.expand_dims(tf.cast(img, tf.float32), axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

with open(TFLITE_SAVE_PATH, 'wb') as f:
    f.write(converter.convert())

with open(LABELS_SAVE_PATH, "w") as f:
    for label in CLASS_NAMES: 
        f.write(label + "\n")

print(f"\n🎉 PHASE 15: COMPILATION SUCCESSFUL! All assets compiled locally inside:\n📁 {OUTPUT_DIR}")