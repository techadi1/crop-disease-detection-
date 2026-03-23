# main.py — Apple Leaf Disease Detection (Best Version)
# Trains with augmentation + EarlyStopping, saves best model (.keras),
# exports float16 TFLite, and generates all plots/reports.

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ========= 0) Paths & basics =========
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"
IMG_H, IMG_W = 128, 128
BATCH = 32
EPOCHS = 15

# ========= 1) Load datasets =========
# val_ds shuffle=False to keep label order stable for evaluation
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=(IMG_H, IMG_W), batch_size=BATCH, shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=(IMG_H, IMG_W), batch_size=BATCH, shuffle=False
)
class_names = train_ds.class_names
print("Class Names:", class_names)

# Prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# ========= 2) Data Augmentation =========
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

# Apply augmentation only on training stream
aug_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# ========= 3) Build model =========
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_H, IMG_W, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128,3, activation='relu'), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
], name="apple_cnn")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ========= 4) Callbacks =========
early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                         restore_best_weights=True)
ckpt  = tf.keras.callbacks.ModelCheckpoint("best_model.keras",
                                           monitor="val_accuracy",
                                           save_best_only=True)

# ========= 5) Train =========
history = model.fit(
    aug_train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early, ckpt],
    verbose=1
)

# Load best model weights and save final export
best = tf.keras.models.load_model("best_model.keras")
val_loss, val_acc = best.evaluate(val_ds, verbose=0)
print(f"\nBest Model → Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss:.3f}")

# ========= 6) Plots (saved) =========
# Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
plt.savefig("accuracy.png", dpi=150, bbox_inches="tight")
plt.show()

# Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
plt.savefig("loss.png", dpi=150, bbox_inches="tight")
plt.show()
# ========= 7) Confusion Matrix & Report (saved) =========
# Predict all val batches
y_true, y_pred = [], []
for images, labels in val_ds:
    probs = best.predict(images, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1))
    y_true.extend(labels.numpy().tolist())
y_true = np.array(y_true); y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
report_text = classification_report(y_true, y_pred, target_names=class_names)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report_text)

# Pretty CM
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix"); plt.colorbar()
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45, ha='right')
plt.yticks(ticks, class_names)
thr = cm.max()/2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > thr else 'black')
plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

with open("classification_report.txt", "w") as f:
    f.write(report_text)

# ========= 8) Sample predictions grid (saved) =========
for images, labels in val_ds.take(1):
    probs = best.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    n = min(8, images.shape[0])
    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[int(labels[i])]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout(); plt.savefig("sample_preds.png", dpi=150, bbox_inches="tight"); plt.show()
    break

# ========= 9) Save Keras model =========
best.save("apple_leaf_model.keras")
print("Saved: apple_leaf_model.keras")

# ========= 10) Export TFLite (float16) =========
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(best)
    # Option B: float16 quantization (your choice)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open("apple_leaf_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Saved: apple_leaf_model.tflite (float16)")
except Exception as e:
    print("TFLite export failed:", e)

# ========= 11) Single-image test (edit path) =========
IMG_PATH = r"C:\Users\Navaneeth Renati\Downloads\AppleLeafAI\dataset\val\Apple___healthy\0c55b379-c6e7-4b89-959f-abc506fed437___RS_HL 5927.JPG"

try:
    img = tf.keras.utils.load_img(IMG_PATH, target_size=(IMG_H, IMG_W))
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)  # batch
    probs = best.predict(arr, verbose=0)[0]
    pred = class_names[int(np.argmax(probs))]
    print("\nSingle-image prediction:", pred)
except Exception as e:
    print("\nSingle-image test skipped:", e)
