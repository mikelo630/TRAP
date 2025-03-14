import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

def load_geojson(geojson_path):
    try:
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"GeoJSON file not found at {geojson_path}")
        gdf = gpd.read_file(geojson_path)
        print("GeoJSON columns:", gdf.columns.tolist())
        required_cols = ['class_id', 'class_name', 'tile_id', 'geometry']
        missing_cols = [col for col in required_cols if col not in gdf.columns]
        if missing_cols:
            raise ValueError(f"GeoJSON missing required columns: {missing_cols}")
        return gdf
    except Exception as e:
        raise Exception(f"Error loading GeoJSON: {str(e)}")

def create_mask(geometry, class_id, img_size=(256, 256)):
    mask = np.zeros(img_size, dtype=np.uint8)
    shapes = [(geom, class_id) for geom in geometry if geom is not None]
    if not shapes:
        return mask
    mask = rasterize(shapes, out_shape=img_size, fill=0, default_value=class_id)
    return mask

def resize_image(image, target_size=(256, 256)):
    """Resize image to target size while maintaining aspect ratio and cropping/padding if needed."""
    img = Image.fromarray(np.uint8(image))
    img = img.resize(target_size, Image.Resampling.LANCZOS)  # High-quality resizing
    return np.array(img)

def load_data(tile_path, geojson_path):
    gdf = load_geojson(geojson_path)

    images = []
    masks = []

    tile_files = [f for f in os.listdir(tile_path) if f.endswith('.tif')]
    if not tile_files:
        raise ValueError(f"No PNG files found in {tile_path}")

    print(f"Found {len(tile_files)} tile images")

    for tile_file in tile_files:
        tile_id = os.path.splitext(tile_file)[0]

        try:
            with rasterio.open(os.path.join(tile_path, tile_file)) as src:
                img = src.read()
                img = np.moveaxis(img, 0, -1)  # CHW to HWC
                if img.shape[-1] == 4:
                    img = img[..., :3]  # Remove alpha channel
                # Resize to 256x256
                if img.shape[:2] != (256, 256):
                    print(f"Resizing {tile_file} from {img.shape[:2]} to (256, 256)")
                    img = resize_image(img)
                if img.shape != (256, 256, 3):
                    raise ValueError(f"After resizing, {tile_file} has shape {img.shape}")
        except Exception as e:
            print(f"Skipping {tile_file}: Error loading/resizing image - {str(e)}")
            continue

        tile_gdf = gdf[gdf['tile_id'] == tile_id]
        if tile_gdf.empty:
            print(f"Warning: No annotations for {tile_id}, using empty mask")
            mask = np.zeros((256, 256), dtype=np.uint8)
        else:
            # Create mask
            mask = np.zeros((256, 256), dtype=np.uint8)
            for _, row in tile_gdf.iterrows():
                class_id = int(row['class_id'])
                geom = row['geometry']
                try:
                    temp_mask = create_mask([geom], class_id)
                    if temp_mask.shape != (256, 256):
                        print(f"Warning: Mask for {tile_id} has shape {temp_mask.shape}")
                        continue
                    mask = np.maximum(mask, temp_mask)
                except Exception as e:
                    print(f"Warning: Error creating mask for {tile_id}: {str(e)}")
                    continue

        images.append(img)
        masks.append(mask)
        print(f"Processed {tile_file}: Image shape {img.shape}, Mask shape {mask.shape}")

    if not images:
        raise ValueError("No valid image-mask pairs were loaded")

    images_array = np.array(images)
    masks_array = np.array(masks)
    print(f"Final images array shape: {images_array.shape}")
    print(f"Final masks array shape: {masks_array.shape}")
    return images_array, masks_array

def unet_model(input_shape=(256, 256, 3), n_classes=3):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)

    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(n_classes, 1, activation='softmax')(c7)

    return Model(inputs, outputs)

try:
    images, masks = load_data(TILE_PATH, GEOJSON_PATH)
    print(f"Loaded {len(images)} image-mask pairs")
except Exception as e:
    print(f"Failed to load data: {str(e)}")
    raise

images = images / 255.0

n_classes = 3  # Background (0), plastic (1), veg (2)
masks_one_hot = tf.keras.utils.to_categorical(masks, num_classes=n_classes)

X_train, X_val, y_train, y_val = train_test_split(
    images, masks_one_hot, test_size=0.2, random_state=42
)

model = unet_model()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

BATCH_SIZE = 8
EPOCHS = 50

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    verbose=1
)

model.save(MODEL_PATH)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

def predict_mask(model, image):
    pred = model.predict(np.expand_dims(image, axis=0))
    return np.argmax(pred[0], axis=-1)

sample_idx = 0
pred_mask = predict_mask(model, X_val[sample_idx])

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(X_val[sample_idx])
plt.title('Input Image')
plt.subplot(132)
plt.imshow(np.argmax(y_val[sample_idx], axis=-1))
plt.title('Ground Truth')
plt.subplot(133)
plt.imshow(pred_mask)
plt.title('Prediction')
plt.show()

from tensorflow.keras.models import load_model

model = load_model(MODEL_PATH)
print("Model loaded successfully.")

def load_tile(tile_id, tile_path):
    tile_file = f"{tile_id}.tif"
    with rasterio.open(os.path.join(tile_path, tile_file)) as src:
        img = src.read()
        img = np.moveaxis(img, 0, -1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape != (256, 256, 3):
            print(f"Resizing {tile_file} from {img.shape[:2]} to (256, 256)")
            img = resize_image(img)
    return img / 255.0  # Normalize

def resize_image(image, target_size=(256, 256)):
    img = Image.fromarray(np.uint8(image))
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img)

def load_ground_truth(tile_id, geojson_path):
    gdf = gpd.read_file(geojson_path)
    tile_gdf = gdf[gdf['tile_id'] == tile_id]
    mask = np.zeros((256, 256), dtype=np.uint8)
    for _, row in tile_gdf.iterrows():
        class_id = int(row['class_id'])
        geom = row['geometry']
        try:
            temp_mask = create_mask([geom], class_id)
            mask = np.maximum(mask, temp_mask)
        except Exception as e:
            print(f"Warning: Error creating mask for {tile_id}: {str(e)}")
            continue
    return mask

def create_mask(geometry, class_id, img_size=(256, 256)):
    mask = np.zeros(img_size, dtype=np.uint8)
    shapes = [(geom, class_id) for geom in geometry if geom is not None]
    if not shapes:
        return mask
    mask = rasterize(shapes, out_shape=img_size, fill=0, default_value=class_id)
    return mask

def predict_mask(model, image):
    pred = model.predict(np.expand_dims(image, axis=0))
    return np.argmax(pred[0], axis=-1)

target_tile = 'TO_61'

input_image = load_tile(target_tile, TILE_PATH)
ground_truth = load_ground_truth(target_tile, GEOJSON_PATH)
pred_mask = predict_mask(model, input_image)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(input_image)
plt.title('Input Image')
plt.subplot(132)
plt.imshow(ground_truth)
plt.title('Ground Truth')
plt.subplot(133)
plt.imshow(pred_mask)
plt.title('Prediction')
plt.show()

print(f"Visualizing tile {target_tile}")
