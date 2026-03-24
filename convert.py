import tensorflow as tf

# 1. I-set ang pangalan ng iyong model files
h5_model_path = "coconut_model_v2_ultra.h5"
tflite_model_path = "coconut_model.tflite"

try:
    print(f"⏳ Loading {h5_model_path}...")
    # I-load ang .h5 model (compile=False para hindi mag-error sa custom optimizers)
    model = tf.keras.models.load_model(h5_model_path, compile=False)

    print("🔄 Converting to TFLite format...")
    # I-initialize ang TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # OPTIONAL: I-optimize ang size (Quantization)
    # Binabawasan nito ang size ng model nang hanggang 75%
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Gawin ang conversion
    tflite_model = converter.convert()

    # 2. I-save ang pinalit na file
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Success! Ang iyong bagong model ay: {tflite_model_path}")

except Exception as e:
    print(f"❌ Error during conversion: {e}")