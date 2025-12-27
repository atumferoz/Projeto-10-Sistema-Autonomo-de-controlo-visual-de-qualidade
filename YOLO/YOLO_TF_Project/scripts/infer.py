import tensorflow as tf

model = tf.saved_model.load("/workspace/model")
infer = model.signatures["serving_default"]

print("INPUT:", infer.structured_input_signature)
print("OUTPUT:", infer.structured_outputs)
