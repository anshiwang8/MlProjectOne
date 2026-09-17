import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("models/trash_classifier.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(r"C:\Users\anshi\Desktop\garbage sorting\data\Raw Data\Testing Dataset", image_size=(224, 224), batch_size = 32)

test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

x, y = model.evaluate(test_ds) #evalutates function
print ("Test Loss: ", x)
print ("Test Accuracy: ", y)