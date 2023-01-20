import cv2
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from matplotlib.backends.backend_agg import FigureCanvasAgg


img_height = img_width = 380


def _resize_img(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    except:
        print("error in resizing")
        return
    return cv2.resize(img, (img_height, img_width))


def _canny_cropping(img):
    convert_img = np.array(img, dtype=np.uint8)

    gray = cv2.cvtColor(convert_img, cv2.COLOR_RGB2GRAY)

    ave_brightness = math.floor(np.average(gray))
    min_pixel = min(gray.flatten())

    edges = cv2.Canny(gray, min_pixel, ave_brightness)
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(edges)
        gray = gray[y : y + h, x : x + w]
        break

    return gray


def _apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    return clahe.apply(img.astype(np.uint8))


def preprocessing_without_clahe(img):
    cropped = _canny_cropping(img)
    return _resize_img(cropped)


def preprocessing_with_clahe(img):
    cropped = _canny_cropping(img)
    clahe = _apply_clahe(cropped)
    return _resize_img(clahe)


def modified_guided_grad_cam(img, model, last_conv_layer=None):

    # Break down the model to prevent graph connection error
    new_model = _parse_model(model)

    # Try to find convolutional layer if one is not provided
    if last_conv_layer is None:
        for layer in reversed(new_model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer = layer.name
                break
    if last_conv_layer is None:
        raise ValueError("Could not find convolutional layer.")

    grad_model = tf.keras.Model(
        inputs=[new_model.inputs],
        outputs=[new_model.get_layer(last_conv_layer).output, new_model.output],
    )
    # Expand image to match model input shape
    inputs = np.expand_dims(img, axis=0)

    # Get outputs from convolutional and prediction layers
    with tf.GradientTape() as tape:
        (conv_out, pred_out) = grad_model(inputs)
        # Watch convolutional output
        tape.watch(conv_out)
        # Only one predicting class present (binary task)
        pred = pred_out[0]
        print("Prediction output:", pred)

    # Calculate gradients from watched output, get rid of batch dimension
    grads = tape.gradient(pred, conv_out)[0]
    conv_out = conv_out[0]

    # Guided backpropagation - select gradients participating positively to final prediction
    guided_grads = (
        tf.cast(conv_out > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    )

    # Apply average pooling
    pooled_guided_grads = tf.reduce_mean(guided_grads, axis=(0, 1))

    # Multiply gradients with feature map and sum values over all filters
    guided_gradcam = tf.reduce_sum(tf.multiply(pooled_guided_grads, conv_out), axis=-1)

    # Clip the values (equivalent to applying ReLU)
    # and then normalise the values
    guided_gradcam = np.clip(guided_gradcam, 0, np.max(guided_gradcam))
    guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (
        guided_gradcam.max() - guided_gradcam.min()
    )

    # Resize heatmap to match input image size
    guided_gradcam = cv2.resize(guided_gradcam, (img.shape[1], img.shape[0]))

    return guided_gradcam


def guided_grad_cam(img, model, last_conv_layer=None):
    # Try to find convolutional layer if one is not provided
    if last_conv_layer is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer = layer.name
                break
    if last_conv_layer is None:
        raise ValueError("Could not find convolutional layer.")

    # Model with orignal inputs and two outputs, one for the last convolutional layer and the other for prediction
    grad_model = tf.keras.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer).output, model.output],
    )

    # Expand image to match model input shape
    inputs = np.expand_dims(img, axis=0)

    # Get outputs from convolutional and prediction layers
    with tf.GradientTape() as tape:
        (conv_out, pred_out) = grad_model(inputs)
        # Watch convolutional output
        tape.watch(conv_out)
        # Only one predicting class present (binary task)
        pred = pred_out[0]
        print("Prediction output:", pred)

    # Calculate gradients from watched output, get rid of batch dimension
    grads = tape.gradient(pred, conv_out)[0]
    conv_out = conv_out[0]

    # Guided backpropagation - select gradients participating positively to final prediction
    guided_grads = (
        tf.cast(conv_out > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    )

    # Apply average pooling
    pooled_guided_grads = tf.reduce_mean(guided_grads, axis=(0, 1))

    # Multiply gradients with feature map and sum values over all filters
    guided_gradcam = tf.reduce_sum(tf.multiply(pooled_guided_grads, conv_out), axis=-1)

    # Clip the values (equivalent to applying ReLU)
    # and then normalise the values
    guided_gradcam = np.clip(guided_gradcam, 0, np.max(guided_gradcam))
    guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (
        guided_gradcam.max() - guided_gradcam.min()
    )

    # Resize heatmap to match input image size
    guided_gradcam = cv2.resize(guided_gradcam, (img.shape[1], img.shape[0]))

    return impose_heatmap(img, guided_gradcam)


def impose_heatmap(img, heatmap):
    fig, ax = plt.subplots(figsize=(6, 6))
    canvas = FigureCanvasAgg(fig)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.axis("off")
    ax.margins(0)

    ax.imshow(img)
    ax.imshow(heatmap, alpha=0.5)

    canvas.draw()
    buf = canvas.buffer_rgba()
    return np.asarray(buf)


def _parse_model(model):
    submodel_index, submodel = _find_layer(model, "efficientnetv2-b3")
    x = submodel.outputs[0]

    for layer_index in range(submodel_index + 1, len(model.layers)):
        extracted_layer = model.layers[layer_index]

    x = extracted_layer(x)
    new_model = Model(inputs=submodel.inputs, outputs=[x])
    return new_model


def _find_layer(model, layer_name):
    for (i, layer) in enumerate(model.layers):
        if layer.name == layer_name:
            return (i, layer)
