from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model



with open('model_a1.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

model.load_weights('model_weights1.weights.h5')
print("Model loaded successfully.")

img_path = 'surprise-img.jpeg'  
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48))
img = img / 255.0  # Normalize the image
img = np.expand_dims(img, axis=-1)  # Add channel dimension (48, 48, 1)
img = np.expand_dims(img, axis=0)   # Add batch dimension (1, 48, 48, 1)


layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]

activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img)

def display_activation_maps(activations, col_size=8, row_size=8):
    layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]
    
    for layer_activation, layer_name in zip(activations, layer_names):
        num_filters = layer_activation.shape[-1]
        
        size = layer_activation.shape[1]
        num_cols = col_size if num_filters >= col_size else num_filters
        num_rows = (num_filters // col_size) if num_filters >= col_size else 1
        
        # initialize the figure for display
        display_grid = np.zeros((size * num_rows, size * num_cols))

        for i in range(num_rows):
            for j in range(num_cols):
                filter_image = layer_activation[0, :, :, i * num_cols + j]

                # process the feature to make it visually understandable
                filter_image -= filter_image.mean()
                filter_image /= filter_image.std() + 1e-5
                filter_image *= 64
                filter_image += 128
                filter_image = np.clip(filter_image, 0, 255).astype('uint8')

                # place each filter's image on the grid
                display_grid[i * size: (i + 1) * size,
                             j * size: (j + 1) * size] = filter_image

        # display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

display_activation_maps(activations)