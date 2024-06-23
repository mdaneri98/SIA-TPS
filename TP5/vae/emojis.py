import numpy as np
from PIL import Image

emojis_name = [
    "emoji1",
    "emoji2",
    "emoji3",
    "emoji4",
    "emoji5",
    "emoji6",
    "emoji7",
    "emoji8"
]

emojis_size = (20, 20)  # New size 20x20
emojis_images = []

def load_emojis_images():
    img = np.asarray(Image.open('emojis.png').convert("L"))
    emojis_per_row = img.shape[1] // 16  # Original size is 16x16

    for i in range(len(emojis_name)):
        y = (i // emojis_per_row) * 16  # Adjust for original size 16x16
        x = (i % emojis_per_row) * 16  # Adjust for original size 16x16
        emoji_matrix = img[y:(y + 16), x:(x + 16)] / 255.0

        emoji_image = Image.fromarray((emoji_matrix * 255).astype(np.uint8)).resize(emojis_size, Image.Resampling.LANCZOS)
        emoji_matrix_resized = np.asarray(emoji_image) / 255.0

        emojis_vector = emoji_matrix_resized.flatten()
        emojis_images.append(emojis_vector)


load_emojis_images()


