import cv2
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────
# Option 1: Most common way (using OpenCV + matplotlib)
# ────────────────────────────────────────────────


def plot_rgb_channels(image_path):
    # Read image in BGR format (default in OpenCV)
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert BGR → RGB for correct color display in matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Split into R, G, B channels
    r, g, b = cv2.split(img_rgb)

    # Create zero arrays for displaying single channels in color
    zeros = np.zeros_like(r)

    # Create images showing only one channel
    red_channel = cv2.merge([zeros, zeros, r])  # only red
    green_channel = cv2.merge([zeros, g, zeros])  # only green
    blue_channel = cv2.merge([b, zeros, zeros])  # only blue

    # Plotting
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(red_channel)
    plt.title("Red Channel")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(green_channel)
    plt.title("Green Channel")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(blue_channel)
    plt.title("Blue Channel")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Optional: show grayscale intensity version of each channel
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(r, cmap="gray")
    plt.title("Red (grayscale)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap="gray")
    plt.title("Green (grayscale)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(b, cmap="gray")
    plt.title("Blue (grayscale)")
    plt.axis("off")

    plt.suptitle("Channel Intensity (grayscale view)", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_default_gray_stackR(image_path: str) -> None:
    img_bgr = cv2.imread(image_path)
    # Convert BGR → RGB for correct color display in matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Split into R, G, B channels
    r, g, b = cv2.split(img_rgb)
    # These look very different:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(r)  # ← viridis (colorful!)
    plt.title("plt.imshow(r)\n(default = viridis colormap)")

    plt.subplot(1, 3, 2)
    plt.imshow(r, cmap="gray")  # ← now proper grayscale
    plt.title("plt.imshow(r, cmap='gray')")

    plt.subplot(1, 3, 3)
    plt.imshow(np.stack([r, r, r], axis=-1))  # or cv2.merge([r,r,r])
    plt.title("Duplicate red into RGB\n(true red shades)")
