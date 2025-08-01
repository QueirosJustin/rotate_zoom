import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time

def rotate(img, angle):
    """Rotate and zoom so no black corners remain."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    # convert angle to radians
    angle_rad = math.radians(abs(angle))

    # trig helpers
    sin_a = abs(math.sin(angle_rad))
    cos_a = abs(math.cos(angle_rad))

    # bounds after rotation
    new_w = w * cos_a + h * sin_a
    new_h = h * cos_a + w * sin_a

    # compute zoom factor
    scale_x = new_w / w
    scale_y = new_h / h
    scale = max(scale_x, scale_y)

    # rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # apply warp
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

# ---- main loop ----
if __name__ == "__main__":
    # Load once
    img = cv2.imread("test_img.jpg")
    if img is None:
        raise FileNotFoundError("test_img.jpg not found")

    # set up matplotlib interactive mode
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    step_size = 1
    for angle in np.arange(0, 360.1, step_size):
        rotated_img = rotate(img, angle)

        # convert from BGR (OpenCV) to RGB (Matplotlib)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rotated_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)

        # clear axes
        for ax in axes:
            ax.clear()

        # show original
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # show rotated
        axes[1].imshow(rotated_rgb)
        axes[1].set_title(f"Rotated {angle:.1f}°")  # formatted to 1 decimal
        axes[1].axis("off")

        plt.tight_layout()
        plt.pause(0.001)  # allows live update

    plt.ioff()
    plt.show()
