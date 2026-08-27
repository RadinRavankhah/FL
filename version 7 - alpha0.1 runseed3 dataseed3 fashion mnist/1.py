import math
import sys
from PIL import Image

def grid_merge(paths, output="combined.png", padding=10, bg=(255, 255, 255)):
    """Arrange PNGs into a grid to fit all images. 2 columns works well for 7 images,
       but it auto-picks a layout based on smallest aspect-ratio deviation from 1:1."""
    images = [Image.open(p).convert("RGB") for p in paths]

    # Pick rows x cols so the overall grid aspect is closest to square.
    best = None
    for cols in range(1, len(images) + 1):
        rows = math.ceil(len(images) / cols)
        # account for padding when computing aspect ratio
        non_empty_rows = (rows - 1) if (len(images) % cols == 0) else (rows - 1)
        # widths/heights are max among images, quick estimate:
        # we'll just compute final dims below and pick the best after.

    # Easier: try all layouts, measure result, pick the most square.
    best_layout = None
    best_score = float("inf")
    for cols in range(1, len(images) + 1):
        rows = math.ceil(len(images) / cols)
        col_w = max(img.width for img in images)
        row_h = max(img.height for img in images)
        total_w = cols * col_w + (cols + 1) * padding
        total_h = rows * row_h + (rows + 1) * padding
        score = abs(total_w - total_h)  # closeness to square
        if score < best_score:
            best_score = score
            best_layout = (rows, cols, col_w, row_h)

    rows, cols, col_w, row_h = best_layout
    total_w = cols * col_w + (cols + 1) * padding
    total_h = rows * row_h + (rows + 1) * padding

    canvas = Image.new("RGB", (total_w, total_h), bg)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = padding + c * (col_w + padding)
        y = padding + r * (row_h + padding)
        # center the (possibly smaller) image in its cell
        ox = (col_w - img.width) // 2
        oy = (row_h - img.height) // 2
        canvas.paste(img, (x + ox, y + oy))

    canvas.save(output)
    print(f"Saved grid ({rows}x{cols}) to {output}")


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python grid.py img1.png img2.png ... [output.png]")
    #     sys.exit(1)
    # # optional trailing output path
    # if sys.argv[-1].lower().endswith(".png"):
    #     *paths, out = sys.argv[1:]
    # else:
    #     paths, out = sys.argv[1:], "combined.png"
    # grid_merge(paths, out)

    filename_pattern = "50p_random_alpha0.1_runseed3_dataseed3_logs"
    
    
    paths = f"""D:\\Github Repos\\FL\\{filename_pattern}_average_hardware_utility_per_round.png
    D:\\Github Repos\\FL\\{filename_pattern}_client_selection_vs_participation_per_round.png
    D:\\Github Repos\\FL\\{filename_pattern}_device_selection_and_participation_frequency.png
    D:\\Github Repos\\FL\\{filename_pattern}_fairness_per_round_higher_is_better.png
    D:\\Github Repos\\FL\\{filename_pattern}_global_test_accuracy_per_round.png
    D:\\Github Repos\\FL\\{filename_pattern}_mean_client_accuracy_per_round.png
    D:\\Github Repos\\FL\\{filename_pattern}_worst_client_accuracy_per_round.png""".split("\n")
    
    paths = [p.strip() for p in paths if p.strip()]
    
    grid_merge(paths, output="combined_50p_random_seed3.png", padding=10, bg=(255, 255, 255))
    