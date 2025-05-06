import os
import cv2
import time
from multiprocessing import Pool
import albumentations as A
import matplotlib.pyplot as plt

time_list = []

def get_images_largest_first(folder_path):
    files_with_size = [
        (f, os.path.getsize(os.path.join(folder_path, f)))
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    files_with_size.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files_with_size]


def get_images_smallest_first(folder_path):
    files_with_size = [
        (f, os.path.getsize(os.path.join(folder_path, f)))
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    files_with_size.sort(key=lambda x: x[1])
    return [f[0] for f in files_with_size]


def augment_image(args):
    folder_path, image_name, output_folder = args
    full_input_path = os.path.join(folder_path, image_name)
    image = cv2.imread(full_input_path)

    if image is None:
        print(f"Warning: Failed to read image {image_name}")
        return

    os.makedirs(output_folder, exist_ok=True)

    new_images = [image]
    count = 0

    transforms = [
        A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ChannelShuffle(p=0.5)
        ]),
        A.Compose([
            A.Rotate(limit=20, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
        ]),
        A.Compose([
            A.GaussNoise(var_limit=(10, 40), p=0.5)
        ]),
        A.Compose([
            A.RandomBrightnessContrast(p=0.5)
        ]),
        A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.Affine(translate_percent=0.1, scale=1.0, rotate=20, p=0.7)
        ]),
        A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.CLAHE(p=0.3),
            A.ToGray(p=0.3)
        ]),
        A.Compose([
            A.OpticalDistortion(distort_limit=1.0, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.5)
        ]),
        A.Compose([
            A.CoarseDropout(max_height=16, max_width=16, min_height=8, min_width=8, p=0.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.4),
            A.RandomShadow(p=0.4),
            A.RandomSunFlare(p=0.3),
            A.Posterize(num_bits=4, p=0.3),
            A.Equalize(p=0.3)
        ])
    ]

    for transform in transforms:
        current_len = len(new_images)
        for j in range(current_len):
            augmented = transform(image=new_images[j])["image"]
            new_images.append(augmented)

    for image in new_images:
        output_name = f"aug_{count}_{image_name}"
        count += 1
        full_output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(full_output_path, image)


def pool_process(args, process_count):
    print(f"\nRunning with {process_count} processes...")
    start = time.time()
    with Pool(process_count) as pool:
        pool.map(augment_image, args)
    end = time.time()
    print(f"Finished in {end - start:.2f} seconds.")
    return round(end - start, 2)


def run_experiment(get_images_func, process_counts, label):
    input_folder = "Input_Images"
    images = get_images_func(input_folder)

    args = [
        (input_folder, img, os.path.join("Augmented_Images", os.path.splitext(img)[0]))
        for img in images
    ]

    times = []
    for p in process_counts:
        times.append(pool_process(args, p))

    plot_data(process_counts, times, label)


def main():
    run_experiment(get_images_largest_first, list(range(1, 17)), "Largest to Smallest - 16 Images")
    run_experiment(get_images_smallest_first, list(range(1, 17)), "Smallest to Largest - 16 Images")


def plot_data(process_count, processes_time, label):
    plt.figure(figsize=(8, 5))
    plt.plot(process_count, processes_time, marker='o', linestyle='-', label=label)
    plt.ylabel('Process Time (seconds)')
    plt.xlabel('Process Count')
    plt.title(f'Image Augmentation: {label}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
