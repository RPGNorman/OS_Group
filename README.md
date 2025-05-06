# OS_Group
# 🖼️ Image Augmentation with Multiprocessing

This project applies advanced image augmentations using Python's `multiprocessing` and the `albumentations` library to compare performance across different process counts. Results are visualized using `matplotlib`.

---

## 📦 Features

- Parallelized image augmentation with configurable process counts
- Robust transformations using `albumentations`
- Automatically saves augmented images to structured directories
- Performance benchmarking and visualization

---

## 🛠️ Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure you're using Python 3.7+.

💻 Setting Up in Visual Studio Code

✅ Step 1: Ensure Git is Installed on Your System
Download Git from: https://git-scm.com/downloads

Install and restart VS Code afterward.

✅ Step 2: Clone the Repository in VS Code
Open Visual Studio Code

Open the Command Palette:
Press Ctrl + Shift + P

Select: Git: Clone

Enter the repository URL:

https://github.com/RPGNorman/OS_Group/
Choose a folder to clone into

When prompted, click Open to open the cloned project in VS Code


## 🧪 How to Use

📁 Directory Structure

OS_Group/

├── Input_Images/           # Put your input images here

├── Augmented_Images/       # Output images will be saved here

├── main.py                 # Main script to run

├── requirements.txt        # Python dependencies

└── README.md               # This file


## 🚀 Running the Script

Place your original images inside the Input_Images/ folder.

Open a terminal in VS Code (Ctrl + `)

Run the script:

```bash
python main.py
```
The script will:

Run two experiments: one sorting images from largest to smallest, and one from smallest to largest

Augment images with multiple transformations

Save outputs in Augmented_Images/<original_image_name>/

Plot and display processing time per process count


## 📊 Output Example

You’ll see a plot like this after execution:

X-axis: Number of processes (1 to 16)

Y-axis: Time taken in seconds

Curves: Time for different image orderings


## ⚠️ Notes

If an image fails to load, a warning will print but the script will continue.

All augmented images are saved with a prefix like aug_0_, aug_1_, etc.

You can adjust transformations in the transforms list in main.py.

You can alter the number of images transformed by manually adding or removing images.

## 🧹 Cleanup

To start fresh:

Delete all contents inside Augmented_Images/

Ensure only the raw inputs remain in Input_Images/
