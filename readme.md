Summary: Run the script, enter your folder path when prompted, and it will identify duplicates (images/videos) and move them to a separate “duplicated_photos” folder; you can adjust the “automatic move”(delete_threshold) and “double-check”(check_threshold) thresholds in the script’s final lines.

![Use Demo Gif](readme/demo.gif)

# Duplicate Image & Video Finder — **README**

This script scans a folder (including its subfolders) for **images** and **videos**, then finds and **moves** duplicates into a separate subfolder rather than deleting them. It uses a combination of:

1. **Content-based similarity**:
    - **Images**: pHash (8×8 DCT)
    - **Videos**: A single-frame pHash (middle frame) plus size/duration info
2. **Exact matching**:
    - If two files have the **exact same file size** but are below the similarity threshold, we do a **SHA-256** check. Identical hashes are treated as **100%** duplicates.
3. **Thresholds**:
    - **`check_threshold`** (default 95%): If two files meet or exceed this similarity, they are considered potential duplicates.
    - **`delete_threshold`** (default 99%): If similarity is above this, the “duplicate” file is automatically moved to the “duplicated_photos” folder.

## Features

-   **Automatic & Manual Modes**:
    -   **Auto Move**: Files with similarity ≥ `delete_threshold` (99% by default) are automatically moved.
    -   **Prompt**: Files with similarity between `check_threshold` (95%) and `delete_threshold` (99%) are shown side by side. You can choose:
        -   `[y]`: Move one file (the “lower-quality” or name-based copy)
        -   `[n]`: Keep both
        -   `[1]`: Stop the entire process immediately
-   **Safer Than Deletion**: Instead of permanently removing duplicates, they are relocated to a dedicated subfolder (`duplicated_photos`).
-   **Name-Based Ties**: If two duplicates have the same “quality” (file size, resolution, etc.), the script keeps the file with the most “original”-looking name (i.e., no “(1)”, “copy”, “복사본”, etc.) and moves the other one.

## Usage

1. **Install Requirements**
    - Python 3
    - Libraries: `opencv-python`, `Pillow`, `matplotlib`
        ```bash
        pip install opencv-python Pillow matplotlib
        ```
2. **Run the Script**
    - In a terminal or Jupyter environment, run:
        ```python
        photo duplicate finder.py or photo duplicate finder.ipynb
        ```
    - The script will ask for the folder path. Enter the directory you want to scan (e.g., `C:\Users\MyUser\Pictures`).
3. **Watch for Prompts**
    - If the script finds duplicates that are below the auto-move threshold but above `check_threshold`, you’ll see a prompt:
        ```
        Options: [y] move one file, [n] keep both, [1] stop now
        ```
    - Make your choice according to what you want to do with the potential duplicate pair.
4. **Check the Duplicates Folder**
    - Duplicates that get moved will appear in `...\duplicated_photos`.
    - You can review or delete them permanently later.

## How It Works

1. **Load Files**
    - Recursively walks through all subfolders, collecting supported **image** and **video** files.
2. **Build Signatures**
    - **Images**: Compute a pHash (a 64-bit perceptual hash).
    - **Videos**: Extract one frame (by default, the middle frame) and compute a pHash, plus store file size and duration.
3. **Compare Files** (Pairwise)
    - Uses the pHash (and metadata for videos) to compute a similarity score in the range **0–100**.
    - If the similarity is below the threshold (95% default), it does a **secondary exact check** if the files are the same byte size (by computing a SHA-256 for each). If those match, treat them as 100% identical.
4. **Decide**
    - **≥ `delete_threshold` (99%)**: Automatically move the lower-quality or “copy-named” file into `duplicated_photos`.
    - **Between 95%–99%**: Prompt the user to confirm.
    - **Below 95%**: Not considered duplicates, so do nothing.
5. **Tally Results**
    - The script tracks how many pairs it checked, how many were auto-moved, how many were moved by user choice, how many pairs were skipped, etc. It prints a summary at the end.

## Tips & Caveats

-   **Back Up** Before Running: Even though the script **moves** instead of deleting, it is still safer to work on a copy of your files.
-   **Large Datasets**: This is an _O(n²)_ approach (checks every pair). If you have tens of thousands of files, it may take a long time.
-   **Video Limitations**: Currently compares only one frame per video, so some near-duplicate videos might not be recognized if that one frame looks different.
-   **Same-Folder Overwrites**: If a file with the same name exists in the “duplicated_photos” folder, the script appends a short random suffix to avoid overwriting.

**Enjoy your safer duplicate management—no more accidental deletions!**
