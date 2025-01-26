import os
import cv2
import numpy as np
import random
import hashlib
from PIL import Image
from IPython.display import clear_output
import matplotlib.pyplot as plt
import shutil
import uuid  # for random unique suffix if needed

###############################################################################
# 1) LOADING FILES (이미지 + 동영상)
###############################################################################

def load_files(folder_path):
    """
    Recursively loads all image and video file paths from a folder (and subfolders).
    """
    supported_images = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    supported_videos = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm')
    
    file_paths = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            lower_file = file.lower()
            if lower_file.endswith(supported_images) or lower_file.endswith(supported_videos):
                file_paths.append(os.path.join(root, file))
    
    return file_paths

def is_video_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm')
    return ext in video_exts

def is_image_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    return ext in image_exts

###############################################################################
# 2) IMAGE pHash & VIDEO SIGNATURE
###############################################################################

def compute_image_phash(image_path):
    """
    Computes a perceptual hash (pHash) for an image using OpenCV.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(image))
        dct_roi = dct[0:8, 0:8]
        median_val = np.median(dct_roi)
        phash = ''.join('1' if px > median_val else '0'
                        for row in dct_roi for px in row)
        return phash
    except:
        return None

def compute_video_signature(video_path):
    """
    A naive signature for videos:
      - file size (bytes)
      - duration (seconds)
      - pHash of the middle frame
    """
    try:
        file_size = os.path.getsize(video_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if frame_count > 0 and fps > 0:
            duration = frame_count / fps
        else:
            duration = 0
        
        # Middle frame
        mid_index = int(frame_count // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_index)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            # fallback: first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        cap.release()
        
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(np.float32(gray))
            dct_roi = dct[0:8, 0:8]
            median_val = np.median(dct_roi)
            phash = ''.join('1' if px > median_val else '0'
                            for row in dct_roi for px in row)
        else:
            phash = None
        
        return {
            'size': file_size,
            'duration': duration,
            'frame_phash': phash
        }
    except:
        return None

def compute_file_signature(filepath):
    """
    For images, returns { 'type': 'image', 'hash': <pHash>, 'size': <bytes> }
    For videos, returns { 'type': 'video', 'size': <bytes>, 'duration': <float>, 'frame_phash': <hash> }
    """
    if is_image_file(filepath):
        phash = compute_image_phash(filepath)
        size = os.path.getsize(filepath)
        return {
            'type': 'image',
            'hash': phash,
            'size': size
        }
    elif is_video_file(filepath):
        vinfo = compute_video_signature(filepath)
        if vinfo is None:
            return None
        return {
            'type': 'video',
            'size': vinfo['size'],
            'duration': vinfo['duration'],
            'frame_phash': vinfo['frame_phash']
        }
    else:
        return None

###############################################################################
# 3) SHA-256 CHECK & SIMILARITY CALC
###############################################################################

def hamming_distance(h1, h2):
    if not h1 or not h2:
        return 64
    return sum(ch1 != ch2 for ch1, ch2 in zip(h1, h2))

def compute_sha256(file_path, chunk_size=65536):
    """
    Computes SHA-256 hash for the file to check if bit-for-bit identical.
    """
    sha = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                sha.update(data)
        return sha.hexdigest()
    except:
        return None

def get_similarity(sig1, sig2, path1, path2, check_threshold):
    """
    1) If same type (image-image or video-video), do a pHash-based (or video-based) comparison
    2) If < check_threshold but file sizes are identical => do SHA-256 check => if identical, treat as 100%.
    3) If different types => 0%
    """
    if not sig1 or not sig2:
        return 0.0
    
    if sig1['type'] != sig2['type']:
        return 0.0
    
    # Image vs Image
    if sig1['type'] == 'image' and sig2['type'] == 'image':
        dist = hamming_distance(sig1['hash'], sig2['hash'])
        phash_sim = (64 - dist) / 64 * 100
        if phash_sim >= check_threshold:
            return phash_sim
        else:
            # fallback: file size same => check SHA
            if sig1['size'] == sig2['size']:
                sha1 = compute_sha256(path1)
                sha2 = compute_sha256(path2)
                if sha1 and sha2 and sha1 == sha2:
                    return 100.0
            return phash_sim
    
    # Video vs Video
    if sig1['type'] == 'video' and sig2['type'] == 'video':
        ph1 = sig1.get('frame_phash')
        ph2 = sig2.get('frame_phash')
        if ph1 and ph2:
            dist = hamming_distance(ph1, ph2)
            frame_sim = (64 - dist) / 64 * 100
        else:
            frame_sim = 0
        
        size1, size2 = sig1['size'], sig2['size']
        dur1, dur2 = sig1.get('duration', 0), sig2.get('duration', 0)
        
        # size similarity
        if size1 == 0 or size2 == 0:
            size_similarity = 0
        else:
            size_diff = abs(size1 - size2)
            max_size = max(size1, size2)
            size_penalty = (size_diff / max_size) * 100
            size_similarity = 100 - size_penalty
            if size_similarity < 0:
                size_similarity = 0
        
        # duration similarity
        if dur1 == 0 or dur2 == 0:
            dur_similarity = 0
        else:
            dur_diff = abs(dur1 - dur2)
            max_dur = max(dur1, dur2)
            dur_penalty = (dur_diff / max_dur) * 100
            dur_similarity = 100 - dur_penalty
            if dur_similarity < 0:
                dur_similarity = 0
        
        combined_sim = (frame_sim + size_similarity + dur_similarity) / 3
        
        if combined_sim >= check_threshold:
            return combined_sim
        else:
            if size1 == size2 and size1 != 0:
                sha1 = compute_sha256(path1)
                sha2 = compute_sha256(path2)
                if sha1 and sha2 and sha1 == sha2:
                    return 100.0
            return combined_sim
    
    return 0.0

###############################################################################
# 4) QUALITY + TIE-BREAK (NAME POLICY)
###############################################################################

def get_file_quality(sig):
    """
    Images => file size
    Videos => file_size + 1000 * duration
    """
    if not sig:
        return 0
    if sig['type'] == 'image':
        return sig['size']
    elif sig['type'] == 'video':
        return sig['size'] + 1000 * sig.get('duration', 0)
    return 0

def is_original_name(filename):
    """
    Returns True if there's *no* sign of copy (like (1), 복사본, copy, etc.)
    """
    lower = filename.lower()
    # If "copy" or "복사본" is in name => false
    if "copy" in lower or "복사본" in lower:
        return False
    # If there's a parenthesis with number => likely a copy
    if "(" in lower and ")" in lower:
        return False
    
    return True

def pick_lower_quality_or_tiebreak(path1, path2, sig1, sig2):
    """
    If qualities differ => remove the lower one.
    If tie => keep the file that looks like the 'original' name, otherwise random.
    """
    q1 = get_file_quality(sig1)
    q2 = get_file_quality(sig2)
    
    if q1 < q2:
        return path1
    elif q2 < q1:
        return path2
    else:
        # tie => check name
        orig1 = is_original_name(os.path.basename(path1))
        orig2 = is_original_name(os.path.basename(path2))
        if orig1 and not orig2:
            return path2
        elif orig2 and not orig1:
            return path1
        else:
            return random.choice([path1, path2])

###############################################################################
# 5) MOVING DUPLICATES INSTEAD OF DELETING
###############################################################################

def move_file_to_duplicates(file_path, duplicates_folder):
    """
    Moves the file to the 'duplicates_folder'. 
    If there's a collision, we append a random suffix to the filename.
    """
    if not os.path.exists(duplicates_folder):
        os.makedirs(duplicates_folder, exist_ok=True)
    
    filename = os.path.basename(file_path)
    destination = os.path.join(duplicates_folder, filename)
    
    # If a file with the same name already exists in duplicates_folder, rename
    if os.path.exists(destination):
        # E.g., insert a unique suffix: "filename (uuid4).ext"
        name, ext = os.path.splitext(filename)
        new_filename = f"{name} ({uuid.uuid4().hex[:6]}){ext}"
        destination = os.path.join(duplicates_folder, new_filename)
    
    shutil.move(file_path, destination)
    print(f"Moved => {destination}")

###############################################################################
# 6) MAIN LOGIC: DETECT & MOVE DUPLICATES
###############################################################################

def get_video_frame(video_path, fraction=0.5):
    """
    Extract a frame at 'fraction' (0..1) of the video length for display
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    target = int(frame_count * fraction)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame

def show_files_side_by_side(path1, path2, sig1, sig2, similarity):
    clear_output(wait=True)
    
    print(f"Similarity = {similarity:.2f}%")
    print(f"File 1: {os.path.basename(path1)}  ({sig1['type']})")
    print(f"File 2: {os.path.basename(path2)}  ({sig2['type']})\n")
    
    if sig1['type'] == 'image' and sig2['type'] == 'image':
        img1 = Image.open(path1)
        img2 = Image.open(path2)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img1)
        axes[0].set_title(os.path.basename(path1))
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].set_title(os.path.basename(path2))
        axes[1].axis('off')
        plt.show()
    
    elif sig1['type'] == 'video' and sig2['type'] == 'video':
        frame1 = get_video_frame(path1, 0.5)
        frame2 = get_video_frame(path2, 0.5)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        if frame1 is not None:
            axes[0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        axes[0].set_title(os.path.basename(path1))
        axes[0].axis('off')
        
        if frame2 is not None:
            axes[1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        axes[1].set_title(os.path.basename(path2))
        axes[1].axis('off')
        plt.show()

def detect_and_move_duplicates(folder_path, check_threshold=95.0, delete_threshold=99.0):
    """
    - Scans the folder for images & videos, building signatures
    - Pairwise compares them (O(n^2))
    - If similarity >= delete_threshold => auto-move to 'duplicated_photos'
    - If check_threshold <= similarity < delete_threshold => show user & ask
      (options: y => move one file, n => keep, 1 => stop)
    - If < check_threshold => skip
    - Stats at the end
    """
    all_files = load_files(folder_path)
    total_files = len(all_files)
    
    # Build signatures
    signatures = {}
    for f in all_files:
        signatures[f] = compute_file_signature(f)
    
    # Stats
    stats = {
        'total_files': total_files,
        'pairs_compared': 0,
        'duplicates_auto_moved': 0,
        'duplicates_user_moved': 0,
        'pairs_prompted': 0,
        'pairs_skipped': 0,
        'pairs_under_threshold': 0
    }
    
    file_list = list(signatures.keys())
    checked_pairs = set()
    user_stopped = False
    
    # We'll move duplicates to a subfolder "duplicated_photos" inside folder_path
    duplicates_folder = os.path.join(folder_path, "duplicated_photos")
    
    for i in range(len(file_list)):
        if user_stopped:
            break
        for j in range(i+1, len(file_list)):
            if user_stopped:
                break
            
            path1 = file_list[i]
            path2 = file_list[j]
            
            if (path1, path2) in checked_pairs or (path2, path1) in checked_pairs:
                continue
            checked_pairs.add((path1, path2))
            
            # If one was already moved (i.e. not in signatures anymore), skip
            if path1 not in signatures or path2 not in signatures:
                continue
            
            sig1 = signatures[path1]
            sig2 = signatures[path2]
            stats['pairs_compared'] += 1
            
            if not sig1 or not sig2:
                continue
            
            similarity = get_similarity(sig1, sig2, path1, path2, check_threshold)
            
            if similarity >= delete_threshold:
                # auto-move
                to_move = pick_lower_quality_or_tiebreak(path1, path2, sig1, sig2)
                print(f"[AUTO] {similarity:.2f}% => Moving to duplicates folder: {to_move}")
                move_file_to_duplicates(to_move, duplicates_folder)
                signatures.pop(to_move, None)
                stats['duplicates_auto_moved'] += 1
            
            elif similarity >= check_threshold:
                # prompt user
                stats['pairs_prompted'] += 1
                show_files_side_by_side(path1, path2, sig1, sig2, similarity)
                
                print("Options: [y] move one file, [n] keep both, [1] stop now")
                choice = input("Choice: ").strip().lower()
                
                if choice == 'y':
                    to_move = pick_lower_quality_or_tiebreak(path1, path2, sig1, sig2)
                    print(f"[USER] {similarity:.2f}% => Moving: {to_move}")
                    move_file_to_duplicates(to_move, duplicates_folder)
                    signatures.pop(to_move, None)
                    stats['duplicates_user_moved'] += 1
                elif choice == 'n':
                    print("[SKIPPED] Kept both.\n")
                    stats['pairs_skipped'] += 1
                elif choice == '1':
                    print("Stopping per user request...")
                    user_stopped = True
                    break
                else:
                    # treat as 'n'
                    print("[SKIPPED] Kept both.\n")
                    stats['pairs_skipped'] += 1
            
            else:
                stats['pairs_under_threshold'] += 1
                # skip
    
    total_moved = stats['duplicates_auto_moved'] + stats['duplicates_user_moved']
    files_remaining = total_files - total_moved
    duplicates_found = total_moved + stats['pairs_skipped']
    
    print("\n===== RUN SUMMARY =====")
    print(f"Total files scanned: {stats['total_files']}")
    print(f"Total pairs compared: {stats['pairs_compared']}")
    print(f"Pairs < {check_threshold}% similarity: {stats['pairs_under_threshold']}")
    print(f"Pairs prompted (≥ {check_threshold}% & < {delete_threshold}%): {stats['pairs_prompted']}")
    print(f"  - Duplicates user moved: {stats['duplicates_user_moved']}")
    print(f"  - Pairs user skipped (kept both): {stats['pairs_skipped']}")
    print(f"Pairs auto-moved (≥ {delete_threshold}%): {stats['duplicates_auto_moved']}")
    print(f"Total duplicates found (any ≥ {check_threshold}%): {duplicates_found}")
    print(f"Total files moved: {total_moved}")
    print(f"Files still remaining in original location: {files_remaining}")
    if user_stopped:
        print("User stopped before checking all pairs.")
    else:
        print("Completed all comparisons.")

def main():
    """
    Example usage with user input for folder path.
    """
    folder_path = input("Put the folder directory here: ")
    detect_and_move_duplicates(folder_path, check_threshold=95.0, delete_threshold=99.0)

main()  # Uncomment to run if you're in a normal Python environment
