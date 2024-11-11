import os
import numpy as np

ENCODINGS_PATH = 'data/encodings.npy'
NAMES_PATH = 'data/names.npy'

def load_known_faces():
    """Load known faces and their encodings."""
    if os.path.exists(ENCODINGS_PATH) and os.path.exists(NAMES_PATH):
        try:
            # Load encodings and names from files
            encodings = np.load(ENCODINGS_PATH, allow_pickle=True)
            names = np.load(NAMES_PATH, allow_pickle=True)
            return encodings, names
        except Exception as e:
            print(f"Error loading encodings: {e}")
            return [], []  # Return empty lists if there’s an issue loading the file
    else:
        print("No known faces found. Please add faces using capture_face.py.")
        return [], []  # Return empty lists if the file doesn't exist
