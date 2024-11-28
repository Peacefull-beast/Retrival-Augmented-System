from path import *
import os
import shutil

def remove_folders(folder_path):
    try:
    # shutil.rmtree is used to remove a directory and all its contents
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' has been removed successfully.")
        else:
            print(f"Folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"Error: {e}")


# Specify the folder path you want to remove
folders = [EMBEDDED_IMAGE_PATH, PAGES, CHROMA_PATH_IMAGE, CHROMA_PATH_TEXT]

for folder in folders:
    folder_path = os.path.join(os.getcwd(), folder)
    remove_folders(folder_path)
# Remove folder using os
