import zipfile
import os

def zip_project(output_filename='project_bundle.zip'):
    print(f"Zipping project to {output_filename}...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Exclude unnecessary directories
            if 'env' in root or '__pycache__' in root or '.git' in root:
                continue
                
            for file in files:
                if file.endswith('.zip') or file.endswith('.pyc'):
                    continue
                    
                file_path = os.path.join(root, file)
                print(f"Adding {file_path}")
                zipf.write(file_path, os.path.relpath(file_path, '.'))
    print("Done!")

if __name__ == "__main__":
    zip_project()
