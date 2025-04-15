import os

def rename_stl_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.STL'):
            base = os.path.splitext(filename)[0]
            new_name = base + '.stl'
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

# Example usage:
# Replace 'your_directory_path' with the path to your STL files
rename_stl_files('/home/noboru/KIMLAB_WS/G1_ws/src/ECE598-Throwing/robot/g1/meshes')
