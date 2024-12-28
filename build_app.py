import PyInstaller.__main__
import os
from pathlib import Path

def build_exe():
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Create icon from the first frame of video
    icon_path = current_dir / "app_icon.ico"
    
    # Determine data files
    data_files = [
        ('README.md', '.'),
    ]
    
    # Convert data files to PyInstaller format
    data_args = []
    for src, dst in data_files:
        data_args.extend(['--add-data', f'{src};{dst}' if os.name == 'nt' else f'{src}:{dst}'])
    
    # Build command
    cmd_args = [
        'main.py',
        '--name=VideoScanner',
        '--onefile',
        '--windowed',
        '--clean',
        '--noconfirm',
    ]
    
    # Add icon if exists
    if icon_path.exists():
        cmd_args.extend(['--icon', str(icon_path)])
    
    # Add hidden imports
    hidden_imports = [
        'torch',
        'cv2',
        'numpy',
        'moviepy',
        'colorama',
        'tqdm',
        'moviepy.editor',
        'PIL',
        'proglog'
    ]
    for imp in hidden_imports:
        cmd_args.extend(['--hidden-import', imp])
    
    # Add data files
    cmd_args.extend(data_args)
    
    # Add debug options if needed
    # cmd_args.extend(['-d', 'all'])  # Uncomment for debug build
    
    print("Building with arguments:", cmd_args)
    PyInstaller.__main__.run(cmd_args)

if __name__ == "__main__":
    build_exe()
