"""
Setup script to help extract and configure the ISL dataset
"""
from pathlib import Path
import subprocess
import sys
import os

def check_extraction_tools():
    """Check for available extraction tools"""
    tools = {
        'WinRAR': [
            r"C:\Program Files\WinRAR\WinRAR.exe",
            r"C:\Program Files (x86)\WinRAR\WinRAR.exe"
        ],
        '7-Zip': [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe"
        ]
    }
    
    found_tools = []
    for tool_name, paths in tools.items():
        for path in paths:
            if Path(path).exists():
                found_tools.append((tool_name, path))
                break
    
    return found_tools

def extract_with_7zip(rar_path, extract_to):
    """Extract RAR using 7-Zip"""
    tools = check_extraction_tools()
    for tool_name, tool_path in tools:
        if tool_name == '7-Zip':
            print(f"Using 7-Zip at: {tool_path}")
            cmd = [tool_path, 'x', str(rar_path), f'-o{extract_to}', '-y']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return True
            except Exception as e:
                print(f"Error: {e}")
    return False

def extract_with_winrar(rar_path, extract_to):
    """Extract RAR using WinRAR"""
    tools = check_extraction_tools()
    for tool_name, tool_path in tools:
        if tool_name == 'WinRAR':
            print(f"Using WinRAR at: {tool_path}")
            cmd = [tool_path, 'x', str(rar_path), str(extract_to), '-y']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    return True
            except Exception as e:
                print(f"Error: {e}")
    return False

def main():
    """Main setup function"""
    print("="*70)
    print("ISL Dataset Setup Script")
    print("="*70)
    
    # Check for RAR file
    rar_path = Path(r"C:\Users\Aakash\Downloads\ISL Hand Gesture Dataset\ISL Hand Gesture Dataset\ISL custom Data.rar")
    extract_to = Path(r"C:\Users\Aakash\Downloads\ISL Hand Gesture Dataset\ISL_Dataset_Extracted")
    
    print(f"\nRAR file location: {rar_path}")
    print(f"Extraction destination: {extract_to}")
    
    if not rar_path.exists():
        print(f"\n[ERROR] RAR file not found at: {rar_path}")
        print(f"Please ensure the RAR file exists at the above location.")
        return
    
    print(f"[OK] RAR file found!")
    
    # Check if already extracted
    dataset_path = extract_to / "ISL custom Data"
    if dataset_path.exists():
        # Check for files
        video_files = list(dataset_path.rglob("*.mp4")) + list(dataset_path.rglob("*.MP4"))
        image_files = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.JPG"))
        
        if len(video_files) > 0 or len(image_files) > 0:
            print(f"\n[INFO] Dataset already extracted!")
            print(f"Found {len(video_files)} videos and {len(image_files)} images")
            print(f"\nDataset path: {dataset_path}")
            print(f"\n[SUCCESS] Dataset is ready to use!")
            return
    
    # Try to extract
    print(f"\n[INFO] Attempting to extract dataset...")
    
    # Check for extraction tools
    tools = check_extraction_tools()
    
    if not tools:
        print(f"\n[ERROR] No extraction tools found!")
        print(f"\nPlease install one of the following:")
        print(f"  1. WinRAR (https://www.winrar.com/)")
        print(f"  2. 7-Zip (https://www.7-zip.org/)")
        print(f"\nThen manually extract:")
        print(f"  Source: {rar_path}")
        print(f"  Destination: {extract_to}")
        print(f"\nAfter extraction, run this script again to verify.")
        return
    
    print(f"\n[INFO] Found extraction tools: {[t[0] for t in tools]}")
    
    # Try extraction
    success = False
    for tool_name, tool_path in tools:
        if tool_name == '7-Zip':
            success = extract_with_7zip(rar_path, extract_to)
        elif tool_name == 'WinRAR':
            success = extract_with_winrar(rar_path, extract_to)
        
        if success:
            print(f"\n[SUCCESS] Dataset extracted successfully!")
            break
    
    if not success:
        print(f"\n[ERROR] Extraction failed. Please extract manually:")
        print(f"  1. Right-click on: {rar_path}")
        print(f"  2. Select 'Extract to...' or 'Extract Here'")
        print(f"  3. Extract to: {extract_to}")
        print(f"\nThen run this script again to verify.")
        return
    
    # Verify extraction
    print(f"\n[INFO] Verifying extraction...")
    if dataset_path.exists():
        video_files = list(dataset_path.rglob("*.mp4")) + list(dataset_path.rglob("*.MP4"))
        image_files = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.JPG"))
        
        if len(video_files) > 0 or len(image_files) > 0:
            print(f"[SUCCESS] Dataset extracted and verified!")
            print(f"  Videos: {len(video_files)}")
            print(f"  Images: {len(image_files)}")
            print(f"\nDataset path: {dataset_path}")
            print(f"\n[SUCCESS] Dataset is ready to use!")
        else:
            print(f"[WARNING] Extraction completed but no files found.")
            print(f"Please check the extraction directory manually.")
    else:
        print(f"[ERROR] Extraction directory not found after extraction.")
    
    print(f"\n" + "="*70)

if __name__ == "__main__":
    main()

