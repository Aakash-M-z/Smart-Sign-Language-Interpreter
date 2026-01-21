#!/usr/bin/env python3
"""
Simple webcam test to check camera availability
"""

import cv2
import numpy as np

def test_webcam():
    """Test webcam functionality"""
    print("Testing webcam access...")
    
    # Try different camera indices
    for cam_id in range(5):
        print(f"\nTrying camera {cam_id}...")
        
        try:
            cap = cv2.VideoCapture(cam_id)
            
            if cap.isOpened():
                print(f"✅ Camera {cam_id} opened successfully")
                
                # Try to read a frame
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Camera {cam_id} can capture frames")
                    print(f"   Frame shape: {frame.shape}")
                    
                    # Show a test window for 3 seconds
                    cv2.imshow(f'Camera {cam_id} Test', frame)
                    print("   Showing test frame for 3 seconds...")
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()
                    
                    cap.release()
                    return cam_id
                else:
                    print(f"❌ Camera {cam_id} cannot capture frames")
                
                cap.release()
            else:
                print(f"❌ Camera {cam_id} failed to open")
                
        except Exception as e:
            print(f"❌ Camera {cam_id} error: {e}")
    
    print("\n❌ No working cameras found")
    return None

def main():
    """Main function"""
    print("=" * 50)
    print("🎥 Webcam Test")
    print("=" * 50)
    
    working_cam = test_webcam()
    
    if working_cam is not None:
        print(f"\n✅ Found working camera: {working_cam}")
        print(f"You can use: python realtime_inference_fixed.py --webcam {working_cam}")
    else:
        print("\n❌ No working cameras found")
        print("Possible solutions:")
        print("1. Check if camera is connected")
        print("2. Close other apps using the camera")
        print("3. Check camera permissions")
        print("4. Try external USB camera")

if __name__ == "__main__":
    main()