"""
Test script to process .abc file using MOD_ModelConverter,
convert to COLMAP format, and load with pycolmap.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add mast3r to path
sys.path.insert(0, str(Path(__file__).parent))

from MOD_ModelConverter import (
    inspect_abc_file, 
    abc_to_colmap,
    AlembicArchive,
    extract_cameras,
    OgawaReader
)
from MOD_ModelConverter.converter import print_inspection

def main():
    # Input file
    abc_file = r"D:\_DataBuffer\RopeCap0121\17\ps_group_004_cam0+1.abc"
    
    print("=" * 70)
    print("STEP 1: Inspect the .abc file structure")
    print("=" * 70)
    
    if not os.path.exists(abc_file):
        print(f"ERROR: File not found: {abc_file}")
        return
    
    # Basic Ogawa file info
    print(f"File: {abc_file}")
    print(f"Size: {os.path.getsize(abc_file)} bytes")
    
    try:
        with OgawaReader(abc_file) as reader:
            print(f"\nOgawa Header:")
            print(f"  Valid: {reader.header.is_valid}")
            print(f"  Frozen: {reader.header.is_frozen}")
            print(f"  Version: {reader.header.version}")
            print(f"  Root offset: {reader.header.root_group_offset}")
            
            root = reader.get_root_group()
            print(f"  Root children: {root.num_children}")
            
            # Try to dump limited structure
            print(f"\nFile structure (limited):")
            reader.dump_structure(max_depth=2)
    except Exception as e:
        print(f"Error reading Ogawa structure: {e}")
    
    # Try full inspection (may fail on some files)
    print("\n--- Full Alembic inspection ---")
    try:
        info = inspect_abc_file(abc_file, max_depth=2)
        print_inspection(info)
    except Exception as e:
        print(f"Error during full inspection: {e}")
    
    print("\n" + "=" * 70)
    print("STEP 2: Extract cameras from the .abc file")
    print("=" * 70)
    
    try:
        with AlembicArchive(abc_file) as archive:
            # First, let's see what objects are in the file
            print("\n--- Objects in archive ---")
            cam_objects = archive.find_cameras()
            xform_objects = archive.find_xforms()
            print(f"Camera objects: {len(cam_objects)}")
            for c in cam_objects:
                print(f"  - {c.full_name} (schema: {c.schema})")
                props = c.get_properties()
                print(f"    Properties: {list(props.keys())}")
                for pname, prop in props.items():
                    print(f"      {pname}: type={prop.header.property_type.name}, dt={prop.header.data_type}")
                    if prop.is_compound:
                        print(f"        (Compound property - checking sub-properties)")
                        sub_props = prop.get_sub_properties()
                        print(f"        Sub-properties: {list(sub_props.keys())}")
                    else:
                        samples = prop.get_all_samples()
                        print(f"        {len(samples)} samples")
                        if samples and len(samples) > 0:
                            sample = samples[0]
                            if isinstance(sample, bytes):
                                print(f"        [0]: bytes({len(sample)}) = {sample[:50]}...")
                            elif isinstance(sample, (list, tuple)) and len(sample) > 10:
                                print(f"        [0]: {sample[:10]}... ({len(sample)} values)")
                            else:
                                print(f"        [0]: {sample}")
            
            print(f"\nXform objects: {len(xform_objects)}")
            for x in xform_objects:
                print(f"  - {x.full_name} (schema: {x.schema})")
                props = x.get_properties()
                print(f"    Properties: {list(props.keys())}")
                for pname, prop in props.items():
                    print(f"      {pname}: type={prop.header.property_type.name}, dt={prop.header.data_type}")
                    if prop.is_compound:
                        print(f"        (Compound property - checking sub-properties)")
                        sub_props = prop.get_sub_properties()
                        print(f"        Sub-properties: {list(sub_props.keys())}")
                    else:
                        samples = prop.get_all_samples()
                        print(f"        {len(samples)} samples")
                        if samples and len(samples) > 0:
                            sample = samples[0]
                            if isinstance(sample, bytes):
                                print(f"        [0]: bytes({len(sample)}) = {sample[:50]}...")
                            elif isinstance(sample, (list, tuple)) and len(sample) > 10:
                                print(f"        [0]: {sample[:10]}... ({len(sample)} values)")
                            else:
                                print(f"        [0]: {sample}")
            
            # Now extract cameras
            print("\n--- Extracted cameras ---")
            cameras = extract_cameras(archive, image_width=1920, image_height=1080)
            print(f"\nFound {len(cameras)} camera(s):")
            for i, cam in enumerate(cameras):
                print(f"\n  Camera {i}: {cam.name}")
                print(f"    Image Size: {cam.image_width} x {cam.image_height}")
                print(f"    Intrinsics:")
                print(f"      Focal Length: {cam.intrinsics.focal_length:.2f} mm")
                print(f"      H-Aperture: {cam.intrinsics.horizontal_aperture:.4f} cm")
                print(f"      V-Aperture: {cam.intrinsics.vertical_aperture:.4f} cm")
                fx, fy = cam.intrinsics.get_focal_length_pixels(cam.image_width, cam.image_height)
                cx, cy = cam.intrinsics.get_principal_point_pixels(cam.image_width, cam.image_height)
                print(f"      fx: {fx:.4f} px, fy: {fy:.4f} px")
                print(f"      cx: {cx:.4f} px, cy: {cy:.4f} px")
                print(f"      FOV H: {cam.intrinsics.get_field_of_view_horizontal():.2f} deg")
                print(f"      FOV V: {cam.intrinsics.get_field_of_view_vertical():.2f} deg")
                print(f"    Transform:")
                print(f"      Translation: {cam.transform.translation}")
                print(f"      Quaternion: {cam.transform.rotation_quaternion}")
    except Exception as e:
        print(f"Error extracting cameras: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("STEP 3: Convert to COLMAP format")
    print("=" * 70)
    
    # Create temp output directory
    output_dir = Path(tempfile.mkdtemp(prefix="abc_colmap_"))
    print(f"Output directory: {output_dir}")
    
    try:
        result = abc_to_colmap(
            abc_file,
            output_dir,
            image_width=1920,
            image_height=1080,
            camera_model="PINHOLE",
            debug=True
        )
        
        print("\nCreated files:")
        for key, path in result.items():
            if key != 'output_dir':
                print(f"  {key}: {path}")
        
        # Print file contents
        print("\n--- cameras.txt ---")
        with open(result['cameras'], 'r') as f:
            print(f.read())
        
        print("--- images.txt ---")
        with open(result['images'], 'r') as f:
            print(f.read())
        
        if 'debug' in result:
            print("--- cameras_debug.txt ---")
            with open(result['debug'], 'r') as f:
                print(f.read())
                
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("STEP 4: Load with pycolmap and print details")
    print("=" * 70)
    
    try:
        import pycolmap
        
        # Load the reconstruction
        reconstruction = pycolmap.Reconstruction(str(output_dir))
        
        print(f"\nPycolmap Reconstruction loaded successfully!")
        print(f"  Number of cameras: {reconstruction.num_cameras()}")
        print(f"  Number of images: {reconstruction.num_images()}")
        print(f"  Number of points3D: {reconstruction.num_points3D()}")
        
        print("\n--- Cameras ---")
        for cam_id, camera in reconstruction.cameras.items():
            print(f"  Camera ID: {cam_id}")
            print(f"    Model: {camera.model}")
            print(f"    Width x Height: {camera.width} x {camera.height}")
            print(f"    Params: {list(camera.params)}")
            # Different pycolmap versions have different APIs
            try:
                print(f"    Focal Length: {camera.focal_length}")
            except:
                print(f"    Focal Length (params[0]): {camera.params[0]}")
            try:
                print(f"    Principal Point: ({camera.principal_point_x}, {camera.principal_point_y})")
            except:
                print(f"    Principal Point (params): ({camera.params[2]}, {camera.params[3]})")
        
        print("\n--- Images ---")
        for img_id, image in reconstruction.images.items():
            print(f"  Image ID: {img_id}")
            print(f"    Name: {image.name}")
            print(f"    Camera ID: {image.camera_id}")
            # Handle different pycolmap API versions
            try:
                # Try cam_from_world() as method
                cfw = image.cam_from_world()
                print(f"    Quaternion (wxyz): {cfw.rotation.quat}")
                print(f"    Translation: {cfw.translation}")
            except Exception as e1:
                try:
                    # Try as property
                    print(f"    Quaternion (wxyz): {image.cam_from_world.rotation.quat}")
                    print(f"    Translation: {image.cam_from_world.translation}")
                except Exception as e2:
                    try:
                        # Older API with qvec/tvec
                        print(f"    Quaternion (wxyz): {image.qvec}")
                        print(f"    Translation: {image.tvec}")
                    except Exception as e3:
                        print(f"    (Could not access pose: {e1})")
        
        print("\n" + "=" * 70)
        print("SUCCESS: ABC file processed, converted, and loaded with pycolmap!")
        print("=" * 70)
        
    except ImportError:
        print("pycolmap not installed. Install with: pip install pycolmap")
    except Exception as e:
        print(f"Error loading with pycolmap: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
