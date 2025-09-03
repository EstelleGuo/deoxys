import numpy as np
from deoxys.utils.cam_utils import load_camera_config
import pyrealsense2 as rs
import time
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import cv2
import yaml
from deoxys import config_root
from typing import Optional, Dict, Tuple

class RealSenseCamera_V1:
    """
    Clean RealSense camera interface with hierarchical configuration support
    """
    def __init__(self, config: dict, serial_number: str = None):
        """
        Initialize RealSense camera with hierarchical config
        
        Args:
            config: Camera configuration dict from load_camera_config()
            serial_number: Camera serial number
        """
        # Basic camera settings
        self.serial_number = serial_number
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.fps = config.get('fps', 30)
        self.processing_preset = config.get('processing_preset', 1)
        
        # Parse hierarchical configuration for stream enables
        color_sensor = config.get('color_sensor', {})
        depth_sensor = config.get('depth_sensor', {})
        ir_sensors = config.get('ir_sensors', {})
        
        # Stream enable flags from hierarchical config
        self.enable_color = color_sensor.get('enable', True)
        self.enable_depth = depth_sensor.get('enable', False)
        self.enable_ir = ir_sensors.get('enable', False)
        
        # Backward compatibility with flat config
        if 'enable_color' in config:
            self.enable_color = config.get('enable_color', True)
        if 'enable_depth' in config:
            self.enable_depth = config.get('enable_depth', False)
        if 'enable_ir' in config:
            self.enable_ir = config.get('enable_ir', False)
        
        # Color sensor settings (hierarchical first, then flat fallback)
        self.color_auto_exposure = color_sensor.get('auto_exposure', config.get('color_auto_exposure', True))
        self.color_auto_white_balance = color_sensor.get('auto_white_balance', config.get('color_auto_white_balance', True))
        self.color_exposure = color_sensor.get('exposure', config.get('color_exposure', 166))
        self.color_gain = color_sensor.get('gain', config.get('color_gain', 16))
        self.color_white_balance = color_sensor.get('white_balance', config.get('color_white_balance', 4000))
        self.color_brightness = color_sensor.get('brightness', config.get('color_brightness', 32))
        self.color_contrast = color_sensor.get('contrast', config.get('color_contrast', 32))
        self.color_saturation = color_sensor.get('saturation', config.get('color_saturation', 32))
        
        # IR sensor settings (hierarchical first, then flat fallback)
        self.ir_auto_exposure = ir_sensors.get('auto_exposure', config.get('ir_auto_exposure', True))
        self.ir_exposure = ir_sensors.get('exposure', config.get('ir_exposure', 8500))
        self.ir_gain = ir_sensors.get('gain', config.get('ir_gain', 16))
        self.ir_disable_emitter = ir_sensors.get('disable_emitter', config.get('ir_disable_emitter', False))
        self.ir_laser_power = ir_sensors.get('laser_power', config.get('ir_laser_power', 360))
        
        # Initialize RealSense components
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.device = None
        self.profile = None
        self.intrinsics = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize camera with configured settings
        
        Returns:
            True if initialization successful
        """
        try:
            # Configure device serial number
            if self.serial_number:
                self.rs_config.enable_device(self.serial_number)
            
            # Configure streams
            self._configure_streams()
            
            # Start pipeline
            self.profile = self.pipeline.start(self.rs_config)
            self.device = self.profile.get_device()
            
            if self.device is None:
                print('[RealSense] Failed to get device')
                return False
            
            # Configure sensors
            time.sleep(1)  # Allow camera to stabilize
            self._configure_color_sensor()
            self._configure_ir_sensor()
            self._configure_depth_sensor()
            
            # Get intrinsics
            if self.enable_color:
                color_stream = self.profile.get_stream(rs.stream.color)
                self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            self.is_initialized = True
            print(f'[RealSense] Camera {self.serial_number} initialized successfully')
            return True
            
        except Exception as e:
            print(f'[RealSense] Initialization failed: {e}')
            return False
    
    def _configure_streams(self):
        """Configure enabled streams"""
        if self.enable_color:
            self.rs_config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
        
        if self.enable_depth:
            self.rs_config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
        
        if self.enable_ir:
            # Enable both IR streams
            self.rs_config.enable_stream(
                rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps
            )
            self.rs_config.enable_stream(
                rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps
            )
    
    def _configure_color_sensor(self):
        """Configure color sensor with hierarchical settings"""
        if not self.enable_color or not self.device:
            return
        
        try:
            color_sensor = self.device.first_color_sensor()
            if not color_sensor:
                print('[RealSense] Color sensor not found')
                return
            
            print(f'[RealSense] Configuring color sensor: {color_sensor.get_info(rs.camera_info.name)}')
            
            # Auto exposure
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1 if self.color_auto_exposure else 0)
                print(f'  Auto exposure: {"enabled" if self.color_auto_exposure else "disabled"}')
            
            # Manual exposure
            if not self.color_auto_exposure and color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, self.color_exposure)
                print(f'  Exposure: {self.color_exposure}')
            
            # Gain
            if color_sensor.supports(rs.option.gain):
                color_sensor.set_option(rs.option.gain, self.color_gain)
                print(f'  Gain: {self.color_gain}')
            
            # Auto white balance
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1 if self.color_auto_white_balance else 0)
                print(f'  Auto white balance: {"enabled" if self.color_auto_white_balance else "disabled"}')
            
            # Manual white balance
            if not self.color_auto_white_balance and color_sensor.supports(rs.option.white_balance):
                color_sensor.set_option(rs.option.white_balance, self.color_white_balance)
                print(f'  White balance: {self.color_white_balance}')
            
            # Other settings
            if color_sensor.supports(rs.option.brightness):
                color_sensor.set_option(rs.option.brightness, self.color_brightness)
                print(f'  Brightness: {self.color_brightness}')
            
            if color_sensor.supports(rs.option.contrast):
                color_sensor.set_option(rs.option.contrast, self.color_contrast)
                print(f'  Contrast: {self.color_contrast}')
            
            if color_sensor.supports(rs.option.saturation):
                color_sensor.set_option(rs.option.saturation, self.color_saturation)
                print(f'  Saturation: {self.color_saturation}')
                
        except Exception as e:
            print(f'[RealSense] Error configuring color sensor: {e}')
    
    def _configure_ir_sensor(self):
        """Configure IR sensor with hierarchical settings"""
        if not self.enable_ir or not self.device:
            return
        
        try:
            # Find stereo module for IR configuration
            for sensor in self.device.query_sensors():
                if sensor.supports(rs.camera_info.name):
                    sensor_name = sensor.get_info(rs.camera_info.name)
                    
                    if "Stereo" in sensor_name:
                        print(f'[RealSense] Configuring IR sensor: {sensor_name}')
                        
                        # Auto exposure
                        if sensor.supports(rs.option.enable_auto_exposure):
                            sensor.set_option(rs.option.enable_auto_exposure, 1 if self.ir_auto_exposure else 0)
                            print(f'  IR Auto exposure: {"enabled" if self.ir_auto_exposure else "disabled"}')
                        
                        # Manual exposure
                        if not self.ir_auto_exposure and sensor.supports(rs.option.exposure):
                            sensor.set_option(rs.option.exposure, self.ir_exposure)
                            print(f'  IR Exposure: {self.ir_exposure}')
                        
                        # Gain
                        if sensor.supports(rs.option.gain):
                            sensor.set_option(rs.option.gain, self.ir_gain)
                            print(f'  IR Gain: {self.ir_gain}')
                        
                        # Laser power
                        if sensor.supports(rs.option.laser_power):
                            sensor.set_option(rs.option.laser_power, self.ir_laser_power)
                            print(f'  IR Laser power: {self.ir_laser_power}')
                        
                        break
            
            # Configure emitter
            if self.ir_disable_emitter:
                depth_sensor = self.device.query_sensors()[0]
                # first_depth_sensor()
                if depth_sensor and depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled, 0 if self.ir_disable_emitter else 1)
                    print(f'  IR Emitter: {"disabled" if self.ir_disable_emitter else "enabled"}')
                    
        except Exception as e:
            print(f'[RealSense] Error configuring IR sensor: {e}')
    
    def _configure_depth_sensor(self):
        """Configure depth sensor settings"""
        if not self.enable_depth or not self.device:
            return
        
        try:
            depth_sensor = self.device.first_depth_sensor()
            if depth_sensor:
                # Set processing preset
                if depth_sensor.supports(rs.option.visual_preset):
                    depth_sensor.set_option(rs.option.visual_preset, self.processing_preset)
                    print(f'[RealSense] Depth processing preset: {self.processing_preset}')
                
                # Get depth scale
                self.depth_scale = depth_sensor.get_depth_scale()
                print(f'[RealSense] Depth scale: {self.depth_scale}')
                
        except Exception as e:
            print(f'[RealSense] Error configuring depth sensor: {e}')
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Get frame data from camera
        
        Returns:
            Tuple of (color_image, depth_image, ir_left, ir_right) or None if failed
            Only enabled streams will return data, others will be None
        """
        if not self.is_initialized:
            print('[RealSense] Camera not initialized')
            return None
        
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_image = None
            depth_image = None
            ir_left = None
            ir_right = None
            
            # Get color frame
            if self.enable_color:
                color_frame = aligned_frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
            
            # Get depth frame
            if self.enable_depth:
                depth_frame = aligned_frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
            
            # Get IR frames
            if self.enable_ir:
                ir_frame_left = frames.get_infrared_frame(1)
                ir_frame_right = frames.get_infrared_frame(2)
                if ir_frame_left:
                    ir_left = np.asanyarray(ir_frame_left.get_data())
                if ir_frame_right:
                    ir_right = np.asanyarray(ir_frame_right.get_data())
            
            return color_image, depth_image, ir_left, ir_right
            
        except Exception as e:
            print(f'[RealSense] Error getting frame: {e}')
            return None
    
    def get_color_frame(self) -> Optional[np.ndarray]:
        """Get only color frame (convenience method)"""
        result = self.get_frame()
        return result[0] if result else None
    
    def get_intrinsics(self) -> Optional[rs.intrinsics]:
        """Get camera intrinsics"""
        return self.intrinsics
    
    def get_camera_info(self) -> dict:
        """Get comprehensive camera information"""
        if not self.device:
            return {}
        
        try:
            device_name = self.device.get_info(rs.camera_info.name)
            firmware_version = self.device.get_info(rs.camera_info.firmware_version)
            
            info = {
                'serial_number': self.serial_number,
                'device_name': device_name,
                'firmware_version': firmware_version,
                'resolution': {'width': self.width, 'height': self.height},
                'fps': self.fps,
                'enabled_streams': {
                    'color': self.enable_color,
                    'depth': self.enable_depth,
                    'ir': self.enable_ir
                }
            }
            
            if self.intrinsics:
                info['intrinsics'] = {
                    'fx': self.intrinsics.fx,
                    'fy': self.intrinsics.fy,
                    'ppx': self.intrinsics.ppx,
                    'ppy': self.intrinsics.ppy,
                    'distortion': list(self.intrinsics.coeffs)
                }
            
            return info
            
        except Exception as e:
            print(f'[RealSense] Error getting camera info: {e}')
            return {}
    
    def _intrinsics_to_dict(self, intrinsics: rs.intrinsics) -> dict:
        """Convert RealSense intrinsics to dictionary"""
        return {
            'width': intrinsics.width,
            'height': intrinsics.height,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'ppx': intrinsics.ppx,
            'ppy': intrinsics.ppy,
            'model': str(intrinsics.model),
            'coeffs': list(intrinsics.coeffs)
        }
    
    def _extrinsics_to_dict(self, extrinsics: rs.extrinsics) -> dict:
        """Convert RealSense extrinsics to dictionary"""
        return {
            'rotation': [list(extrinsics.rotation[i:i+3]) for i in range(0, 9, 3)],
            'translation': list(extrinsics.translation)
        }
    
    def get_comprehensive_camera_parameters(self) -> Optional[Dict]:
        """
        Get comprehensive camera calibration parameters including all streams
        
        Returns:
            Dictionary containing all camera parameters including intrinsics and extrinsics
        """
        if not self.is_initialized or not self.device or not self.profile:
            print("[RealSense] Camera not properly initialized")
            return None
            
        try:
            # Get device information
            serial_number = self.device.get_info(rs.camera_info.serial_number)
            device_name = self.device.get_info(rs.camera_info.name)
            
            camera_params = {
                'device_info': {
                    'serial_number': serial_number,
                    'device_name': device_name,
                    'resolution': {
                        'width': self.width, 
                        'height': self.height
                    }
                },
                'intrinsics': {},
                'extrinsics': {}
            }
            
            # Get stream objects and intrinsics for enabled streams
            streams = {}
            
            if self.enable_color:
                color_stream = self.profile.get_stream(rs.stream.color)
                color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                streams['color'] = color_stream
                camera_params['intrinsics']['color'] = self._intrinsics_to_dict(color_intrinsics)
            
            if self.enable_ir:
                ir1_stream = self.profile.get_stream(rs.stream.infrared, 1)  # Left IR
                ir2_stream = self.profile.get_stream(rs.stream.infrared, 2)  # Right IR
                ir1_intrinsics = ir1_stream.as_video_stream_profile().get_intrinsics()
                ir2_intrinsics = ir2_stream.as_video_stream_profile().get_intrinsics()
                streams['ir_left'] = ir1_stream
                streams['ir_right'] = ir2_stream
                camera_params['intrinsics']['left_ir'] = self._intrinsics_to_dict(ir1_intrinsics)
                camera_params['intrinsics']['right_ir'] = self._intrinsics_to_dict(ir2_intrinsics)
            
            if self.enable_depth:
                depth_stream = self.profile.get_stream(rs.stream.depth)
                depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
                streams['depth'] = depth_stream
                camera_params['intrinsics']['depth'] = self._intrinsics_to_dict(depth_intrinsics)
            
            # Get extrinsics between enabled streams
            if self.enable_ir and len(streams) >= 2:
                ir1_stream = streams['ir_left']
                ir2_stream = streams['ir_right']
                
                # IR to IR extrinsics
                extrinsics_ir1_to_ir2 = ir1_stream.get_extrinsics_to(ir2_stream)
                extrinsics_ir2_to_ir1 = ir2_stream.get_extrinsics_to(ir1_stream)
                camera_params['extrinsics']['left_ir_to_right_ir'] = self._extrinsics_to_dict(extrinsics_ir1_to_ir2)
                camera_params['extrinsics']['right_ir_to_left_ir'] = self._extrinsics_to_dict(extrinsics_ir2_to_ir1)
                
                # Stereo baseline calculation
                camera_params['stereo_baseline'] = abs(extrinsics_ir1_to_ir2.translation[0])
                
                # IR to Color extrinsics (if color is enabled)
                if self.enable_color:
                    color_stream = streams['color']
                    extrinsics_ir1_to_color = ir1_stream.get_extrinsics_to(color_stream)
                    extrinsics_color_to_ir1 = color_stream.get_extrinsics_to(ir1_stream)
                    camera_params['extrinsics']['left_ir_to_color'] = self._extrinsics_to_dict(extrinsics_ir1_to_color)
                    camera_params['extrinsics']['color_to_left_ir'] = self._extrinsics_to_dict(extrinsics_color_to_ir1)
                
                # IR to Depth extrinsics (if depth is enabled)
                if self.enable_depth:
                    depth_stream = streams['depth']
                    extrinsics_ir1_to_depth = ir1_stream.get_extrinsics_to(depth_stream)
                    extrinsics_depth_to_ir1 = depth_stream.get_extrinsics_to(ir1_stream)
                    camera_params['extrinsics']['left_ir_to_depth'] = self._extrinsics_to_dict(extrinsics_ir1_to_depth)
                    camera_params['extrinsics']['depth_to_left_ir'] = self._extrinsics_to_dict(extrinsics_depth_to_ir1)
            
            # Color to Depth extrinsics (if both are enabled)
            if self.enable_color and self.enable_depth:
                color_stream = streams['color']
                depth_stream = streams['depth']
                extrinsics_color_to_depth = color_stream.get_extrinsics_to(depth_stream)
                extrinsics_depth_to_color = depth_stream.get_extrinsics_to(color_stream)
                camera_params['extrinsics']['color_to_depth'] = self._extrinsics_to_dict(extrinsics_color_to_depth)
                camera_params['extrinsics']['depth_to_color'] = self._extrinsics_to_dict(extrinsics_depth_to_color)
            
            return camera_params
            
        except Exception as e:
            print(f"[RealSense] Error getting comprehensive camera parameters: {e}")
            return None
    
    def release(self):
        """Release camera resources"""
        try:
            if self.pipeline:
                self.pipeline.stop()
            self.is_initialized = False
            print(f'[RealSense] Camera {self.serial_number} released')
        except Exception as e:
            print(f'[RealSense] Error releasing camera: {e}')


def export_camera_parameters(camera_params: dict, output_path: str = None, camera_name: str = None) -> bool:
    """
    Export camera parameters to a YAML file
    
    Args:
        camera_params: Dictionary from camera.get_comprehensive_camera_parameters()
        output_path: Path to save the YAML file. If None, uses default naming
        camera_name: Optional camera name to include in filename
        
    Returns:
        True if export successful, False otherwise
    """
    import os
    from datetime import datetime
    
    if camera_params is None:
        print("[Export] No camera parameters provided")
        return False
    
    try:
        # Generate default filename if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            serial_num = camera_params.get('device_info', {}).get('serial_number', 'unknown')
            if camera_name:
                filename = f"camera_params_{camera_name}_{serial_num}_{timestamp}.yml"
            else:
                filename = f"camera_params_{serial_num}_{timestamp}.yml"
            output_path = os.path.join(os.getcwd(), filename)
        
        # Prepare data for YAML export with better organization
        export_data = {
            'camera_calibration': {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'exported_by': 'RealSense_V1_Camera_Interface',
                    'camera_name': camera_name if camera_name else 'Unknown'
                },
                'device_info': camera_params.get('device_info', {}),
                'intrinsics': camera_params.get('intrinsics', {}),
                'extrinsics': camera_params.get('extrinsics', {}),
            }
        }
        
        # Add stereo baseline if available
        if 'stereo_baseline' in camera_params:
            export_data['camera_calibration']['stereo_baseline'] = float(camera_params['stereo_baseline'])
        
        # Add depth scale if available
        if 'depth_scale' in camera_params:
            export_data['camera_calibration']['depth_scale'] = float(camera_params['depth_scale'])
        
        # Convert numpy arrays to lists for YAML serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif hasattr(obj, 'item'):   # numpy scalar
                return obj.item()
            else:
                return obj
        
        export_data = convert_numpy_types(export_data)
        
        # Write to YAML file
        with open(output_path, 'w') as file:
            yaml.dump(export_data, file, default_flow_style=False, indent=2, sort_keys=False)
        
        print(f"[Export] Camera parameters exported successfully to: {output_path}")
        
        # Print summary
        intrinsics_count = len(export_data['camera_calibration']['intrinsics'])
        extrinsics_count = len(export_data['camera_calibration']['extrinsics'])
        print(f"[Export] Exported {intrinsics_count} intrinsic parameter sets and {extrinsics_count} extrinsic transformations")
        
        return True
        
    except Exception as e:
        print(f"[Export] Error exporting camera parameters: {e}")
        return False


def load_exported_camera_parameters(yaml_path: str) -> Optional[dict]:
    """
    Load previously exported camera parameters from YAML file
    
    Args:
        yaml_path: Path to the exported camera parameters YAML file
        
    Returns:
        Dictionary containing camera parameters or None if failed
    """
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        
        if 'camera_calibration' in data:
            print(f"[Load] Successfully loaded camera parameters from: {yaml_path}")
            export_info = data['camera_calibration'].get('export_info', {})
            print(f"[Load] Export timestamp: {export_info.get('timestamp', 'Unknown')}")
            print(f"[Load] Camera name: {export_info.get('camera_name', 'Unknown')}")
            return data['camera_calibration']
        else:
            print(f"[Load] Invalid camera parameters file format: {yaml_path}")
            return None
            
    except Exception as e:
        print(f"[Load] Error loading camera parameters: {e}")
        return None


if __name__ == "__main__":
    # Load hierarchical camera configuration
    config = load_camera_config('/home/nuc/Documents/WorkSpaces/nyc_ws/deoxys_control/deoxys/config/camera_setup_config_test.yml') 
    
    # Get first camera info
    cam_info_0 = config['camera_infos'][0]
    print("Camera Configuration:")
    print(cam_info_0)
    print()
    
    # # Reset hardware before initialization
    # ctx = rs.context()
    # devices = ctx.query_devices()
    # for dev in devices:
    #     dev.hardware_reset()
    # print("Hardware reset completed. Waiting 5 seconds...")
    # time.sleep(5)
    
    # # Create RealSense camera with clean interface
    # serial_number = getattr(cam_info_0, 'camera_serial_num', None)
    # cam_name = getattr(cam_info_0, 'camera_name', None)
    # cam_config = getattr(cam_info_0, 'cfg', {})
    
    # print(f"Initializing camera: {cam_name} (Serial: {serial_number})")
    # camera = RealSenseCamera_V1(cam_config, serial_number=serial_number)
    
    # if not camera.initialize():
    #     print("Failed to initialize camera")
    #     exit(1)
    
    # print("\nCamera Information:")
    # print(camera.get_camera_info())
    
    # # Get comprehensive camera parameters including intrinsics and extrinsics
    # print("\nComprehensive Camera Parameters:")
    # camera_params = camera.get_comprehensive_camera_parameters()
    # if camera_params:
    #     print("Device Info:", camera_params['device_info'])
    #     print("Available Intrinsics:", list(camera_params['intrinsics'].keys()))
    #     print("Available Extrinsics:", list(camera_params['extrinsics'].keys()))
        
    #     # Print stereo baseline if available
    #     if 'stereo_baseline' in camera_params:
    #         print(f"Stereo Baseline: {camera_params['stereo_baseline']:.4f} meters")
        
    #     # Print IR intrinsics if available
    #     if 'left_ir' in camera_params['intrinsics']:
    #         left_ir = camera_params['intrinsics']['left_ir']
    #         print(f"Left IR Intrinsics - fx: {left_ir['fx']:.2f}, fy: {left_ir['fy']:.2f}")
    #         print(f"Left IR Principal Point - ppx: {left_ir['ppx']:.2f}, ppy: {left_ir['ppy']:.2f}")
        
    #     # Export camera parameters to YAML file
    #     print("\nExporting camera parameters...")
    #     export_success = export_camera_parameters(
    #         camera_params, 
    #         output_path=None,  # Use default naming
    #         camera_name=cam_name
    #     )
        
    #     if export_success:
    #         print("✓ Camera parameters exported successfully")
    #     else:
    #         print("✗ Failed to export camera parameters")
            
    # else:
    #     print("Failed to get comprehensive camera parameters")
    
    # print(f"\nStarting live feed for {cam_name}. Press ESC to exit...")
    
    # try:
    #     while True:
    #         # Get frame data (returns tuple of enabled streams)
    #         frame_data = camera.get_frame()
    #         if frame_data is None:
    #             continue
            
    #         color_image, depth_image, ir_left, ir_right = frame_data
            
    #         # Display enabled streams
    #         if color_image is not None:
    #             cv2.imshow(f"{cam_name} - Color", color_image)
            
    #         if depth_image is not None:
    #             # Convert depth to 8-bit for display
    #             depth_colormap = cv2.applyColorMap(
    #                 cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    #             )
    #             cv2.imshow(f"{cam_name} - Depth", depth_colormap)
            
    #         if ir_left is not None:
    #             cv2.imshow(f"{cam_name} - IR Left", ir_left)
            
    #         if ir_right is not None:
    #             cv2.imshow(f"{cam_name} - IR Right", ir_right)
            
    #         key = cv2.waitKey(1) & 0xFF
    #         if key == 27:  # ESC to exit
    #             break
                
    # except KeyboardInterrupt:
    #     print("\nStopping camera...")
    
    # finally:
    #     camera.release()
    #     cv2.destroyAllWindows()
    #     print("Camera released successfully")

