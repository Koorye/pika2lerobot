from dataclasses import dataclass


@dataclass
class DataConfig:
    overwrite = True

    source_data_roots = [
        '/media/shihanwu/新加卷/pika_wushihan/left',
        '/media/shihanwu/新加卷/pika_wushihan/right',
    ]

    image_height = 480
    image_width = 640
    rgb_dirs = [
        'camera/color/camera_realsense_c',
        'camera/color/pikaDepthCamera_l',
        'camera/color/pikaFisheyeCamera_l',
        'camera/color/pikaDepthCamera_r',
        'camera/color/pikaFisheyeCamera_r',
    ]
    rgb_names = [
        'third_base_rgb',
        'left_wrist_base_rgb',
        'left_wrist_fisheye_rgb',
        'right_wrist_base_rgb',
        'right_wrist_fisheye_rgb',
    ]

    action_len = 14
    action_dirs = [
        'localization/pose/pika_l',
        'gripper/encoder/pika_l',
        'localization/pose/pika_r',
        'gripper/encoder/pika_r',
    ]
    action_keys_list = [
        ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        ['angle'],
        ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        ['angle'],
    ]
    nonoop_threshold = 1e-3

    instruction_path = 'instructions.json'
    default_instruction = 'put the banana in the basket'

    repo_id = 'Koorye/pika'
    data_root = None
    fps=30
    video_backend = 'pyav'


