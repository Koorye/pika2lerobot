import json
import os
import numpy as np
import shutil
from collections import defaultdict
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from config import DataConfig
from utils import get_lerobot_default_root, load_image


class DataProcessor(object):
    def __init__(self, config):
        self.config = config

        if self.config.overwrite:
            data_root = get_lerobot_default_root() if self.config.data_root is None else self.config.data_root
            shutil.rmtree(os.path.join(data_root, self.config.repo_id), ignore_errors=True)
        
        self.create_dataset()

    def create_dataset(self):
        rgb_config = {
            'dtype': 'video',
            'shape': (self.config.image_height, self.config.image_width, 3),
            'name': ['height', 'width', 'channels'],
        }
        features = {rgb_name: rgb_config for rgb_name in self.config.rgb_names}
        features['states'] = {
            'dtype': 'float64',
            'shape': (self.config.action_len,),
            'name': ['states'],
        }
        features['actions'] = {
            'dtype': 'float64',
            'shape': (self.config.action_len,),
            'name': ['actions'],
        }

        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            root=self.config.data_root,
            video_backend=self.config.video_backend,
            features=features,
        )
    
    def process_data(self):
        for source_idx, source_data_root in enumerate(self.config.source_data_roots):
            episode_dirs = [d for d in os.listdir(source_data_root)]
            for episode_idx, episode_dir in enumerate(sorted(episode_dirs, key=lambda x: int(x[7:]))):
                episode_path = os.path.join(source_data_root, episode_dir)
                print(f'Processing source {source_idx + 1}/{len(self.config.source_data_roots)}, episode {episode_idx + 1}/{len(episode_dirs)}: {episode_path}')
                self._add_episode(episode_path)
    
    def _add_episode(self, episode_path):
        raw_images, raw_actions, instruction = self._load_episode(episode_path)
        indexs = list(range(len(raw_images[self.config.rgb_names[0]])))
        
        for i in tqdm(indexs[:-1], desc=f'Adding episode {episode_path}'):
            states = np.concatenate([raw_actions[action_dir][i] for action_dir in self.config.action_dirs])
            actions = np.concatenate([raw_actions[action_dir][i + 1] for action_dir in self.config.action_dirs])
            if not self._check_nonoop_actions(states, actions):
                continue

            frame = {rgb_name: load_image(raw_images[rgb_name][i]) for rgb_name in self.config.rgb_names}
            frame['states'] = states
            frame['actions'] = actions
            self.dataset.add_frame(frame, task=instruction)
            
        self.dataset.save_episode()
        
    def _load_episode(self, episode_path):
        raw_images = defaultdict(list)
        for rgb_dir, rgb_name in zip(self.config.rgb_dirs, self.config.rgb_names):
            rgb_dir = os.path.join(episode_path, rgb_dir)
            for file_name in sorted(os.listdir(rgb_dir), key=lambda x: float(x[:-5])):
                image_path = os.path.join(rgb_dir, file_name)
                raw_images[rgb_name].append(image_path)
            
        raw_actions = defaultdict(list)
        for action_dir, action_keys in zip(self.config.action_dirs, self.config.action_keys_list):
            action_dir_ = os.path.join(episode_path, action_dir)
            for file_name in sorted(os.listdir(action_dir_), key=lambda x: float(x[:-5])):
                action_path = os.path.join(action_dir_, file_name)
                with open(action_path, 'r') as f:
                    action_data = json.load(f)
                action_data = np.array([action_data[key] for key in action_keys])
                raw_actions[action_dir].append(action_data)
        
        instruction_path = os.path.join(episode_path, self.config.instruction_path)
        with open(instruction_path, 'r') as f:
            instruction_data = json.load(f)
        instruction = instruction_data['instructions'][0]
        if instruction == 'null':
            instruction = self.config.default_instruction
        
        lens = []
        for rgb_name, images_list in raw_images.items():
            lens.append(len(images_list))
        for action_dir, actions_list in raw_actions.items():
            lens.append(len(actions_list))
        
        assert all(lens[0] == l for l in lens), "All lists must have the same length"
        
        return raw_images, raw_actions, instruction
    
    def _check_nonoop_actions(self, states, actions):
        return np.abs(states - actions).max() > self.config.nonoop_threshold


def main():
    config = DataConfig()
    processor = DataProcessor(config)
    processor.process_data()


if __name__ == "__main__":
    main()
