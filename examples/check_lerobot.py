from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = 'Koorye/pika-tiny'
dataset = LeRobotDataset(repo_id=repo_id)
iterator = iter(dataset)