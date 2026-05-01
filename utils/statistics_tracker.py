#Tracks the differents and statistics from node to node

from treesearch.node import Node
from utils.log import _ROOT_LOGGER
import os

logger = _ROOT_LOGGER.getChild("statistics")


class StatisticNode:

    def __init__(self):
        self.id : str = ""
        self.score : float = 0.0
        self.is_buggy : bool = False
        self.is_satisfactory : bool = False
        self.exec_time : float = 0.0
        self.position : int = 0

        self.out_dir : str = None

    def save_to_file(self):
        if self.out_dir is None:
            return
        file_path = os.path.join(self.out_dir, str(self.position) + "_" + self.id + ".txt")
        with open(file_path, "w") as f:
            f.write("ID: " + self.id + "\n")
            f.write("Position: " + str(self.position) + "\n")
            f.write("Score: " + str(self.score) + "\n")
            f.write("Is Buggy: " + str(self.is_buggy) + "\n")
            f.write("Is Satisfactory: " + str(self.is_satisfactory) + "\n")
            f.write("Execution Time: " + str(self.exec_time) + "\n")

        logger.debug("Saved statistic for node " + self.id + " to file: " + file_path)

class StatisticsTracker:

    def __init__(self):
        self.out_dir = None
        self.nodes_ordered = []

    def set_out_dir(self, out_dir):
        stats_folder = os.path.join(out_dir, "statistics")
        os.makedirs(stats_folder, exist_ok=True)
        self.out_dir = stats_folder

    def add_node(self, node: Node):
        new_node = StatisticNode()
        new_node.out_dir = self.out_dir
        new_node.id = node.id
        new_node.score = node.score.score
        new_node.is_buggy = node.is_buggy
        new_node.is_satisfactory = node.score.is_satisfactory
        new_node.exec_time = node.exec_time
        new_node.position = len(self.nodes_ordered)
        
        self.nodes_ordered.append(new_node)
        new_node.save_to_file()



_STATISTICS_TRACKER = StatisticsTracker()
def get_statistics_tracker():
    return _STATISTICS_TRACKER