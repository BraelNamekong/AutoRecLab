#Tracks the differents and statistics from node to node

from treesearch.node import Node
from utils.log import _ROOT_LOGGER
import os
import ast
import subprocess
from pathlib import Path

logger = _ROOT_LOGGER.getChild("statistics")


class StatisticNode:

    def __init__(self):
        self.out_dir : str = None
        self.checkpoint_dir : str = None
        self.last_node : Node = None

        self.id : str = ""
        self.position : int = 0

        self.score : float = 0.0
        self.is_buggy : bool = False
        self.is_satisfactory : bool = False
        self.exec_time : float = 0.0

        self.loc = 0
        self.empty_lines = 0
        self.comment_lines = 0
        self.total_characters = 0
        self.libraries_imported = []
        self.sum_libraries_imported = 0
        self.sum_functions = 0
        self.sum_classes = 0
        self.avg_ags_per_function = 0.0
        self.variable_assignments = 0
        self.sum_loops = 0
        self.sum_conditions = 0
        self.insertions = 0
        self.deletions = 0

    def analyze_code(self):
        code_file = os.path.join(self.checkpoint_dir, self.id, "code.py")
        try:
            logger.debug("Analyzing code for statistic from node " + self.id + " at file: " + code_file)

            with open(code_file, "r", encoding="utf-8") as f:
                code_content = f.read()

            lines = code_content.splitlines()
            self.loc = len(lines)
            self.total_characters = len(code_content)
            self.empty_lines = sum(1 for l in lines if not l.strip())
            self.comment_lines = sum(1 for l in lines if l.strip().startswith("#"))

            ast_tree = ast.parse(code_content)
            imports = set()
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
            self.libraries_imported = list(imports)
            self.sum_libraries_imported = len(self.libraries_imported)

            sum_functions   = [n for n in ast.walk(ast_tree) if isinstance(n, ast.FunctionDef)]
            sum_classes     = [n for n in ast.walk(ast_tree) if isinstance(n, ast.ClassDef)]
            self.avg_ags_per_function    = (sum(len(f.args.args) for f in sum_functions) / len(sum_functions) if sum_functions else 0)
            self.variable_assignments = sum(1 for n in ast.walk(ast_tree) if isinstance(n, ast.Assign))
            self.sum_loops       = sum(1 for n in ast.walk(ast_tree) if isinstance(n, (ast.For, ast.While)))
            self.sum_conditions  = sum(1 for n in ast.walk(ast_tree) if isinstance(n, ast.If))
            self.sum_functions   = len(sum_functions)
            self.sum_classes     = len(sum_classes)

            self.insertions = self.loc
            self.deletions = 0
            if self.last_node is not None:
                code_file_last = os.path.join(self.checkpoint_dir, self.last_node.id, "code.py")
                self.insertions, self.deletions = self.git_diff_stat(Path(code_file_last), Path(code_file))

        except Exception as e:
            logger.warning("Could not analyze code for statistic from node " + self.id + ": " + str(e))
            return

    def normalize_file(self, path: Path):  
        text = path.read_text().rstrip("\n")
        path.write_text(text + "\n")
        
    def git_diff_stat(self, file1: Path, file2: Path):
        self.normalize_file(file1)
        self.normalize_file(file2)

        proc = subprocess.run(
            ["git", "diff", "--no-index", "--numstat", file1, file2],
            capture_output=True,
            text=True,
        )

        s = proc.stdout.split("\t")
        if len(s) < 2:
            return 0, 0
        insertions = int(s[0])
        deletions = int(s[1])
        return insertions, deletions


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
            f.write("Lines of Code: " + str(self.loc) + "\n")
            f.write("Empty Lines: " + str(self.empty_lines) + "\n")
            f.write("Comment Lines: " + str(self.comment_lines) + "\n")
            f.write("Total Characters: " + str(self.total_characters) + "\n")
            f.write("Libraries Imported: " + ", ".join(self.libraries_imported) + "\n")
            f.write("Count of Libraries Imported: " + str(self.sum_libraries_imported) + "\n")
            f.write("Functions: " + str(self.sum_functions) + "\n")
            f.write("Classes: " + str(self.sum_classes) + "\n")
            f.write("Average Arguments per Function: " + str(self.avg_ags_per_function) + "\n")
            f.write("Variable Assignments: " + str(self.variable_assignments) + "\n")
            f.write("Loops: " + str(self.sum_loops) + "\n")
            f.write("Conditions: " + str(self.sum_conditions) + "\n")
            f.write("Insertions: " + str(self.insertions) + "\n")
            f.write("Deletions: " + str(self.deletions) + "\n")

        logger.debug("Saved statistic for node " + self.id + " to file: " + file_path)



class StatisticsTracker:

    def __init__(self):
        self.out_dir = None
        self.checkpoint_dir = None
        self.nodes_ordered = []

    def set_out_dir(self, out_dir):
        self.checkpoint_dir = os.path.join(out_dir, "checkpoint")

        stats_folder = os.path.join(out_dir, "statistics")
        os.makedirs(stats_folder, exist_ok=True)
        self.out_dir = stats_folder


    def add_node(self, node: Node):
        new_node = StatisticNode()
        new_node.out_dir = self.out_dir
        new_node.checkpoint_dir = self.checkpoint_dir
        new_node.id = node.id
        new_node.score = node.score.score
        new_node.is_buggy = node.is_buggy
        new_node.is_satisfactory = node.score.is_satisfactory
        new_node.exec_time = node.exec_time
        new_node.position = len(self.nodes_ordered)

        if(len(self.nodes_ordered) > 0):
            new_node.last_node = self.nodes_ordered[-1]
        
        self.nodes_ordered.append(new_node)

        new_node.analyze_code()
        new_node.save_to_file()


_STATISTICS_TRACKER = StatisticsTracker()
def get_statistics_tracker():
    return _STATISTICS_TRACKER