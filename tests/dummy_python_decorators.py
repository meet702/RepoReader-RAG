from dataclasses import dataclass
from typing import List

@dataclass
class TaskModel:
    id: int
    title: str
    completed: bool

    @staticmethod
    def create_empty():
        return TaskModel(0, "", False)
    
    @property
    def is_done(self) -> bool:
        return self.completed

@app.route("/api/tasks", methods=["GET"])
def get_all_tasks():
    pass
