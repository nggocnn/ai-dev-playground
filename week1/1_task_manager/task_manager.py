from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from pathlib import Path
import json
import sys
import textwrap

DATA_FILE = Path(__file__).parent / "tasks.json"


@dataclass
class Task:
    id: int
    description: str
    completed: bool = False

    def mark_completed(self):
        self.completed = True

    @property
    def status(self) -> str:
        return "Done" if self.completed else "Pending"

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> "Task":
        return Task(**data)


class TaskManager:
    def __init__(self, storage: Path):
        self.storage = storage
        self.tasks: List[Task] = []
        self._load()

    def _load(self):
        if self.storage.exists():
            try:
                data = json.loads(self.storage.read_text(encoding="utf-8"))
                self.tasks = [Task.from_dict(t) for t in data]
            except json.JSONDecodeError:
                print("Warning: Corrupt JSON, starting with empty task list.")
                self.tasks = []

    def _save(self):
        data = [t.to_dict() for t in self.tasks]
        self.storage.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_task(self, description: str):
        new_id = self.tasks[-1].id + 1 if self.tasks else 1
        task = Task(new_id, description.strip())
        self.tasks.append(task)
        self._save()
        print(f"Task '{task.description}' added with ID {task.id}.")

    def view_tasks(self):
        if not self.tasks:
            print("No tasks available.")
            return

        desc_header = "Description"
        max_desc_len = max(
            (len(t.description) for t in self.tasks), default=len(desc_header)
        )
        desc_w = min(max(max_desc_len, len(desc_header)), 40)
        id_w = 4
        status_header = "Status"
        status_w = max(len(status_header), len("Pending"), len("Done"))

        print(f"{'ID':<{id_w}} {desc_header:<{desc_w}} {status_header:<{status_w}}")
        print(f"{'-'*id_w} {'-'*desc_w} {'-'*status_w}")

        import textwrap

        for t in self.tasks:
            wrapped = textwrap.wrap(t.description, width=desc_w) or [""]
            for i, line in enumerate(wrapped):
                if i == 0:
                    print(f"{t.id:<{id_w}} {line:<{desc_w}} {t.status:<{status_w}}")
                else:
                    print(f"{'':<{id_w}} {line:<{desc_w}} {'':<{status_w}}")

    def complete_task(self, task_id: int):
        for t in self.tasks:
            if t.id == task_id:
                t.mark_completed()
                self._save()
                print(f"Task ID {t.id} marked as completed.")
                return
        print(f"No task found with ID {task_id}.")

    def delete_task(self, task_id: int):
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.id != task_id]
        if len(self.tasks) < before:
            self._save()
            print(f"Task ID {task_id} deleted.")
        else:
            print(f"No task found with ID {task_id}.")


def interactive_loop(manager: TaskManager):
    print("Welcome to Task Manager! Type 'help' for commands, 'exit' to quit.")
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Task Manager. Goodbye!")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if cmd == "add":
            if arg:
                manager.add_task(arg)
            else:
                print("Usage: add <description>")

        elif cmd == "view":
            manager.view_tasks()

        elif cmd == "complete":
            if arg and arg.isdigit():
                manager.complete_task(int(arg))
            else:
                print("Usage: complete <task_id>")

        elif cmd == "delete":
            if arg and arg.isdigit():
                manager.delete_task(int(arg))
            else:
                print("Usage: delete <task_id>")

        elif cmd == "help":
            print("Commands: add <desc>, view, complete <id>, delete <id>, exit")

        elif cmd == "exit":
            print("Exiting Task Manager. Goodbye!")
            break

        else:
            print(f"Unknown command: {cmd}")


def cli_mode(manager: TaskManager, args: List[str]):
    """Run commands from CLI args instead of interactive loop."""
    cmd = args[0].lower()

    if cmd == "add" and len(args) > 1:
        manager.add_task(" ".join(args[1:]))

    elif cmd == "view":
        manager.view_tasks()

    elif cmd == "complete" and len(args) > 1 and args[1].isdigit():
        manager.complete_task(int(args[1]))

    elif cmd == "delete" and len(args) > 1 and args[1].isdigit():
        manager.delete_task(int(args[1]))

    else:
        print("Invalid command or arguments.")


def run_commands(manager: TaskManager, commands: List[Tuple[str, ...]]):
    """Execute a list of command tuples like in automated demo."""
    for cmd in commands:
        action = cmd[0].lower()
        arg = cmd[1] if len(cmd) > 1 else None

        if action == "add" and arg:
            manager.add_task(arg)

        elif action == "view":
            manager.view_tasks()

        elif action == "complete" and arg and arg.isdigit():
            manager.complete_task(int(arg))

        elif action == "delete" and arg and arg.isdigit():
            manager.delete_task(int(arg))

        elif action == "exit":
            print("Exiting Task Manager. Goodbye!")
            break

        else:
            print(f"Invalid command: {cmd}")


if __name__ == "__main__":
    manager = TaskManager(DATA_FILE)

    if "--demo" in sys.argv:
        commands_to_execute = [
            ("add", "Buy groceries"),
            ("add", "Walk the dog"),
            ("view",),
            ("complete", "1"),
            ("view",),
            ("delete", "2"),
            ("view",),
            ("exit",),
        ]
        run_commands(manager, commands_to_execute)
    elif len(sys.argv) > 1:
        cli_mode(manager, [arg for arg in sys.argv[1:] if arg != "--demo"])
    else:
        interactive_loop(manager)
