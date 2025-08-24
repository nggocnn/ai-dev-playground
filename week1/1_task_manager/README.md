# CLI Task Manager (OOP, Persistent, Auto-Size Table, Demo Mode)

A simple task manager with persistent storage, interactive CLI loop, one-off commands, and demo mode.

## Features
- **OOP design**: `Task` and `TaskManager`
- **Persistent storage** in `tasks.json`
- **Interactive CLI** with `input()`
- **One-off CLI commands** (`add`, `view`, `complete`, `delete`)
- **Demo mode** (`--demo`) for automated runs
- **Table view** with auto-sized & wrapped description column

---

## 1. Approach
The application follows an **Object-Oriented Programming (OOP)** approach:
- **Task** class: Represents a task with `id`, `description`, and `completed` status. Includes helper methods for marking complete and serializing to/from dictionaries.
- **TaskManager** class: Manages a list of tasks, providing methods for adding, viewing, completing, and deleting tasks. Handles **JSON persistence** automatically by loading at startup and saving after every change.
- **Interactive CLI loop**: Default mode using `input()` to process commands until the user types `exit`.
- **CLI command mode**: Runs a single command from arguments (e.g., `add`, `view`, `complete`, `delete`).
- **Demo mode**: Uses a fixed list of commands for demonstration or testing purposes.

The **view** method uses dynamic column sizing and `textwrap` to wrap long descriptions without breaking table alignment.

---

## 2. Challenges & Solutions
1. **Persistent storage without external dependencies**  
   - Used built-in `json` module to read/write tasks to a local `tasks.json` file.
2. **Maintaining unique task IDs after deletions**  
   - IDs increment from the last task's ID, ensuring no conflicts even if tasks are deleted.
3. **Formatting a clean table output with long descriptions**  
   - Measured the longest description and capped width at 40 characters for readability.
   - Used `textwrap.wrap()` to split long descriptions into multiple lines while preserving table structure.
4. **Graceful error handling**  
   - Wrapped file reading in `try/except` for JSON decode errors and provided warnings.
   - Checked input arguments for commands like `complete` and `delete` to ensure valid integers.

---

## 3. Enhancements & Extensions
Implemented enhancements:
- **Auto-size and wrapping view** for better readability.
- **Persistent storage** loaded automatically and saved after each change.
- **Demo mode** (`--demo`) to auto-run a command sequence.

---

## 4. Input Validation
The program includes basic input validation:
- **Description**: Ensures it's non-empty before adding.
- **Task ID**: Ensures it's a positive integer for `complete` and `delete` commands.
- **Invalid commands**: Shows a usage hint if a command is unknown or missing arguments.
- **JSON corruption**: Displays a warning and starts with an empty task list.

---

## 5. Usage

### Interactive mode (default)
```bash
python task_manager.py
```

### One-off CLI commands
```bash
python task_manager.py add "Test task"
python task_manager.py view
python task_manager.py complete 1
python task_manager.py delete 2
```

### Demo mode
```bash
python task_manager.py --demo
```

---

## File structure
```
task_manager.py  # main program
tasks.json       # auto-created for persistent storage
```

---

## Notes
- Requires Python 3.7+
- Data saved after every change automatically
