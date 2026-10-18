import trackio
from trackio.sqlite_storage import SQLiteStorage

# Delete every local trackio project (all runs, metrics and media).
projects = SQLiteStorage.get_projects()

if not projects:
    print("No trackio projects found.")

for project in projects:
    deleted = trackio.delete_project(project, force=True)
    print(f"{'Deleted' if deleted else 'Failed to delete'}: {project}")
