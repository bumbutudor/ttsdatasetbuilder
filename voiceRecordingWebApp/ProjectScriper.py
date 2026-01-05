import os

def collect_project_files(directory, base_folder, num_files=3, excluded_files=None, excluded_directories=None):
    # Set default empty lists for excluded_files and excluded_directories if None
    if excluded_files is None:
        excluded_files = []
    if excluded_directories is None:
        excluded_directories = []

    # Collect all relevant files
    relevant_files = []

    for root, dirs, files in os.walk(directory):
        # Skip directories that are in the excluded_directories list (match only directory names, not substrings)
        if any(excluded_dir in os.path.normpath(root).split(os.sep) for excluded_dir in excluded_directories):
            continue

        for file in files:
            # Check file extension, file exclusion list, and directory exclusion list
            if file.endswith((".py", ".html", ".yaml", ".css", ".js", ".txt", ".json", ".yml", ".md", "Dockerfile")) and file not in excluded_files:
                relevant_files.append(os.path.join(root, file))

    # Calculate split points based on the specified number of files
    split_points = [len(relevant_files) * i // num_files for i in range(1, num_files)]
    file_parts = [relevant_files[i:j] for i, j in zip([0] + split_points, split_points + [None])]

    # Write each part to its respective output file
    for index, file_list in enumerate(file_parts, start=1):
        output_file = os.path.join(base_folder, f"ProjectContent_Part{index}.txt")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for file_path in file_list:
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(f"--- Start of {file_path} ---\n")
                        outfile.write(infile.read())
                        outfile.write(f"\n--- End of {file_path} ---\n")
                        outfile.write("\n\n")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        print(f"Part {index} saved to {output_file}")

if __name__ == "__main__":
    # Set the project directory and base folder for output files
    project_directory = "D:\\Disk D 13-09-25\\CS\\AI Guguta\\ttsdatasetbuilder\\voiceRecordingWebApp"
    base_folder = "D:\\Disk D 13-09-25\\CS\\AI Guguta\\ttsdatasetbuilder\\voiceRecordingWebApp\\projectFilesCopy"
    os.makedirs(base_folder, exist_ok=True)  # Ensure the output folder exists

    # List of files and directories to exclude
    excluded_files = ["ProjectScriper.py"]
    excluded_directories = ["env", "node_modules", "projectFilesCopy", "docs", "data", "__pycache__"]

    # Number of output files desired
    num_files = 1

    collect_project_files(project_directory, base_folder, num_files, excluded_files, excluded_directories)
    print(f"Project files have been split into {num_files} parts in folder {base_folder}.")