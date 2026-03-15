from rich import print
from rich.console import Console
from rich.table import Table

import glob
import os
import sys
import csv

if __name__ == '__main__':

	console = Console()
		
	table = Table()
	table.add_column("CSV Cleaner (Romanian)", style="cyan")
	table.add_row("Validates metadata.csv by checking if audio files exist.")
	table.add_row("Removes entries where the .wav file is missing.")
	table.add_row("Common tool for both TTS and STT datasets.")
	
	console.print(table)
		
	# Select a project folder
	app_folder = os.path.dirname(os.path.realpath(__file__))
	project_directories = glob.glob(os.path.join(app_folder, 'project*/'))
	
	if len(project_directories) == 0:
		console.print("There are no project directories. Create a folder starting with 'project' that contains a 'metadata.csv'. This works best using main_generate_csv.py.")
		sys.exit(0)
		
	console.print("Please select a [red]project folder[/red] by entering its id (default 0 (%s))" % project_directories[0])
	for index, pd in enumerate(project_directories):
		console.print("%d: %s" % (index, pd))
	
	in_folder_id = input()
	if not in_folder_id:
		in_folder_id = 0
	project_folder = project_directories[int(in_folder_id)]
	
	metadata_csv_file = os.path.join(project_folder, 'metadata.csv')
	if not os.path.exists(metadata_csv_file):
		console.print("Project folder does not contain a metadata.csv file. Exiting.")
		sys.exit(0)
	
	console.print("Project folder is [red]%s" % project_folder)
	
	with open(metadata_csv_file, encoding = "utf-8") as csv_file:
			csv_reader = csv.reader(csv_file, delimiter='|')
			csv_data = list(csv_reader)
			
	console.print("Found %d sentences in CSV." % len(csv_data))

	if len(csv_data) == 0:
		console.print("CSV is empty. Exiting.")
		sys.exit(0)
		
	metadata_csv_file_cleaned = os.path.join(project_folder, 'metadata_cleaned.csv')
	
	# Open with newline='' to prevent extra blank lines in Windows
	csv_out = open(metadata_csv_file_cleaned, 'w', encoding='utf-8', newline='')
	writer = csv.writer(csv_out, delimiter='|')
	
	valid_count = 0
	removed_count = 0
	
	i = 0
	while i < len(csv_data):
		wav_path = os.path.join(project_folder, csv_data[i][0])
		
		# Check if wav file exists
		if os.path.exists(wav_path):
			# Write the row exactly as is (normalization is now done in generation step)
			writer.write(csv_data[i])
			valid_count += 1
		else:
			removed_count += 1
			
		i += 1
	csv_out.close()
	
	# Backup original
	if os.path.exists(os.path.join(project_folder, 'metadata_original.csv')):
		os.remove(os.path.join(project_folder, 'metadata_original.csv'))
		
	os.rename(metadata_csv_file, os.path.join(project_folder, 'metadata_original.csv'))
	os.rename(metadata_csv_file_cleaned, metadata_csv_file)
	
	console.print(f"Done. Kept {valid_count} valid entries. Removed {removed_count} missing files.")

