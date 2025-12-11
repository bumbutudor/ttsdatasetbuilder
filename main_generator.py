from rich import print
from rich.console import Console
import glob
import os
import time
import pyaudio
import keyboard
from rich.progress import Progress
from rich.progress import track
from rich.table import Table
import wave
from datetime import datetime
import librosa
import struct
import numpy as np
import soundfile
import sys
import csv


# Audio processing
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1
frame_rate = 44100
timeout = 60

def trim_silence_tts(wav_path):
	# Strict trimming for TTS
	# Load with librosa
	y, sr = librosa.load(wav_path, sr=None)
	# Trim silence (top_db=30 is standard for speech)
	yt, _ = librosa.effects.trim(y, top_db=30)
	# Add short padding (0.1s)
	pad_len = int(sr * 0.1)
	yt = np.pad(yt, (pad_len, pad_len), mode='constant')
	soundfile.write(wav_path, yt, sr)

def trim_silence_stt(wav_path):
	# Relaxed trimming for STT
	y, sr = librosa.load(wav_path, sr=None)
	# Less aggressive trim (top_db=20) or just trim edges
	yt, _ = librosa.effects.trim(y, top_db=20)
	# Add generous padding (0.5s)
	pad_len = int(sr * 0.5)
	yt = np.pad(yt, (pad_len, pad_len), mode='constant')
	soundfile.write(wav_path, yt, sr)

if __name__ == '__main__':

	console = Console()
		
	table = Table()
	table.add_column("TSS/STT Dataset Creator (Romanian)", style="cyan")
	table.add_row("2021 - padmalcom")
	table.add_row("Adapted for Romanian language")
	table.add_row("Supports TTS and STT modes")
	
	console.print(table)
	
	console.print("\nSelect Mode:")
	console.print("1. [cyan]TTS[/cyan] (Strict trimming, 0.1s silence)")
	console.print("2. [magenta]STT[/magenta] (Relaxed trimming, 0.5s silence)")
	mode_in = input() or "1"
	is_stt = mode_in == "2"
	
	console.print("\nPlease select your [red]microphone[/red] (enter the device number).")
	# Initialisiere pyaudio
	pyaudio = pyaudio.PyAudio()
	
	# select input device
	info = pyaudio.get_host_api_info_by_index(0)
	numdevices = info.get('deviceCount')
	for i in range(0, numdevices):
		if (pyaudio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
			print ("Input Device id ", i, " - ", pyaudio.get_device_info_by_host_api_device_index(0, i).get('name'))
	in_mic_id = int(input())
	console.print("You have selected [red]%s[/red] as input device." % pyaudio.get_device_info_by_host_api_device_index(0, in_mic_id).get('name'))
	
	
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
	
	console.print("wavs will be saved in [red]%s" % project_folder)
	
	# 0.1 select languages
	console.print("Language is set to [red]ro[/red] (Romanian).")
	in_lang = 'ro'
				
	# load csv
	with open(metadata_csv_file, encoding = "utf-8") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter='|')
		csv_data = list(csv_reader)
		
	console.print("Found %d sentences." % len(csv_data))

	if len(csv_data) == 0:
		console.print("You need to add some sentences to your text files first. Exiting.")
		sys.exit(0)
	
	# 1.1 Hit Space to skip to save+next, hit backspace to discard
	console.print("[green]n[/green] = next sentence, [yellow]d[/yellow] = discard and repeat last recording, [red]e[/red] = exit recording.")
	
	console.print("Ready?", style="green")
	input("Press Enter to start")
	
	i = 0
	cancelled = False
	while i < len(csv_data):
		console.clear()
		
		current_sentence = csv_data[i][1]
		current_sentence = current_sentence.replace("\n", " ")
		current_sentence = current_sentence.replace("\t", " ")
		
		# If this wav file has been recorded before, continue
		wav_file_name = csv_data[i][0]
		if os.path.exists(os.path.join(project_folder, wav_file_name)):
			i += 1
			continue
		
		console.print("\n\n" + current_sentence + "\n\n", style = "black on white", justify="center")
		console.print("(%d/%d) [green]n[/green] = next sentence, [yellow]d[/yellow] = discard and repeat last recording, [blue]s[/blue] = skip, [red]e[/red] = exit recording." % ((i+1), len(csv_data)))
		
		start_time = time.time()
		current_time = time.time()
		frames = []
		stream = pyaudio.open(input_device_index = in_mic_id, format=sample_format, channels=channels, rate=frame_rate, frames_per_buffer=chunk, input=True)
		
		with Progress() as recording_progress:
			recording_task = recording_progress.add_task("[red]Recording...", total=timeout)
						
			#while (current_time - start_time) < timeout:
			while not recording_progress.finished:
				recording_progress.update(recording_task, completed = current_time - start_time)
							
				data = stream.read(chunk)
				frames.append(data)
				current_time = time.time()

					
				if keyboard.is_pressed('n'):
					while keyboard.is_pressed('n'):
						time.sleep(0.1)
		
					# Write the wav file
					data = stream.read(chunk)
					frames.append(data)
					
					# Check duration
					recorded_seconds = (len(frames) * chunk) / frame_rate
					limit = 30 if is_stt else 10
					
					if recorded_seconds > limit:
						console.print(f"\n[bold red]Warning:[/bold red] Recording is {recorded_seconds:.2f}s long (Limit: {limit}s).")
						console.print("Press: [green]k[/green] to keep, [yellow]d[/yellow] to discard/retry, [blue]s[/blue] to skip.")
						
						action = None
						while action is None:
							if keyboard.is_pressed('k'):
								while keyboard.is_pressed('k'): time.sleep(0.1)
								action = 'keep'
							elif keyboard.is_pressed('d'):
								while keyboard.is_pressed('d'): time.sleep(0.1)
								action = 'discard'
							elif keyboard.is_pressed('s'):
								while keyboard.is_pressed('s'): time.sleep(0.1)
								action = 'skip'
							time.sleep(0.05)
						
						if action == 'discard':
							stream.close()
							break
						elif action == 'skip':
							stream.close()
							i += 1
							break

					stream.close()
					wf = wave.open(os.path.join(project_folder, wav_file_name), 'wb')
					wf.setnchannels(channels)
					wf.setsampwidth(pyaudio.get_sample_size(sample_format))
					wf.setframerate(frame_rate)										
					wf.writeframes(b''.join(frames))
					wf.close()
					
					# trim silence based on mode
					wav_path = str(os.path.join(project_folder, wav_file_name))
					if is_stt:
						trim_silence_stt(wav_path)
					else:
						trim_silence_tts(wav_path)
					
					i += 1
					break
				elif keyboard.is_pressed("s"):
					while keyboard.is_pressed('s'):
						time.sleep(0.1)				
					i += 1
					stream.close()
					break
				elif keyboard.is_pressed("d"):
					while keyboard.is_pressed('d'):
						time.sleep(0.1)			
					stream.close()
					break
				elif keyboard.is_pressed("e"):
					while keyboard.is_pressed('e'):
						time.sleep(0.1)
					stream.close()
					cancelled = True
					break
					
			if cancelled:
				break
		

	pyaudio.terminate()