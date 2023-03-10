import os
if 'XDG_CACHE_HOME' not in os.environ:
	os.environ['XDG_CACHE_HOME'] = os.path.realpath(os.path.join(os.getcwd(), './models/'))

if 'TORTOISE_MODELS_DIR' not in os.environ:
	os.environ['TORTOISE_MODELS_DIR'] = os.path.realpath(os.path.join(os.getcwd(), './models/tortoise/'))

if 'TRANSFORMERS_CACHE' not in os.environ:
	os.environ['TRANSFORMERS_CACHE'] = os.path.realpath(os.path.join(os.getcwd(), './models/transformers/'))

import argparse
import time
import json
import base64
import re
import urllib.request
import signal
import gc
import subprocess
import psutil
import yaml
import hashlib

import tqdm
import torch
import torchaudio
import music_tag
import gradio as gr
import gradio.utils
import pandas as pd

from datetime import datetime
from datetime import timedelta

from tortoise.api import TextToSpeech, MODELS, get_model_path, pad_or_truncate
from tortoise.utils.audio import load_audio, load_voice, load_voices, get_voice_dir
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.device import get_device_name, set_device_name, get_device_count, get_device_vram

MODELS['dvae.pth'] = "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/3704aea61678e7e468a06d8eea121dba368a798e/.models/dvae.pth"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2"]
WHISPER_SPECIALIZED_MODELS = ["tiny.en", "base.en", "small.en", "medium.en"]
WHISPER_BACKENDS = ["openai/whisper", "lightmare/whispercpp", "m-bain/whisperx"]
VOCODERS = ['univnet', 'bigvgan_base_24khz_100band', 'bigvgan_24khz_100band']

GENERATE_SETTINGS_ARGS = None

LEARNING_RATE_SCHEMES = {"Multistep": "MultiStepLR", "Cos. Annealing": "CosineAnnealingLR_Restart"}
LEARNING_RATE_SCHEDULE = [ 9, 18, 25, 33 ]

args = None
tts = None
tts_loading = False
webui = None
voicefixer = None
whisper_model = None
training_state = None

current_voice = None

def generate(**kwargs):
	parameters = {}
	parameters.update(kwargs)

	voice = parameters['voice']
	progress = parameters['progress'] if 'progress' in parameters else None
	if parameters['seed'] == 0:
		parameters['seed'] = None

	usedSeed = parameters['seed']

	global args
	global tts

	unload_whisper()
	unload_voicefixer()

	if not tts:
		# should check if it's loading or unloaded, and load it if it's unloaded
		if tts_loading:
			raise Exception("TTS is still initializing...")
		if progress is not None:
			progress(0, "Initializing TTS...")
		load_tts()
	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	do_gc()

	voice_samples = None
	conditioning_latents =None
	sample_voice = None

	voice_cache = {}
	def fetch_voice( voice ):
		cache_key = f'{voice}:{tts.autoregressive_model_hash[:8]}'
		if cache_key in voice_cache:
			return voice_cache[cache_key]

		print(f"Loading voice: {voice} with model {tts.autoregressive_model_hash[:8]}")
		sample_voice = None
		if voice == "microphone":
			if parameters['mic_audio'] is None:
				raise Exception("Please provide audio from mic when choosing `microphone` as a voice input")
			voice_samples, conditioning_latents = [load_audio(parameters['mic_audio'], tts.input_sample_rate)], None
		elif voice == "random":
			voice_samples, conditioning_latents = None, tts.get_random_conditioning_latents()
		else:
			if progress is not None:
				progress(0, desc=f"Loading voice: {voice}")

			voice_samples, conditioning_latents = load_voice(voice, model_hash=tts.autoregressive_model_hash)
			
		if voice_samples and len(voice_samples) > 0:
			if conditioning_latents is None:
				conditioning_latents = compute_latents(voice=voice, voice_samples=voice_samples, voice_latents_chunks=parameters['voice_latents_chunks'])
				
			sample_voice = torch.cat(voice_samples, dim=-1).squeeze().cpu()
			voice_samples = None

		voice_cache[cache_key] = (voice_samples, conditioning_latents, sample_voice)
		return voice_cache[cache_key]

	def get_settings( override=None ):
		settings = {
			'temperature': float(parameters['temperature']),

			'top_p': float(parameters['top_p']),
			'diffusion_temperature': float(parameters['diffusion_temperature']),
			'length_penalty': float(parameters['length_penalty']),
			'repetition_penalty': float(parameters['repetition_penalty']),
			'cond_free_k': float(parameters['cond_free_k']),

			'num_autoregressive_samples': parameters['num_autoregressive_samples'],
			'sample_batch_size': args.sample_batch_size,
			'diffusion_iterations': parameters['diffusion_iterations'],

			'voice_samples': None,
			'conditioning_latents': None,

			'use_deterministic_seed': parameters['seed'],
			'return_deterministic_state': True,
			'k': parameters['candidates'],
			'diffusion_sampler': parameters['diffusion_sampler'],
			'breathing_room': parameters['breathing_room'],
			'progress': parameters['progress'],
			'half_p': "Half Precision" in parameters['experimentals'],
			'cond_free': "Conditioning-Free" in parameters['experimentals'],
			'cvvp_amount': parameters['cvvp_weight'],
			'autoregressive_model': args.autoregressive_model,
		}

		# could be better to just do a ternary on everything above, but i am not a professional
		selected_voice = voice
		if override is not None:
			if 'voice' in override:
				selected_voice = override['voice']

			for k in override:
				if k not in settings:
					continue
				settings[k] = override[k]

		if settings['autoregressive_model'] is not None:
			if settings['autoregressive_model'] == "auto":
				settings['autoregressive_model'] = deduce_autoregressive_model(selected_voice)
			tts.load_autoregressive_model(settings['autoregressive_model'])

		settings['voice_samples'], settings['conditioning_latents'], _ = fetch_voice(voice=selected_voice)

		# clamp it down for the insane users who want this
		# it would be wiser to enforce the sample size to the batch size, but this is what the user wants
		settings['sample_batch_size'] = args.sample_batch_size
		if not settings['sample_batch_size']:
			settings['sample_batch_size'] = tts.autoregressive_batch_size
		if settings['num_autoregressive_samples'] < settings['sample_batch_size']:
			settings['sample_batch_size'] = settings['num_autoregressive_samples']

		if settings['conditioning_latents'] is not None and len(settings['conditioning_latents']) == 2 and settings['cvvp_amount'] > 0:
			print("Requesting weighing against CVVP weight, but voice latents are missing some extra data. Please regenerate your voice latents with 'Slimmer voice latents' unchecked.")
			settings['cvvp_amount'] = 0
			
		return settings

	if not parameters['delimiter']:
		parameters['delimiter'] = "\n"
	elif parameters['delimiter'] == "\\n":
		parameters['delimiter'] = "\n"

	if parameters['delimiter'] and parameters['delimiter'] != "" and parameters['delimiter'] in parameters['text']:
		texts = parameters['text'].split(parameters['delimiter'])
	else:
		texts = split_and_recombine_text(parameters['text'])
 
	full_start_time = time.time()
 
	outdir = f"./results/{voice}/"
	os.makedirs(outdir, exist_ok=True)

	audio_cache = {}
	resample = None

	if tts.output_sample_rate != args.output_sample_rate:
		resampler = torchaudio.transforms.Resample(
			tts.output_sample_rate,
			args.output_sample_rate,
			lowpass_filter_width=16,
			rolloff=0.85,
			resampling_method="kaiser_window",
			beta=8.555504641634386,
		)

	volume_adjust = torchaudio.transforms.Vol(gain=args.output_volume, gain_type="amplitude") if args.output_volume != 1 else None

	idx = 0
	idx_cache = {}
	for i, file in enumerate(os.listdir(outdir)):
		filename = os.path.basename(file)
		extension = os.path.splitext(filename)[1]
		if extension != ".json" and extension != ".wav":
			continue
		match = re.findall(rf"^{voice}_(\d+)(?:.+?)?{extension}$", filename)
		if match and len(match) > 0:
			key = int(match[0])
			idx_cache[key] = True

	if len(idx_cache) > 0:
		keys = sorted(list(idx_cache.keys()))
		idx = keys[-1] + 1

	idx = pad(idx, 4)

	def get_name(line=0, candidate=0, combined=False):
		name = f"{idx}"
		if combined:
			name = f"{name}_combined"
		elif len(texts) > 1:
			name = f"{name}_{line}"
		if parameters['candidates'] > 1:
			name = f"{name}_{candidate}"
		return name

	def get_info( voice, settings = None, latents = True ):
		info = {}
		info.update(parameters)
		info['time'] = time.time()-full_start_time

		info['datetime'] = datetime.now().isoformat()
		info['model'] = tts.autoregressive_model_path
		info['model_hash'] = tts.autoregressive_model_hash 
		info['progress'] = None
		del info['progress']

		if info['delimiter'] == "\n":
			info['delimiter'] = "\\n"

		if settings is not None:
			for k in settings:
				if k in info:
					info[k] = settings[k]

			if 'half_p' in settings and 'cond_free' in settings:
				info['experimentals'] = []
				if settings['half_p']:
					info['experimentals'].append("Half Precision")
				if settings['cond_free']:
					info['experimentals'].append("Conditioning-Free")

		if latents and "latents" not in info:
			voice = info['voice']
			model_hash = settings["model_hash"][:8] if settings is not None and "model_hash" in settings else tts.autoregressive_model_hash[:8]

			dir = f'{get_voice_dir()}/{voice}/'
			latents_path = f'{dir}/cond_latents_{model_hash}.pth'

			if voice == "random" or voice == "microphone":
				if latents and settings is not None and settings['conditioning_latents']:
					os.makedirs(dir, exist_ok=True)
					torch.save(conditioning_latents, latents_path)

			if latents_path and os.path.exists(latents_path):
				try:
					with open(latents_path, 'rb') as f:
						info['latents'] = base64.b64encode(f.read()).decode("ascii")
				except Exception as e:
					pass

		return info

	for line, cut_text in enumerate(texts):
		if parameters['emotion'] == "Custom":
			if parameters['prompt'] and parameters['prompt'].strip() != "":
				cut_text = f"[{parameters['prompt']},] {cut_text}"
		elif parameters['emotion'] != "None" and parameters['emotion']:
			cut_text = f"[I am really {parameters['emotion'].lower()},] {cut_text}"
		
		progress.msg_prefix = f'[{str(line+1)}/{str(len(texts))}]'
		print(f"{progress.msg_prefix} Generating line: {cut_text}")
		start_time = time.time()

		# do setting editing
		match = re.findall(r'^(\{.+\}) (.+?)$', cut_text) 
		override = None
		if match and len(match) > 0:
			match = match[0]
			try:
				override = json.loads(match[0])
				cut_text = match[1].strip()
			except Exception as e:
				raise Exception("Prompt settings editing requested, but received invalid JSON")

		settings = get_settings( override=override )
		gen, additionals = tts.tts(cut_text, **settings )

		parameters['seed'] = additionals[0]
		run_time = time.time()-start_time
		print(f"Generating line took {run_time} seconds")

		if not isinstance(gen, list):
			gen = [gen]

		for j, g in enumerate(gen):
			audio = g.squeeze(0).cpu()
			name = get_name(line=line, candidate=j)

			settings['text'] = cut_text
			settings['time'] = run_time
			settings['datetime'] = datetime.now().isoformat(),
			settings['model'] = tts.autoregressive_model_path
			settings['model_hash'] = tts.autoregressive_model_hash

			audio_cache[name] = {
				'audio': audio,
				'settings': get_info(voice=override['voice'] if override and 'voice' in override else voice, settings=settings)
			}
			# save here in case some error happens mid-batch
			torchaudio.save(f'{outdir}/{voice}_{name}.wav', audio, tts.output_sample_rate)

	del gen
	do_gc()

	for k in audio_cache:
		audio = audio_cache[k]['audio']

		if resampler is not None:
			audio = resampler(audio)
		if volume_adjust is not None:
			audio = volume_adjust(audio)

		audio_cache[k]['audio'] = audio
		torchaudio.save(f'{outdir}/{voice}_{k}.wav', audio, args.output_sample_rate)

	output_voices = []
	for candidate in range(parameters['candidates']):
		if len(texts) > 1:
			audio_clips = []
			for line in range(len(texts)):
				name = get_name(line=line, candidate=candidate)
				audio = audio_cache[name]['audio']
				audio_clips.append(audio)
			
			name = get_name(candidate=candidate, combined=True)
			audio = torch.cat(audio_clips, dim=-1)
			torchaudio.save(f'{outdir}/{voice}_{name}.wav', audio, args.output_sample_rate)

			audio = audio.squeeze(0).cpu()
			audio_cache[name] = {
				'audio': audio,
				'settings': get_info(voice=voice),
				'output': True
			}
		else:
			name = get_name(candidate=candidate)
			audio_cache[name]['output'] = True


	if args.voice_fixer:
		if not voicefixer:
			progress(0, "Loading voicefix...")
			load_voicefixer()

		try:
			fixed_cache = {}
			for name in progress.tqdm(audio_cache, desc="Running voicefix..."):
				del audio_cache[name]['audio']
				if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
					continue

				path = f'{outdir}/{voice}_{name}.wav'
				fixed = f'{outdir}/{voice}_{name}_fixed.wav'
				voicefixer.restore(
					input=path,
					output=fixed,
					cuda=get_device_name() == "cuda" and args.voice_fixer_use_cuda,
					#mode=mode,
				)
				
				fixed_cache[f'{name}_fixed'] = {
					'settings': audio_cache[name]['settings'],
					'output': True
				}
				audio_cache[name]['output'] = False
			
			for name in fixed_cache:
				audio_cache[name] = fixed_cache[name]
		except Exception as e:
			print(e)
			print("\nFailed to run Voicefixer")

	for name in audio_cache:
		if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
			if args.prune_nonfinal_outputs:
				audio_cache[name]['pruned'] = True
				os.remove(f'{outdir}/{voice}_{name}.wav')
			continue

		output_voices.append(f'{outdir}/{voice}_{name}.wav')

		if not args.embed_output_metadata:
			with open(f'{outdir}/{voice}_{name}.json', 'w', encoding="utf-8") as f:
				f.write(json.dumps(audio_cache[name]['settings'], indent='\t') )

	if args.embed_output_metadata:
		for name in progress.tqdm(audio_cache, desc="Embedding metadata..."):
			if 'pruned' in audio_cache[name] and audio_cache[name]['pruned']:
				continue

			metadata = music_tag.load_file(f"{outdir}/{voice}_{name}.wav")
			metadata['lyrics'] = json.dumps(audio_cache[name]['settings'])
			metadata.save()
 
	if sample_voice is not None:
		sample_voice = (tts.input_sample_rate, sample_voice.numpy())

	info = get_info(voice=voice, latents=False)
	print(f"Generation took {info['time']} seconds, saved to '{output_voices[0]}'\n")

	info['seed'] = usedSeed
	if 'latents' in info:
		del info['latents']

	os.makedirs('./config/', exist_ok=True)
	with open(f'./config/generate.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps(info, indent='\t') )

	stats = [
		[ parameters['seed'], "{:.3f}".format(info['time']) ]
	]

	return (
		sample_voice,
		output_voices,
		stats,
	)

def cancel_generate():
	import tortoise.api
	tortoise.api.STOP_SIGNAL = True

def hash_file(path, algo="md5", buffer_size=0):
	hash = None
	if algo == "md5":
		hash = hashlib.md5()
	elif algo == "sha1":
		hash = hashlib.sha1()
	else:
		raise Exception(f'Unknown hash algorithm specified: {algo}')

	if not os.path.exists(path):
		raise Exception(f'Path not found: {path}')

	with open(path, 'rb') as f:
		if buffer_size > 0:
			while True:
				data = f.read(buffer_size)
				if not data:
					break
				hash.update(data)
		else:
			hash.update(f.read())

	return "{0}".format(hash.hexdigest())

def update_baseline_for_latents_chunks( voice ):
	global current_voice
	current_voice = voice

	path = f'{get_voice_dir()}/{voice}/'
	if not os.path.isdir(path):
		return 1

	dataset_file = f'./training/{voice}/train.txt'
	if os.path.exists(dataset_file):
		return 0 # 0 will leverage using the LJspeech dataset for computing latents

	files = os.listdir(path)
	
	total = 0
	total_duration = 0

	for file in files:
		if file[-4:] != ".wav":
			continue

		metadata = torchaudio.info(f'{path}/{file}')
		duration = metadata.num_channels * metadata.num_frames / metadata.sample_rate
		total_duration += duration
		total = total + 1


	# brain too fried to figure out a better way
	if args.autocalculate_voice_chunk_duration_size == 0:
		return int(total_duration / total) if total > 0 else 1
	return int(total_duration / args.autocalculate_voice_chunk_duration_size) if total_duration > 0 else 1

def compute_latents(voice=None, voice_samples=None, voice_latents_chunks=0, progress=None):
	global tts
	global args
	
	unload_whisper()
	unload_voicefixer()

	if not tts:
		if tts_loading:
			raise Exception("TTS is still initializing...")
		load_tts()

	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	if args.autoregressive_model == "auto":
		tts.load_autoregressive_model(deduce_autoregressive_model(voice))

	if voice:
		load_from_dataset = voice_latents_chunks == 0

		if load_from_dataset:
			dataset_path = f'./training/{voice}/train.txt'
			if not os.path.exists(dataset_path):
				load_from_dataset = False
			else:
				with open(dataset_path, 'r', encoding="utf-8") as f:
					lines = f.readlines()

				print("Leveraging LJSpeech dataset for computing latents")

				voice_samples = []
				max_length = 0
				for line in lines:
					filename = f'./training/{voice}/{line.split("|")[0]}'
					
					waveform = load_audio(filename, 22050)
					max_length = max(max_length, waveform.shape[-1])
					voice_samples.append(waveform)

				for i in range(len(voice_samples)):
					voice_samples[i] = pad_or_truncate(voice_samples[i], max_length)

				voice_latents_chunks = len(voice_samples)
		if not load_from_dataset:
			voice_samples, _ = load_voice(voice, load_latents=False)

	if voice_samples is None:
		return

	conditioning_latents = tts.get_conditioning_latents(voice_samples, return_mels=not args.latents_lean_and_mean, slices=voice_latents_chunks, force_cpu=args.force_cpu_for_conditioning_latents, progress=progress)

	if len(conditioning_latents) == 4:
		conditioning_latents = (conditioning_latents[0], conditioning_latents[1], conditioning_latents[2], None)
	
	outfile = f'{get_voice_dir()}/{voice}/cond_latents_{tts.autoregressive_model_hash[:8]}.pth'
	torch.save(conditioning_latents, outfile)
	print(f'Saved voice latents: {outfile}')

	return conditioning_latents

# superfluous, but it cleans up some things
class TrainingState():
	def __init__(self, config_path, keep_x_past_checkpoints=0, start=True):
		# parse config to get its iteration
		with open(config_path, 'r') as file:
			self.config = yaml.safe_load(file)

		gpus = self.config["gpus"]

		self.killed = False

		self.dataset_dir = f"./training/{self.config['name']}/finetune/"
		self.batch_size = self.config['datasets']['train']['batch_size']
		self.dataset_path = self.config['datasets']['train']['path']
		with open(self.dataset_path, 'r', encoding="utf-8") as f:
			self.dataset_size = len(f.readlines())

		self.it = 0
		self.its = self.config['train']['niter']

		self.epoch = 0
		self.epochs = int(self.its*self.batch_size/self.dataset_size)

		self.checkpoint = 0
		self.checkpoints = int(self.its / self.config['logger']['save_checkpoint_freq'])

		self.buffer = []

		self.open_state = False
		self.training_started = False

		self.info = {}

		self.epoch_rate = ""
		self.epoch_time_start = 0
		self.epoch_time_end = 0
		self.epoch_time_deltas = 0
		self.epoch_taken = 0
		
		self.it_rate = ""
		self.it_time_start = 0
		self.it_time_end = 0
		self.it_time_deltas = 0
		self.it_taken = 0
		self.last_step = 0

		self.eta = "?"
		self.eta_hhmmss = "?"

		self.nan_detected = False

		self.last_info_check_at = 0
		self.statistics = {
			'loss': [],
			'lr': [],
		}
		self.losses = []
		self.metrics = {
			'step': "",
			'rate': "",
			'loss': "",
		}

		self.loss_milestones = [ 1.0, 0.15, 0.05 ]

		self.load_statistics()
		if keep_x_past_checkpoints > 0:
			self.cleanup_old(keep=keep_x_past_checkpoints)
		if start:
			self.spawn_process(config_path=config_path, gpus=gpus)

	def spawn_process(self, config_path, gpus=1):
		self.cmd = ['train.bat', config_path] if os.name == "nt" else ['./train.sh', str(int(gpus)), config_path]

		print("Spawning process: ", " ".join(self.cmd))
		self.process = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

	def load_statistics(self, update=False):
		if not os.path.isdir(f'{self.dataset_dir}/tb_logger/'):
			return
		try:
			from tensorboard.backend.event_processing import event_accumulator
			use_tensorboard = True
		except Exception as e:
			use_tensorboard = False

		keys = ['loss_text_ce', 'loss_mel_ce', 'loss_gpt_total', 'val_loss_text_ce', 'val_loss_mel_ce', 'learning_rate_gpt_0']
		infos = {}
		highest_step = self.last_info_check_at

		if not update:
			self.statistics['loss'] = []
			self.statistics['lr'] = []

		logs = sorted([f'{self.dataset_dir}/tb_logger/{d}' for d in os.listdir(f'{self.dataset_dir}/tb_logger/') if d[:6] == "events" ])
		if update:
			logs = [logs[-1]]

		for log in logs:
				ea = event_accumulator.EventAccumulator(log, size_guidance={event_accumulator.SCALARS: 0})
				ea.Reload()

				scalars = ea.Tags()['scalars']

				for key in keys:
					if key not in scalars:
						continue

					try:
						scalar = ea.Scalars(key)
						for s in scalar:
							if update and s.step <= self.last_info_check_at:
								continue
							highest_step = max( highest_step, s.step )
							target = 'lr' if key == "learning_rate_gpt_0" else 'loss'
							self.statistics[target].append( { "step": s.step, "value": s.value, "type": key } )
							if key == 'loss_gpt_total':
								self.losses.append( { "step": s.step, "value": s.value, "type": key } )
					except Exception as e:
						pass

		self.last_info_check_at = highest_step

	def cleanup_old(self, keep=2):
		if keep <= 0:
			return

		if not os.path.isdir(self.dataset_dir):
			return
			
		models = sorted([ int(d[:-8]) for d in os.listdir(f'{self.dataset_dir}/models/') if d[-8:] == "_gpt.pth" ])
		states = sorted([ int(d[:-6]) for d in os.listdir(f'{self.dataset_dir}/training_state/') if d[-6:] == ".state" ])

		remove_models = models[:-keep]
		remove_states = states[:-keep]

		for d in remove_models:
			path = f'{self.dataset_dir}/models/{d}_gpt.pth'
			print("Removing", path)
			os.remove(path)
		for d in remove_states:
			path = f'{self.dataset_dir}/training_state/{d}.state'
			print("Removing", path)
			os.remove(path)

	def parse(self, line, verbose=False, keep_x_past_checkpoints=0, buffer_size=8, progress=None ):
		self.buffer.append(f'{line}')

		should_return = False
		percent = 0
		message = None

		# rip out iteration info
		if not self.training_started:
			if line.find('Start training from epoch') >= 0:
				self.it_time_start = time.time()
				self.epoch_time_start = time.time()
				self.training_started = True # could just leverage the above variable, but this is python, and there's no point in these aggressive microoptimizations
				should_return = True

				match = re.findall(r'epoch: ([\d,]+)', line)
				if match and len(match) > 0:
					self.epoch = int(match[0].replace(",", ""))
				match = re.findall(r'iter: ([\d,]+)', line)
				if match and len(match) > 0:
					self.it = int(match[0].replace(",", ""))

				self.checkpoints = int((self.its - self.it) / self.config['logger']['save_checkpoint_freq'])
		else:
			lapsed = False

			message = None
			if line.find('INFO: [epoch:') >= 0:
				info_line = line.split("INFO:")[-1]
				# to-do, actually validate this works, and probably kill training when it's found, the model's dead by this point
				if ': nan' in info_line and not self.nan_detected:
					self.nan_detected = self.it

				# easily rip out our stats...
				match = re.findall(r'\b([a-z_0-9]+?)\b: *?([0-9]\.[0-9]+?e[+-]\d+|[\d,]+)\b', info_line)
				if match and len(match) > 0:
					for k, v in match:
						self.info[k] = float(v.replace(",", ""))

				self.load_statistics(update=True)
				should_return = True

				if 'epoch' in self.info:
					self.epoch = int(self.info['epoch'])
				if 'iter' in self.info:
					self.it = int(self.info['iter'])

			elif line.find('Saving models and training states') >= 0:
				self.checkpoint = self.checkpoint + 1

				percent = self.checkpoint / float(self.checkpoints)
				message = f'[{self.checkpoint}/{self.checkpoints}] Saving checkpoint...'
				if progress is not None:
					progress(percent, message)

				print(f'{"{:.3f}".format(percent*100)}% {message}')
				self.buffer.append(f'{"{:.3f}".format(percent*100)}% {message}')

				self.cleanup_old(keep=keep_x_past_checkpoints)

			if line.find('%|') > 0:
				match = re.findall(r'(\d+)%\|(.+?)\| (\d+|\?)\/(\d+|\?) \[(.+?)<(.+?), +(.+?)\]', line)
				if match and len(match) > 0:
					match = match[0]
					per_cent = int(match[0])/100.0
					progressbar = match[1]
					step = int(match[2])
					steps = int(match[3])
					elapsed = match[4]
					until = match[5]
					rate = match[6]

					last_step = self.last_step
					self.last_step = step
					if last_step < step:
						self.it = self.it + (step - last_step)

					if last_step == step and step == steps:
						lapsed = True

					self.it_time_end = time.time()
					self.it_time_delta = self.it_time_end-self.it_time_start
					self.it_time_start = time.time()
					self.it_taken = self.it_taken + 1
					if self.it_time_delta:
						try:
							rate = f'{"{:.3f}".format(self.it_time_delta)}s/it' if self.it_time_delta >= 1 or self.it_time_delta == 0 else f'{"{:.3f}".format(1/self.it_time_delta)}it/s'
							self.it_rate = rate
						except Exception as e:
							pass

					self.metrics['step'] = [f"{self.epoch}/{self.epochs}"]
					if self.epochs != self.its:
						self.metrics['step'].append(f"{self.it}/{self.its}")
					if steps > 1:
						self.metrics['step'].append(f"{step}/{steps}")
					self.metrics['step'] = ", ".join(self.metrics['step'])

			if lapsed:
				self.epoch = self.epoch + 1
				self.it = int(self.epoch * (self.dataset_size / self.batch_size))

				self.epoch_time_end = time.time()
				self.epoch_time_delta = self.epoch_time_end-self.epoch_time_start
				self.epoch_time_start = time.time()
				try:
					self.epoch_rate = f'{"{:.3f}".format(self.epoch_time_delta)}s/epoch' if self.epoch_time_delta >= 1 or self.epoch_time_delta == 0 else f'{"{:.3f}".format(1/self.epoch_time_delta)}epoch/s' # I doubt anyone will have it/s rates, but its here
				except Exception as e:
					pass
				
				#self.eta = (self.epochs - self.epoch) * self.epoch_time_delta
				self.epoch_time_deltas = self.epoch_time_deltas + self.epoch_time_delta
				self.epoch_taken = self.epoch_taken + 1
				self.eta = (self.epochs - self.epoch) * (self.epoch_time_deltas / self.epoch_taken)
				try:
					eta = str(timedelta(seconds=int(self.eta)))
					self.eta_hhmmss = eta
				except Exception as e:
					pass

			self.metrics['rate'] = []
			if self.epoch_rate:
				self.metrics['rate'].append(self.epoch_rate)
			if self.it_rate and self.epoch_rate != self.it_rate:
				self.metrics['rate'].append(self.it_rate)
			self.metrics['rate'] = ", ".join(self.metrics['rate'])

			eta_hhmmss = "?"
			if self.eta_hhmmss:
				eta_hhmmss = self.eta_hhmmss
			else:
				try:
					eta = (self.its - self.it) * (self.it_time_deltas / self.it_taken)
					eta = str(timedelta(seconds=int(eta)))
					eta_hhmmss = eta
				except Exception as e:
					pass
			
			self.metrics['loss'] = []

			if 'learning_rate_gpt_0' in self.info:
				self.metrics['loss'].append(f'LR: {"{:.3e}".format(self.info["learning_rate_gpt_0"])}')

			if len(self.losses) > 0:
				self.metrics['loss'].append(f'Loss: {"{:.3f}".format(self.losses[-1]["value"])}')

			if len(self.losses) >= 2:
				# """riemann sum""" but not really as this is for derivatives and not integrals
				deriv = 0
				accum_length = len(self.losses)//2 # i *guess* this is fine when you think about it
				loss_value = self.losses[-1]["value"]

				for i in range(accum_length):
					d1_loss = self.losses[accum_length-i-1]["value"]
					d2_loss = self.losses[accum_length-i-2]["value"]
					dloss = (d2_loss - d1_loss)

					d1_step = self.losses[accum_length-i-1]["step"]
					d2_step = self.losses[accum_length-i-2]["step"]
					dstep = (d2_step - d1_step)

					if dstep == 0:
						continue
			
					inst_deriv = dloss / dstep
					deriv += inst_deriv

				deriv = deriv / accum_length

				if deriv != 0: # dloss < 0:
					next_milestone = None
					for milestone in self.loss_milestones:
						if loss_value > milestone:
							next_milestone = milestone
							break
							
					if next_milestone:
						# tfw can do simple calculus but not basic algebra in my head
						est_its = (next_milestone - loss_value) / deriv
						if est_its >= 0:
							self.metrics['loss'].append(f'Est. milestone {next_milestone} in: {int(est_its)}its')
					else:
						est_loss = inst_deriv * (self.its - self.it) + loss_value
						if est_loss >= 0:
							self.metrics['loss'].append(f'Est. final loss: {"{:.3f}".format(est_loss)}')

			self.metrics['loss'] = ", ".join(self.metrics['loss'])

			message = f"[{self.metrics['step']}] [{self.metrics['rate']}] [ETA: {eta_hhmmss}]\n[{self.metrics['loss']}]"
			if self.nan_detected:
				message = f"[!NaN DETECTED! {self.nan_detected}] {message}"

			if message:
				percent = self.it / float(self.its) # self.epoch / float(self.epochs)
				if progress is not None:
					progress(percent, message)

				self.buffer.append(f'[{"{:.3f}".format(percent*100)}%] {message}')

		if verbose and not self.training_started:
			should_return = True

		self.buffer = self.buffer[-buffer_size:]
		
		result = None
		if should_return:
			result = "".join(self.buffer) if not self.training_started else message

		return (
			result,
			percent,
			message,
		)

try:
	import altair as alt
	alt.data_transformers.enable('default', max_rows=None)
except Exception as e:
	print(e)
	pass

def run_training(config_path, verbose=False, keep_x_past_checkpoints=0, progress=gr.Progress(track_tqdm=True)):
	global training_state
	if training_state and training_state.process:
		return "Training already in progress"


	# ensure we have the dvae.pth
	get_model_path('dvae.pth')
	
	# I don't know if this is still necessary, as it was bitching at me for not doing this, despite it being in a separate process
	torch.multiprocessing.freeze_support()

	unload_tts()
	unload_whisper()
	unload_voicefixer()

	training_state = TrainingState(config_path=config_path, keep_x_past_checkpoints=keep_x_past_checkpoints)

	for line in iter(training_state.process.stdout.readline, ""):
		if training_state.killed:
			return

		result, percent, message = training_state.parse( line=line, verbose=verbose, keep_x_past_checkpoints=keep_x_past_checkpoints, progress=progress )
		print(f"[Training] [{datetime.now().isoformat()}] {line[:-1]}")
		if result:
			yield result

			if progress is not None and message:
				progress(percent, message)

	if training_state:
		training_state.process.stdout.close()
		return_code = training_state.process.wait()
		training_state = None

def update_training_dataplot(config_path=None):
	global training_state
	losses = None
	lrs = None

	if not training_state:
		if config_path:
			training_state = TrainingState(config_path=config_path, start=False)
			if len(training_state.statistics['loss']) > 0:
				losses = gr.LinePlot.update(value=pd.DataFrame(training_state.statistics['loss']), x_lim=[0,training_state.its], x="step", y="value", title="Training Metrics", color="type", tooltip=['step', 'value', 'type'], width=500, height=350,)
			if len(training_state.statistics['lr']) > 0:
				lrs = gr.LinePlot.update(value=pd.DataFrame(training_state.statistics['lr']), x_lim=[0,training_state.its], x="step", y="value", title="Training Metrics", color="type", tooltip=['step', 'value', 'type'], width=500, height=350,)
			del training_state
			training_state = None
	else:
		training_state.load_statistics()
		if len(training_state.statistics['loss']) > 0:
			losses = gr.LinePlot.update(value=pd.DataFrame(training_state.statistics['loss']), x_lim=[0,training_state.its], x="step", y="value", title="Training Metrics", color="type", tooltip=['step', 'value', 'type'], width=500, height=350,)
		if len(training_state.statistics['lr']) > 0:
			lrs = gr.LinePlot.update(value=pd.DataFrame(training_state.statistics['lr']), x_lim=[0,training_state.its], x="step", y="value", title="Training Metrics", color="type", tooltip=['step', 'value', 'type'], width=500, height=350,)

	return (losses, lrs)

def reconnect_training(verbose=False, progress=gr.Progress(track_tqdm=True)):
	global training_state
	if not training_state or not training_state.process:
		return "Training not in progress"

	for line in iter(training_state.process.stdout.readline, ""):
		result, percent, message = training_state.parse( line=line, verbose=verbose, progress=progress )
		print(f"[Training] [{datetime.now().isoformat()}] {line[:-1]}")
		if result:
			yield result

			if progress is not None and message:
				progress(percent, message)

def stop_training():
	global training_state
	if training_state is None:
		return "No training in progress"
	print("Killing training process...")
	training_state.killed = True

	children = []
	# wrapped in a try/catch in case for some reason this fails outside of Linux
	try:
		children = [p.info for p in psutil.process_iter(attrs=['pid', 'name', 'cmdline']) if './src/train.py' in p.info['cmdline']]
	except Exception as e:
		pass

	training_state.process.stdout.close()
	training_state.process.terminate()
	training_state.process.kill()
	return_code = training_state.process.wait()

	for p in children:
		os.kill( p['pid'], signal.SIGKILL )

	training_state = None
	print("Killed training process.")
	return f"Training cancelled: {return_code}"

def get_halfp_model_path():
	autoregressive_model_path = get_model_path('autoregressive.pth')
	return autoregressive_model_path.replace(".pth", "_half.pth")

def convert_to_halfp():
	autoregressive_model_path = get_model_path('autoregressive.pth')
	print(f'Converting model to half precision: {autoregressive_model_path}')
	model = torch.load(autoregressive_model_path)
	for k in model:
		model[k] = model[k].half()

	outfile = get_halfp_model_path()
	torch.save(model, outfile)
	print(f'Converted model to half precision: {outfile}')

def whisper_transcribe( file, language=None ):
	# shouldn't happen, but it's for safety
	if not whisper_model:
		load_whisper_model(language=language)

	if args.whisper_backend == "openai/whisper":
		if not language:
			language = None

		return whisper_model.transcribe(file, language=language)

	elif args.whisper_backend == "lightmare/whispercpp":
		res = whisper_model.transcribe(file)
		segments = whisper_model.extract_text_and_timestamps( res )

		result = {
			'segments': []
		}
		for segment in segments:
			reparsed = {
				'start': segment[0] / 100.0,
				'end': segment[1] / 100.0,
				'text': segment[2],
			}
			result['segments'].append(reparsed)

		return result

	# credit to https://git.ecker.tech/yqxtqymn for the busywork of getting this added
	elif args.whisper_backend == "m-bain/whisperx":
		import whisperx
		device = "cuda" if get_device_name() == "cuda" else "cpu"
		result = whisper_model.transcribe(file)
		model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
		result_aligned = whisperx.align(result["segments"], model_a, metadata, file, device)

		for i in range(len(result_aligned['segments'])):
			del result_aligned['segments'][i]['word-segments']
			del result_aligned['segments'][i]['char-segments']

		result['segments'] = result_aligned['segments']

		return result

def prepare_dataset( files, outdir, language=None, skip_existings=False, progress=None ):
	unload_tts()

	global whisper_model
	if whisper_model is None:
		load_whisper_model(language=language)

	os.makedirs(f'{outdir}/audio/', exist_ok=True)

	results = {}
	transcription = []
	files = sorted(files)

	previous_list = []
	if skip_existings and os.path.exists(f'{outdir}/train.txt'):
		parsed_list = []
		with open(f'{outdir}/train.txt', 'r', encoding="utf-8") as f:
			parsed_list = f.readlines()

		for line in parsed_list:
			match = re.findall(r"^(.+?)_\d+\.wav$", line.split("|")[0])

			if match is None or len(match) == 0:
				continue
			
			if match[0] not in previous_list:
				previous_list.append(f'{match[0].split("/")[-1]}.wav')

	for file in enumerate_progress(files, desc="Iterating through voice files", progress=progress):
		basename = os.path.basename(file)

		if basename in previous_list:
			print(f"Skipping already parsed file: {basename}")
			continue

		result = whisper_transcribe(file, language=language)
		results[basename] = result
		print(f"Transcribed file: {file}, {len(result['segments'])} found.")

		waveform, sampling_rate = torchaudio.load(file)
		num_channels, num_frames = waveform.shape

		idx = 0
		for segment in result['segments']: # enumerate_progress(result['segments'], desc="Segmenting voice file", progress=progress):
			start = int(segment['start'] * sampling_rate)
			end = int(segment['end'] * sampling_rate)

			sliced_waveform = waveform[:, start:end]
			sliced_name = basename.replace(".wav", f"_{pad(idx, 4)}.wav")

			if not torch.any(sliced_waveform < 0):
				print(f"Error with {sliced_name}, skipping...")
				continue

			torchaudio.save(f"{outdir}/audio/{sliced_name}", sliced_waveform, sampling_rate)

			idx = idx + 1
			line = f"audio/{sliced_name}|{segment['text'].strip()}"
			transcription.append(line)
			with open(f'{outdir}/train.txt', 'a', encoding="utf-8") as f:
				f.write(f'\n{line}')

		do_gc()
	
	with open(f'{outdir}/whisper.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps(results, indent='\t'))

	unload_whisper()

	joined = "\n".join(transcription)
	if not skip_existings:
		with open(f'{outdir}/train.txt', 'w', encoding="utf-8") as f:
			f.write(joined)

	return f"Processed dataset to: {outdir}\n{joined}"

def prepare_validation_dataset( voice, text_length ):
	indir = f'./training/{voice}/'
	infile = f'{indir}/dataset.txt'
	if not os.path.exists(infile):
		infile = f'{indir}/train.txt'
		with open(f'{indir}/train.txt', 'r', encoding="utf-8") as src:
			with open(f'{indir}/dataset.txt', 'w', encoding="utf-8") as dst:
				dst.write(src.read())

	if not os.path.exists(infile):
		raise Exception(f"Missing dataset: {infile}")

	with open(infile, 'r', encoding="utf-8") as f:
		lines = f.readlines()

	validation = []
	training = []

	for line in lines:
		split = line.split("|")
		filename = split[0]
		text = split[1]

		if len(text) < text_length:
			validation.append(line.strip())
		else:
			training.append(line.strip())

	with open(f'{indir}/train.txt', 'w', encoding="utf-8") as f:
		f.write("\n".join(training))

	with open(f'{indir}/validation.txt', 'w', encoding="utf-8") as f:
		f.write("\n".join(validation))

	msg = f"Culled {len(validation)} lines"
	print(msg)
	return msg

def calc_iterations( epochs, lines, batch_size ):
	iterations = int(epochs * lines / float(batch_size))
	return iterations

def schedule_learning_rate( iterations, schedule=LEARNING_RATE_SCHEDULE ):
	return [int(iterations * d) for d in schedule]

def optimize_training_settings( **kwargs ):
	messages = []
	settings = {}
	settings.update(kwargs)

	dataset_path = f"./training/{settings['voice']}/train.txt"
	with open(dataset_path, 'r', encoding="utf-8") as f:
		lines = len(f.readlines())

	if settings['batch_size'] > lines:
		settings['batch_size'] = lines
		messages.append(f"Batch size is larger than your dataset, clamping batch size to: {settings['batch_size']}")	

	"""
	if lines % settings['batch_size'] != 0:
		settings['batch_size'] = int(lines / settings['batch_size'])
		if settings['batch_size'] == 0:
			settings['batch_size'] = 1
		messages.append(f"Batch size not neatly divisible by dataset size, adjusting batch size to: {settings['batch_size']}")
	"""
	if settings['gradient_accumulation_size'] == 0:
		settings['gradient_accumulation_size'] = 1
	
	if settings['batch_size'] / settings['gradient_accumulation_size'] < 2:
		settings['gradient_accumulation_size'] = int(settings['batch_size'] / 2)
		if settings['gradient_accumulation_size'] == 0:
			settings['gradient_accumulation_size'] = 1

		messages.append(f"Gradient accumulation size is too large for a given batch size, clamping gradient accumulation size to: {settings['gradient_accumulation_size']}")
	"""
	elif settings['batch_size'] % settings['gradient_accumulation_size'] != 0:
		settings['gradient_accumulation_size'] = int(settings['batch_size'] / settings['gradient_accumulation_size'])
		if settings['gradient_accumulation_size'] == 0:
			settings['gradient_accumulation_size'] = 1

		messages.append(f"Batch size is not evenly divisible by the gradient accumulation size, adjusting gradient accumulation size to: {settings['gradient_accumulation_size']}")

	if settings['batch_size'] % settings['gpus'] != 0:
		settings['batch_size'] = int(settings['batch_size'] / settings['gpus'])
		if settings['batch_size'] == 0:
			settings['batch_size'] = 1
		messages.append(f"Batch size not neatly divisible by GPU count, adjusting batch size to: {settings['batch_size']}")
	"""


	def get_device_batch_size( vram ):
		DEVICE_BATCH_SIZE_MAP = [
			(32, 64), # based on my two 6800XTs, I can only really safely get a ratio of 156:2 = 78
			(16, 8), # based on an A4000, I can do a ratio of 512:64 = 8:1
			(8, 4), # interpolated
			(6, 2), # based on my 2060, it only really lets me have a batch ratio of 2:1
		]
		for k, v in DEVICE_BATCH_SIZE_MAP:
			if vram > (k-1):
				return v
		return 1

	if settings['gpus'] > get_device_count():
		settings['gpus'] = get_device_count()
		messages.append(f"GPU count exceeds defacto GPU count, clamping to: {settings['gpus']}")

	if settings['gpus'] <= 1:
		settings['gpus'] = 1
	else:
		messages.append(f"! EXPERIMENTAL ! Multi-GPU training is extremely particular, expect issues.")
	
	# assuming you have equal GPUs
	vram = get_device_vram() * settings['gpus']
	batch_ratio = int(settings['batch_size'] / settings['gradient_accumulation_size'])
	batch_cap = get_device_batch_size(vram)

	if batch_ratio > batch_cap:
		settings['gradient_accumulation_size'] = int(settings['batch_size'] / batch_cap)
		messages.append(f"Batch ratio ({batch_ratio}) is expected to exceed your VRAM capacity ({'{:.3f}'.format(vram)}GB, suggested {batch_cap} batch size cap), adjusting gradient accumulation size to: {settings['gradient_accumulation_size']}")

	iterations = calc_iterations(epochs=settings['epochs'], lines=lines, batch_size=settings['batch_size'])

	if settings['epochs'] < settings['print_rate']:
		settings['print_rate'] = settings['epochs']
		messages.append(f"Print rate is too small for the given iteration step, clamping print rate to: {settings['print_rate']}")
	
	if settings['epochs'] < settings['save_rate']:
		settings['save_rate'] = settings['epochs']
		messages.append(f"Save rate is too small for the given iteration step, clamping save rate to: {settings['save_rate']}")

	if settings['epochs'] < settings['validation_rate']:
		settings['validation_rate'] = settings['epochs']
		messages.append(f"Validation rate is too small for the given iteration step, clamping validation rate to: {settings['validation_rate']}")

	if settings['resume_state'] and not os.path.exists(settings['resume_state']):
		settings['resume_state'] = None
		messages.append("Resume path specified, but does not exist. Disabling...")

	if settings['bitsandbytes']:
		messages.append("! EXPERIMENTAL ! BitsAndBytes requested.")

	if settings['half_p']:
		if settings['bitsandbytes']:
			settings['half_p'] = False
			messages.append("Half Precision requested, but BitsAndBytes is also requested. Due to redundancies, disabling half precision...")
		else:
			messages.append("! EXPERIMENTAL ! Half Precision requested.")
			if not os.path.exists(get_halfp_model_path()):
				convert_to_halfp()	

	messages.append(f"For {settings['epochs']} epochs with {lines} lines in batches of {settings['batch_size']}, iterating for {iterations} steps ({int(iterations / settings['epochs'])} steps per epoch)")

	return settings, messages

def save_training_settings( **kwargs ):
	messages = []
	settings = {}
	settings.update(kwargs)

	outjson = f'./training/{settings["voice"]}/train.json'
	with open(outjson, 'w', encoding="utf-8") as f:
		f.write(json.dumps(settings, indent='\t') )

	settings['dataset_path'] = f"./training/{settings['voice']}/train.txt"
	settings['validation_path'] = f"./training/{settings['voice']}/validation.txt"

	with open(settings['dataset_path'], 'r', encoding="utf-8") as f:
		lines = len(f.readlines())

	if not settings['source_model'] or settings['source_model'] == "auto":
		settings['source_model'] = f"./models/tortoise/autoregressive{'_half' if settings['half_p'] else ''}.pth"

	if settings['half_p']:
		if not os.path.exists(get_halfp_model_path()):
			convert_to_halfp()

	settings['iterations'] = calc_iterations(epochs=settings['epochs'], lines=lines, batch_size=settings['batch_size'])
	messages.append(f"For {settings['epochs']} epochs with {lines} lines, iterating for {settings['iterations']} steps")

	iterations_per_epoch = settings['iterations'] / settings['epochs']

	settings['print_rate'] = int(settings['print_rate'] * iterations_per_epoch)
	settings['save_rate'] = int(settings['save_rate'] * iterations_per_epoch)
	settings['validation_rate'] = int(settings['validation_rate'] * iterations_per_epoch)

	iterations_per_epoch = int(iterations_per_epoch)
	
	if settings['print_rate'] < 1:
		settings['print_rate'] = 1
	if settings['save_rate'] < 1:
		settings['save_rate'] = 1
	if settings['validation_rate'] < 1:
		settings['validation_rate'] = 1

	settings['validation_batch_size'] = int(settings['batch_size'] / settings['gradient_accumulation_size'])

	settings['iterations'] = calc_iterations(epochs=settings['epochs'], lines=lines, batch_size=settings['batch_size'])
	if settings['iterations'] % settings['save_rate'] != 0:
		adjustment = int(settings['iterations'] / settings['save_rate']) * settings['save_rate']
		messages.append(f"Iteration rate is not evenly divisible by save rate, adjusting: {settings['iterations']} => {adjustment}")
		settings['iterations'] = adjustment

	if not os.path.exists(settings['validation_path']):
		settings['validation_enabled'] = False
		messages.append("Validation not found, disabling validation...")
	elif settings['validation_batch_size'] == 0:
		settings['validation_enabled'] = False
		messages.append("Validation batch size == 0, disabling validation...")
	else:
		settings['validation_enabled'] = True
		with open(settings['validation_path'], 'r', encoding="utf-8") as f:
			validation_lines = len(f.readlines())

		if validation_lines < settings['validation_batch_size']:
			settings['validation_batch_size'] = validation_lines
			messages.append(f"Batch size exceeds validation dataset size, clamping validation batch size to {validation_lines}")


	if settings['gpus'] > get_device_count():
		settings['gpus'] = get_device_count()

	# what an utter mistake this was
	settings['optimizer'] = 'adamw' # if settings['gpus'] == 1 else 'adamw_zero'

	if 'learning_rate_scheme' not in settings or settings['learning_rate_scheme'] not in LEARNING_RATE_SCHEMES:
		settings['learning_rate_scheme'] = "Multistep"

	settings['learning_rate_scheme'] = LEARNING_RATE_SCHEMES[settings['learning_rate_scheme']]

	learning_rate_schema = [f"default_lr_scheme: {settings['learning_rate_scheme']}"]
	if settings['learning_rate_scheme'] == "MultiStepLR":
		if not settings['learning_rate_schedule']:
			settings['learning_rate_schedule'] = LEARNING_RATE_SCHEDULE
		elif isinstance(settings['learning_rate_schedule'],str):
			settings['learning_rate_schedule'] = json.loads(settings['learning_rate_schedule'])

		settings['learning_rate_schedule'] = schedule_learning_rate( iterations_per_epoch, settings['learning_rate_schedule'] )

		learning_rate_schema.append(f"  gen_lr_steps: {settings['learning_rate_schedule']}")
		learning_rate_schema.append(f"  lr_gamma: 0.5")
	elif settings['learning_rate_scheme'] == "CosineAnnealingLR_Restart":
		epochs = settings['epochs']
		restarts = settings['learning_rate_restarts']
		restart_period = int(epochs / restarts)

		if 'learning_rate_warmup' not in settings:
			settings['learning_rate_warmup'] = 0
		if 'learning_rate_min' not in settings:
			settings['learning_rate_min'] = 1e-08

		if 'learning_rate_period' not in settings:
			settings['learning_rate_period'] = [ iterations_per_epoch * restart_period for x in range(epochs) ]

		settings['learning_rate_restarts'] = [ iterations_per_epoch * (x+1) * restart_period for x in range(restarts) ] # [52, 104, 156, 208]

		if 'learning_rate_restart_weights' not in settings:
			settings['learning_rate_restart_weights'] = [ ( restarts - x - 1 ) / restarts for x in range(restarts) ] # [.75, .5, .25, .125]
			settings['learning_rate_restart_weights'][-1] = settings['learning_rate_restart_weights'][-2] * 0.5

		learning_rate_schema.append(f"  T_period: {settings['learning_rate_period']}")
		learning_rate_schema.append(f"  warmup: {settings['learning_rate_warmup']}")
		learning_rate_schema.append(f"  eta_min: !!float {settings['learning_rate_min']}")
		learning_rate_schema.append(f"  restarts: {settings['learning_rate_restarts']}")
		learning_rate_schema.append(f"  restart_weights: {settings['learning_rate_restart_weights']}")
	settings['learning_rate_scheme'] = "\n".join(learning_rate_schema)

	if settings['resume_state']:
		settings['source_model'] = f"# pretrain_model_gpt: '{settings['source_model']}'"
		settings['resume_state'] = f"resume_state: '{settings['resume_state']}'"
	else:
		settings['source_model'] = f"pretrain_model_gpt: '{settings['source_model']}'"
		settings['resume_state'] = f"# resume_state: '{settings['resume_state']}'"

	with open(f'./models/.template.yaml', 'r', encoding="utf-8") as f:
		yaml = f.read()

	# i could just load and edit the YAML directly, but this is easier, as I don't need to bother with path traversals
	for k in settings:
		if settings[k] is None:
			continue
		yaml = yaml.replace(f"${{{k}}}", str(settings[k]))

	outyaml = f'./training/{settings["voice"]}/train.yaml'
	with open(outyaml, 'w', encoding="utf-8") as f:
		f.write(yaml)
	

	messages.append(f"Saved training output to: {outyaml}")
	return settings, messages

def import_voices(files, saveAs=None, progress=None):
	global args

	if not isinstance(files, list):
		files = [files]

	for file in enumerate_progress(files, desc="Importing voice files", progress=progress):
		j, latents = read_generate_settings(file, read_latents=True)
		
		if j is not None and saveAs is None:
			saveAs = j['voice']
		if saveAs is None or saveAs == "":
			raise Exception("Specify a voice name")

		outdir = f'{get_voice_dir()}/{saveAs}/'
		os.makedirs(outdir, exist_ok=True)

		if latents:
			print(f"Importing latents to {latents}")
			with open(f'{outdir}/cond_latents.pth', 'wb') as f:
				f.write(latents)
			latents = f'{outdir}/cond_latents.pth'
			print(f"Imported latents to {latents}")
		else:
			filename = file.name
			if filename[-4:] != ".wav":
				raise Exception("Please convert to a WAV first")

			path = f"{outdir}/{os.path.basename(filename)}"
			print(f"Importing voice to {path}")

			waveform, sampling_rate = torchaudio.load(filename)

			if args.voice_fixer:
				if not voicefixer:
					load_voicefixer()

				# resample to best bandwidth since voicefixer will do it anyways through librosa
				if sampling_rate != 44100:
					print(f"Resampling imported voice sample: {path}")
					resampler = torchaudio.transforms.Resample(
						sampling_rate,
						44100,
						lowpass_filter_width=16,
						rolloff=0.85,
						resampling_method="kaiser_window",
						beta=8.555504641634386,
					)
					waveform = resampler(waveform)
					sampling_rate = 44100

				torchaudio.save(path, waveform, sampling_rate)

				print(f"Running 'voicefixer' on voice sample: {path}")
				voicefixer.restore(
					input = path,
					output = path,
					cuda=get_device_name() == "cuda" and args.voice_fixer_use_cuda,
					#mode=mode,
				)
			else:
				torchaudio.save(path, waveform, sampling_rate)

			print(f"Imported voice to {path}")

def get_voice_list(dir=get_voice_dir(), append_defaults=False):
	defaults = [ "random", "microphone" ]
	os.makedirs(dir, exist_ok=True)
	res = sorted([d for d in os.listdir(dir) if d not in defaults and os.path.isdir(os.path.join(dir, d)) and len(os.listdir(os.path.join(dir, d))) > 0 ])
	if append_defaults:
		res = res + defaults
	return res

def get_autoregressive_models(dir="./models/finetunes/", prefixed=False):
	os.makedirs(dir, exist_ok=True)
	base = [get_model_path('autoregressive.pth')]
	halfp = get_halfp_model_path()
	if os.path.exists(halfp):
		base.append(halfp)

	additionals = sorted([f'{dir}/{d}' for d in os.listdir(dir) if d[-4:] == ".pth" ])
	found = []
	for training in os.listdir(f'./training/'):
		if not os.path.isdir(f'./training/{training}/') or not os.path.isdir(f'./training/{training}/finetune/') or not os.path.isdir(f'./training/{training}/finetune/models/'):
			continue
		models = sorted([ int(d[:-8]) for d in os.listdir(f'./training/{training}/finetune/models/') if d[-8:] == "_gpt.pth" ])
		found = found + [ f'./training/{training}/finetune/models/{d}_gpt.pth' for d in models ]

	if len(found) > 0 or len(additionals) > 0:
		base = ["auto"] + base

	res = base + additionals + found
	
	if prefixed:
		for i in range(len(res)):
			path = res[i]
			hash = hash_file(path)
			shorthash = hash[:8]

			res[i] = f'[{shorthash}] {path}'

	return res

def get_dataset_list(dir="./training/"):
	return sorted([d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and "train.txt" in os.listdir(os.path.join(dir, d)) ])

def get_training_list(dir="./training/"):
	return sorted([f'./training/{d}/train.yaml' for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and "train.yaml" in os.listdir(os.path.join(dir, d)) ])

def do_gc():
	gc.collect()
	try:
		torch.cuda.empty_cache()
	except Exception as e:
		pass

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

def curl(url):
	try:
		req = urllib.request.Request(url, headers={'User-Agent': 'Python'})
		conn = urllib.request.urlopen(req)
		data = conn.read()
		data = data.decode()
		data = json.loads(data)
		conn.close()
		return data
	except Exception as e:
		print(e)
		return None

def check_for_updates( dir = None ):
	if dir is None:
		check_for_updates("./.git/")
		check_for_updates("./.git/modules/dlas/")
		check_for_updates("./.git/modules/tortoise-tts/")
		return

	git_dir = dir
	if not os.path.isfile(f'{git_dir}/FETCH_HEAD'):
		print(f"Cannot check for updates for {dir}: not from a git repo")
		return False

	with open(f'{git_dir}/FETCH_HEAD', 'r', encoding="utf-8") as f:
		head = f.read()
	
	match = re.findall(r"^([a-f0-9]+).+?https:\/\/(.+?)\/(.+?)\/(.+?)\n", head)
	if match is None or len(match) == 0:
		print(f"Cannot check for updates for {dir}: cannot parse FETCH_HEAD")
		return False

	match = match[0]

	local = match[0]
	host = match[1]
	owner = match[2]
	repo = match[3]

	res = curl(f"https://{host}/api/v1/repos/{owner}/{repo}/branches/") #this only works for gitea instances

	if res is None or len(res) == 0:
		print(f"Cannot check for updates for {dir}: cannot fetch from remote")
		return False

	remote = res[0]["commit"]["id"]

	if remote != local:
		print(f"New version found for {dir}: {local[:8]} => {remote[:8]}")
		return True

	return False

def enumerate_progress(iterable, desc=None, progress=None, verbose=None):
	if verbose and desc is not None:
		print(desc)

	if progress is None:
		return tqdm(iterable, disable=not verbose)
	return progress.tqdm(iterable, desc=f'{progress.msg_prefix} {desc}' if hasattr(progress, 'msg_prefix') else desc, track_tqdm=True)

def notify_progress(message, progress=None, verbose=True):
	if verbose:
		print(message)

	if progress is None:
		return

	progress(0, desc=message)

def get_args():
	global args
	return args

def setup_args():
	global args

	default_arguments = {
		'share': False,
		'listen': None,
		'check-for-updates': False,
		'models-from-local-only': False,
		'low-vram': False,
		'sample-batch-size': None,
		'embed-output-metadata': True,
		'latents-lean-and-mean': True,
		'voice-fixer': False, # getting tired of long initialization times in a Colab for downloading a large dataset for it
		'voice-fixer-use-cuda': True,
		'force-cpu-for-conditioning-latents': False,
		'defer-tts-load': False,
		'device-override': None,
		'prune-nonfinal-outputs': True,
		'vocoder-model': VOCODERS[-1],
		'concurrency-count': 2,
		'autocalculate-voice-chunk-duration-size': 0,
		'output-sample-rate': 44100,
		'output-volume': 1,
		
		'autoregressive-model': None,
		'whisper-backend': 'openai/whisper',
		'whisper-model': "base",

		'training-default-halfp': False,
		'training-default-bnb': True,
	}

	if os.path.isfile('./config/exec.json'):
		with open(f'./config/exec.json', 'r', encoding="utf-8") as f:
			try:
				overrides = json.load(f)
				for k in overrides:
					default_arguments[k] = overrides[k]
			except Exception as e:
				print(e)
				pass

	parser = argparse.ArgumentParser()
	parser.add_argument("--share", action='store_true', default=default_arguments['share'], help="Lets Gradio return a public URL to use anywhere")
	parser.add_argument("--listen", default=default_arguments['listen'], help="Path for Gradio to listen on")
	parser.add_argument("--check-for-updates", action='store_true', default=default_arguments['check-for-updates'], help="Checks for update on startup")
	parser.add_argument("--models-from-local-only", action='store_true', default=default_arguments['models-from-local-only'], help="Only loads models from disk, does not check for updates for models")
	parser.add_argument("--low-vram", action='store_true', default=default_arguments['low-vram'], help="Disables some optimizations that increases VRAM usage")
	parser.add_argument("--no-embed-output-metadata", action='store_false', default=not default_arguments['embed-output-metadata'], help="Disables embedding output metadata into resulting WAV files for easily fetching its settings used with the web UI (data is stored in the lyrics metadata tag)")
	parser.add_argument("--latents-lean-and-mean", action='store_true', default=default_arguments['latents-lean-and-mean'], help="Exports the bare essentials for latents.")
	parser.add_argument("--voice-fixer", action='store_true', default=default_arguments['voice-fixer'], help="Uses python module 'voicefixer' to improve audio quality, if available.")
	parser.add_argument("--voice-fixer-use-cuda", action='store_true', default=default_arguments['voice-fixer-use-cuda'], help="Hints to voicefixer to use CUDA, if available.")
	parser.add_argument("--force-cpu-for-conditioning-latents", default=default_arguments['force-cpu-for-conditioning-latents'], action='store_true', help="Forces computing conditional latents to be done on the CPU (if you constantyl OOM on low chunk counts)")
	parser.add_argument("--defer-tts-load", default=default_arguments['defer-tts-load'], action='store_true', help="Defers loading TTS model")
	parser.add_argument("--prune-nonfinal-outputs", default=default_arguments['prune-nonfinal-outputs'], action='store_true', help="Deletes non-final output files on completing a generation")
	parser.add_argument("--vocoder-model", default=default_arguments['vocoder-model'], action='store_true', help="Specifies with vocoder to use")
	parser.add_argument("--device-override", default=default_arguments['device-override'], help="A device string to override pass through Torch")
	parser.add_argument("--sample-batch-size", default=default_arguments['sample-batch-size'], type=int, help="Sets how many batches to use during the autoregressive samples pass")
	parser.add_argument("--concurrency-count", type=int, default=default_arguments['concurrency-count'], help="How many Gradio events to process at once")
	parser.add_argument("--autocalculate-voice-chunk-duration-size", type=float, default=default_arguments['autocalculate-voice-chunk-duration-size'], help="Number of seconds to suggest voice chunk size for (for example, 100 seconds of audio at 10 seconds per chunk will suggest 10 chunks)")
	parser.add_argument("--output-sample-rate", type=int, default=default_arguments['output-sample-rate'], help="Sample rate to resample the output to (from 24KHz)")
	parser.add_argument("--output-volume", type=float, default=default_arguments['output-volume'], help="Adjusts volume of output")
	
	parser.add_argument("--autoregressive-model", default=default_arguments['autoregressive-model'], help="Specifies which autoregressive model to use for sampling.")
	parser.add_argument("--whisper-backend", default=default_arguments['whisper-backend'], action='store_true', help="Picks which whisper backend to use (openai/whisper, lightmare/whispercpp, m-bain/whisperx)")
	parser.add_argument("--whisper-model", default=default_arguments['whisper-model'], help="Specifies which whisper model to use for transcription.")
	
	parser.add_argument("--training-default-halfp", action='store_true', default=default_arguments['training-default-halfp'], help="Training default: halfp")
	parser.add_argument("--training-default-bnb", action='store_true', default=default_arguments['training-default-bnb'], help="Training default: bnb")
	
	parser.add_argument("--os", default="unix", help="Specifies which OS, easily")
	args = parser.parse_args()

	args.embed_output_metadata = not args.no_embed_output_metadata

	if not args.device_override:
		set_device_name(args.device_override)


	args.listen_host = None
	args.listen_port = None
	args.listen_path = None
	if args.listen:
		try:
			match = re.findall(r"^(?:(.+?):(\d+))?(\/.*?)?$", args.listen)[0]

			args.listen_host = match[0] if match[0] != "" else "127.0.0.1"
			args.listen_port = match[1] if match[1] != "" else None
			args.listen_path = match[2] if match[2] != "" else "/"
		except Exception as e:
			pass

	if args.listen_port is not None:
		args.listen_port = int(args.listen_port)
		if args.listen_port == 0:
			args.listen_port = None
	
	return args

def update_args( **kwargs ):
	global args

	settings = {}
	settings.update(kwargs)

	args.listen = settings['listen']
	args.share = settings['share']
	args.check_for_updates = settings['check_for_updates']
	args.models_from_local_only = settings['models_from_local_only']
	args.low_vram = settings['low_vram']
	args.force_cpu_for_conditioning_latents = settings['force_cpu_for_conditioning_latents']
	args.defer_tts_load = settings['defer_tts_load']
	args.prune_nonfinal_outputs = settings['prune_nonfinal_outputs']
	args.device_override = settings['device_override']
	args.sample_batch_size = settings['sample_batch_size']
	args.embed_output_metadata = settings['embed_output_metadata']
	args.latents_lean_and_mean = settings['latents_lean_and_mean']
	args.voice_fixer = settings['voice_fixer']
	args.voice_fixer_use_cuda = settings['voice_fixer_use_cuda']
	args.concurrency_count = settings['concurrency_count']
	args.output_sample_rate = 44000
	args.autocalculate_voice_chunk_duration_size = settings['autocalculate_voice_chunk_duration_size']
	args.output_volume = settings['output_volume']
	
	args.autoregressive_model = settings['autoregressive_model']
	args.vocoder_model = settings['vocoder_model']
	args.whisper_backend = settings['whisper_backend']
	args.whisper_model = settings['whisper_model']

	args.training_default_halfp = settings['training_default_halfp']
	args.training_default_bnb = settings['training_default_bnb']

	save_args_settings()

def save_args_settings():
	global args
	settings = {
		'listen': None if not args.listen else args.listen,
		'share': args.share,
		'low-vram':args.low_vram,
		'check-for-updates':args.check_for_updates,
		'models-from-local-only':args.models_from_local_only,
		'force-cpu-for-conditioning-latents': args.force_cpu_for_conditioning_latents,
		'defer-tts-load': args.defer_tts_load,
		'prune-nonfinal-outputs': args.prune_nonfinal_outputs,
		'device-override': args.device_override,
		'sample-batch-size': args.sample_batch_size,
		'embed-output-metadata': args.embed_output_metadata,
		'latents-lean-and-mean': args.latents_lean_and_mean,
		'voice-fixer': args.voice_fixer,
		'voice-fixer-use-cuda': args.voice_fixer_use_cuda,
		'concurrency-count': args.concurrency_count,
		'output-sample-rate': args.output_sample_rate,
		'autocalculate-voice-chunk-duration-size': args.autocalculate_voice_chunk_duration_size,
		'output-volume': args.output_volume,
		
		'autoregressive-model': args.autoregressive_model,
		'vocoder-model': args.vocoder_model,
		'whisper-backend': args.whisper_backend,
		'whisper-model': args.whisper_model,

		'training-default-halfp': args.training_default_halfp,
		'training-default-bnb': args.training_default_bnb,
	}

	os.makedirs('./config/', exist_ok=True)
	with open(f'./config/exec.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps(settings, indent='\t') )

# super kludgy )`;
def import_generate_settings(file="./config/generate.json"):
	res = {
		'text': None,
		'delimiter': None,
		'emotion': None,
		'prompt': None,
		'voice': None,
		'mic_audio': None,
		'voice_latents_chunks': None,
		'candidates': None,
		'seed': None,
		'num_autoregressive_samples': 16,
		'diffusion_iterations': 30,
		'temperature': 0.8,
		'diffusion_sampler': "DDIM",
		'breathing_room': 8  ,
		'cvvp_weight': 0.0,
		'top_p': 0.8,
		'diffusion_temperature': 1.0,
		'length_penalty': 1.0,
		'repetition_penalty': 2.0,
		'cond_free_k': 2.0,
		'experimentals': None,
	}

	settings, _ = read_generate_settings(file, read_latents=False)
	if settings is not None:
		res.update(settings)
	return res

def read_generate_settings(file, read_latents=True):
	j = None
	latents = None

	if isinstance(file, list) and len(file) == 1:
		file = file[0]

	try:
		if file is not None:
			if hasattr(file, 'name'):
				file = file.name

			if file[-4:] == ".wav":
					metadata = music_tag.load_file(file)
					if 'lyrics' in metadata:
						j = json.loads(str(metadata['lyrics']))
			elif file[-5:] == ".json":
				with open(file, 'r') as f:
					j = json.load(f)
	except Exception as e:
		pass

	if j is not None:
		if 'latents' in j:
			if read_latents:
				latents = base64.b64decode(j['latents'])
			del j['latents']
		

		if "time" in j:
			j["time"] = "{:.3f}".format(j["time"])



	return (
		j,
		latents,
	)

def version_check_tts( min_version ):
	global tts
	if not tts:
		raise Exception("TTS is not initialized")

	if not hasattr(tts, 'version'):
		return False

	if min_version[0] > tts.version[0]:
		return True
	if min_version[1] > tts.version[1]:
		return True
	if min_version[2] >= tts.version[2]:
		return True
	return False

def load_tts( restart=False, autoregressive_model=None ):
	global args
	global tts

	if restart:
		unload_tts()

	if autoregressive_model:
		args.autoregressive_model = autoregressive_model
	else:
		autoregressive_model = args.autoregressive_model

	if autoregressive_model == "auto":
		autoregressive_model = deduce_autoregressive_model()

	print(f"Loading TorToiSe... (AR: {autoregressive_model}, vocoder: {args.vocoder_model})")

	tts_loading = True
	try:
		tts = TextToSpeech(minor_optimizations=not args.low_vram, autoregressive_model_path=autoregressive_model, vocoder_model=args.vocoder_model)
	except Exception as e:
		tts = TextToSpeech(minor_optimizations=not args.low_vram)
		load_autoregressive_model(autoregressive_model)

	tts_loading = False

	get_model_path('dvae.pth')
	print("Loaded TorToiSe, ready for generation.")
	return tts

setup_tortoise = load_tts

def unload_tts():
	global tts

	if tts:
		del tts
		tts = None
		print("Unloaded TTS")
	do_gc()

def reload_tts( model=None ):
	load_tts( restart=True, model=model )

def get_current_voice():
	global current_voice
	if current_voice:
		return current_voice

	settings, _ = read_generate_settings("./config/generate.json", read_latents=False)
	
	if settings and "voice" in settings['voice']:
		return settings["voice"]
	
	return None

def deduce_autoregressive_model(voice=None):
	if not voice:
		voice = get_current_voice()

	if voice:
		if os.path.exists(f'./models/finetunes/{voice}.pth'):
			return f'./models/finetunes/{voice}.pth'
		
		dir = f'./training/{voice}/finetune/models/'
		if os.path.isdir(dir):
			counts = sorted([ int(d[:-8]) for d in os.listdir(dir) if d[-8:] == "_gpt.pth" ])
			names = [ f'{dir}/{d}_gpt.pth' for d in counts ]
			if len(names) > 0:
				return names[-1]

	if args.autoregressive_model != "auto":
		return args.autoregressive_model

	return get_model_path('autoregressive.pth')

def update_autoregressive_model(autoregressive_model_path):
	match = re.findall(r'^\[[a-fA-F0-9]{8}\] (.+?)$', autoregressive_model_path)
	if match:
		autoregressive_model_path = match[0]

	if not autoregressive_model_path or not os.path.exists(autoregressive_model_path):
		print(f"Invalid model: {autoregressive_model_path}")
		return

	args.autoregressive_model = autoregressive_model_path
	save_args_settings()
	print(f'Stored autoregressive model to settings: {autoregressive_model_path}')

	global tts
	if not tts:
		if tts_loading:
			raise Exception("TTS is still initializing...")
		return
	
	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	if autoregressive_model_path == "auto":
		autoregressive_model_path = deduce_autoregressive_model()

	if autoregressive_model_path == tts.autoregressive_model_path:
		return

	tts.load_autoregressive_model(autoregressive_model_path)

	do_gc()
	
	return autoregressive_model_path

def update_vocoder_model(vocoder_model):
	args.vocoder_model = vocoder_model
	save_args_settings()
	print(f'Stored vocoder model to settings: {vocoder_model}')

	global tts
	if not tts:
		if tts_loading:
			raise Exception("TTS is still initializing...")
		return

	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	print(f"Loading model: {vocoder_model}")
	tts.load_vocoder_model(vocoder_model)
	print(f"Loaded model: {tts.vocoder_model}")

	do_gc()
	
	return vocoder_model

def load_voicefixer(restart=False):
	global voicefixer

	if restart:
		unload_voicefixer()

	try:
		print("Loading Voicefixer")
		from voicefixer import VoiceFixer
		voicefixer = VoiceFixer()
		print("Loaded Voicefixer")
	except Exception as e:
		print(f"Error occurred while tring to initialize voicefixer: {e}")
		if voicefixer:
			del voicefixer
		voicefixer = None

def unload_voicefixer():
	global voicefixer

	if voicefixer:
		del voicefixer
		voicefixer = None
		print("Unloaded Voicefixer")

	do_gc()

def load_whisper_model(language=None, model_name=None, progress=None):
	global whisper_model

	if args.whisper_backend not in WHISPER_BACKENDS:
		raise Exception(f"unavailable backend: {args.whisper_backend}")

	if args.whisper_backend != "m-bain/whisperx" and model_name == "large-v2":
		raise Exception("large-v2 is only available for m-bain/whisperx backend")
	
	if not model_name:
		model_name = args.whisper_model
	else:
		args.whisper_model = model_name
		save_args_settings()

	if language and f'{model_name}.{language}' in WHISPER_SPECIALIZED_MODELS:
		model_name = f'{model_name}.{language}'
		print(f"Loading specialized model for language: {language}")

	notify_progress(f"Loading Whisper model: {model_name}", progress)

	if args.whisper_backend == "openai/whisper":
		import whisper
		whisper_model = whisper.load_model(model_name)
	elif args.whisper_backend == "lightmare/whispercpp":
		from whispercpp import Whisper
		if not language:
			language = 'auto'

		b_lang = language.encode('ascii')
		whisper_model = Whisper(model_name, models_dir='./models/', language=b_lang)
	elif args.whisper_backend == "m-bain/whisperx":
		import whisperx
		device = "cuda" if get_device_name() == "cuda" else "cpu"
		whisper_model = whisperx.load_model(model_name, device)

	print("Loaded Whisper model")

def unload_whisper():
	global whisper_model

	if whisper_model:
		del whisper_model
		whisper_model = None
		print("Unloaded Whisper")

	do_gc()	