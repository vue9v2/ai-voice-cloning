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

import tqdm
import torch
import torchaudio
import music_tag
import gradio as gr
import gradio.utils
import pandas as pd

from datetime import datetime
from datetime import timedelta

from tortoise.api import TextToSpeech, MODELS, get_model_path
from tortoise.utils.audio import load_audio, load_voice, load_voices, get_voice_dir
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.device import get_device_name, set_device_name

MODELS['dvae.pth'] = "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/3704aea61678e7e468a06d8eea121dba368a798e/.models/dvae.pth"
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
WHISPER_SPECIALIZED_MODELS = ["tiny.en", "base.en", "small.en", "medium.en"]
EPOCH_SCHEDULE = [ 9, 18, 25, 33 ]

args = None
tts = None
tts_loading = False
webui = None
voicefixer = None
whisper_model = None
training_state = None

def generate(
	text,
	delimiter,
	emotion,
	prompt,
	voice,
	mic_audio,
	voice_latents_chunks,
	seed,
	candidates,
	num_autoregressive_samples,
	diffusion_iterations,
	temperature,
	diffusion_sampler,
	breathing_room,
	cvvp_weight,
	top_p,
	diffusion_temperature,
	length_penalty,
	repetition_penalty,
	cond_free_k,
	experimental_checkboxes,
	progress=None
):
	global args
	global tts

	unload_whisper()
	unload_voicefixer()

	if not tts:
		# should check if it's loading or unloaded, and load it if it's unloaded
		if tts_loading:
			raise Exception("TTS is still initializing...")
		load_tts()

	do_gc()

	if voice != "microphone":
		voices = [voice]
	else:
		voices = []

	if voice == "microphone":
		if mic_audio is None:
			raise Exception("Please provide audio from mic when choosing `microphone` as a voice input")
		mic = load_audio(mic_audio, tts.input_sample_rate)
		voice_samples, conditioning_latents = [mic], None
	elif voice == "random":
		voice_samples, conditioning_latents = None, tts.get_random_conditioning_latents()
	else:
		progress(0, desc="Loading voice...")
		# nasty check for users that, for whatever reason, updated the web UI but not mrq/tortoise-tts
		if hasattr(tts, 'autoregressive_model_hash'):
			voice_samples, conditioning_latents = load_voice(voice, model_hash=tts.autoregressive_model_hash)
		else:
			voice_samples, conditioning_latents = load_voice(voice)

	if voice_samples and len(voice_samples) > 0:
		sample_voice = torch.cat(voice_samples, dim=-1).squeeze().cpu()

		conditioning_latents = tts.get_conditioning_latents(voice_samples, return_mels=not args.latents_lean_and_mean, progress=progress, slices=voice_latents_chunks, force_cpu=args.force_cpu_for_conditioning_latents)
		if len(conditioning_latents) == 4:
			conditioning_latents = (conditioning_latents[0], conditioning_latents[1], conditioning_latents[2], None)
			
		if voice != "microphone":
			if hasattr(tts, 'autoregressive_model_hash'):
				torch.save(conditioning_latents, f'{get_voice_dir()}/{voice}/cond_latents_{tts.autoregressive_model_hash[:8]}.pth')
			else:
				torch.save(conditioning_latents, f'{get_voice_dir()}/{voice}/cond_latents.pth')
		voice_samples = None
	else:
		if conditioning_latents is not None:
			sample_voice, _ = load_voice(voice, load_latents=False)
			if sample_voice and len(sample_voice) > 0:
				sample_voice = torch.cat(sample_voice, dim=-1).squeeze().cpu()
		else:
			sample_voice = None

	if seed == 0:
		seed = None

	if conditioning_latents is not None and len(conditioning_latents) == 2 and cvvp_weight > 0:
		print("Requesting weighing against CVVP weight, but voice latents are missing some extra data. Please regenerate your voice latents.")
		cvvp_weight = 0


	settings = {
		'temperature': float(temperature),

		'top_p': float(top_p),
		'diffusion_temperature': float(diffusion_temperature),
		'length_penalty': float(length_penalty),
		'repetition_penalty': float(repetition_penalty),
		'cond_free_k': float(cond_free_k),

		'num_autoregressive_samples': num_autoregressive_samples,
		'sample_batch_size': args.sample_batch_size,
		'diffusion_iterations': diffusion_iterations,

		'voice_samples': voice_samples,
		'conditioning_latents': conditioning_latents,
		'use_deterministic_seed': seed,
		'return_deterministic_state': True,
		'k': candidates,
		'diffusion_sampler': diffusion_sampler,
		'breathing_room': breathing_room,
		'progress': progress,
		'half_p': "Half Precision" in experimental_checkboxes,
		'cond_free': "Conditioning-Free" in experimental_checkboxes,
		'cvvp_amount': cvvp_weight,
	}

	# clamp it down for the insane users who want this
	# it would be wiser to enforce the sample size to the batch size, but this is what the user wants
	sample_batch_size = args.sample_batch_size
	if not sample_batch_size:
		sample_batch_size = tts.autoregressive_batch_size
	if num_autoregressive_samples < sample_batch_size:
		settings['sample_batch_size'] = num_autoregressive_samples

	if delimiter is None:
		delimiter = "\n"
	elif delimiter == "\\n":
		delimiter = "\n"

	if delimiter and delimiter != "" and delimiter in text:
		texts = text.split(delimiter)
	else:
		texts = split_and_recombine_text(text)
 
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
		if candidates > 1:
			name = f"{name}_{candidate}"
		return name

	for line, cut_text in enumerate(texts):
		if emotion == "Custom":
			if prompt and prompt.strip() != "":
				cut_text = f"[{prompt},] {cut_text}"
		else:
			cut_text = f"[I am really {emotion.lower()},] {cut_text}"

		progress.msg_prefix = f'[{str(line+1)}/{str(len(texts))}]'
		print(f"{progress.msg_prefix} Generating line: {cut_text}")

		start_time = time.time()
		gen, additionals = tts.tts(cut_text, **settings )
		seed = additionals[0]
		run_time = time.time()-start_time
		print(f"Generating line took {run_time} seconds")
 
		if not isinstance(gen, list):
			gen = [gen]

		for j, g in enumerate(gen):
			audio = g.squeeze(0).cpu()
			name = get_name(line=line, candidate=j)
			audio_cache[name] = {
				'audio': audio,
				'text': cut_text,
				'time': run_time
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
	for candidate in range(candidates):
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
				'text': text,
				'time': time.time()-full_start_time,
				'output': True
			}
		else:
			name = get_name(candidate=candidate)
			audio_cache[name]['output'] = True

	info = {
		'text': text,
		'delimiter': '\\n' if delimiter and delimiter == "\n" else delimiter,
		'emotion': emotion,
		'prompt': prompt,
		'voice': voice,
		'seed': seed,
		'candidates': candidates,
		'num_autoregressive_samples': num_autoregressive_samples,
		'diffusion_iterations': diffusion_iterations,
		'temperature': temperature,
		'diffusion_sampler': diffusion_sampler,
		'breathing_room': breathing_room,
		'cvvp_weight': cvvp_weight,
		'top_p': top_p,
		'diffusion_temperature': diffusion_temperature,
		'length_penalty': length_penalty,
		'repetition_penalty': repetition_penalty,
		'cond_free_k': cond_free_k,
		'experimentals': experimental_checkboxes,
		'time': time.time()-full_start_time,

		'datetime': datetime.now().isoformat(),
		'model': tts.autoregressive_model_path,
		'model_hash': tts.autoregressive_model_hash if hasattr(tts, 'autoregressive_model_hash') else None,
	}

	"""
	# kludgy yucky codesmells
	for name in audio_cache:
		if 'output' not in audio_cache[name]:
			continue

		#output_voices.append(f'{outdir}/{voice}_{name}.wav')
		output_voices.append(name)
		if not args.embed_output_metadata:
			with open(f'{outdir}/{voice}_{name}.json', 'w', encoding="utf-8") as f:
				f.write(json.dumps(info, indent='\t') )
	"""

	if args.voice_fixer:
		if not voicefixer:
			progress(0, "Loading voicefix...")
			load_voicefixer()

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
				'text': audio_cache[name]['text'],
				'time': audio_cache[name]['time'],
				'output': True
			}
			audio_cache[name]['output'] = False
		
		for name in fixed_cache:
			audio_cache[name] = fixed_cache[name]

	for name in audio_cache:
		if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
			if args.prune_nonfinal_outputs:
				audio_cache[name]['pruned'] = True
				os.remove(f'{outdir}/{voice}_{name}.wav')
			continue

		output_voices.append(f'{outdir}/{voice}_{name}.wav')

		if not args.embed_output_metadata:
			with open(f'{outdir}/{voice}_{name}.json', 'w', encoding="utf-8") as f:
				f.write(json.dumps(info, indent='\t') )


	if voice and voice != "random" and conditioning_latents is not None:
		latents_path = f'{get_voice_dir()}/{voice}/cond_latents.pth'

		if hasattr(tts, 'autoregressive_model_hash'):
			latents_path = f'{get_voice_dir()}/{voice}/cond_latents_{tts.autoregressive_model_hash[:8]}.pth'

		try:
			with open(latents_path, 'rb') as f:
				info['latents'] = base64.b64encode(f.read()).decode("ascii")
		except Exception as e:
			pass

	if args.embed_output_metadata:
		for name in progress.tqdm(audio_cache, desc="Embedding metadata..."):
			if 'pruned' in audio_cache[name] and audio_cache[name]['pruned']:
				continue

			info['text'] = audio_cache[name]['text']
			info['time'] = audio_cache[name]['time']

			metadata = music_tag.load_file(f"{outdir}/{voice}_{name}.wav")
			metadata['lyrics'] = json.dumps(info) 
			metadata.save()
 
	if sample_voice is not None:
		sample_voice = (tts.input_sample_rate, sample_voice.numpy())

	print(f"Generation took {info['time']} seconds, saved to '{output_voices[0]}'\n")

	info['seed'] = settings['use_deterministic_seed']
	if 'latents' in info:
		del info['latents']

	os.makedirs('./config/', exist_ok=True)
	with open(f'./config/generate.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps(info, indent='\t') )

	stats = [
		[ seed, "{:.3f}".format(info['time']) ]
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
	import hashlib

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
	path = f'{get_voice_dir()}/{voice}/'
	if not os.path.isdir(path):
		return 1

	files = os.listdir(path)
	total_duration = 0
	for file in files:
		if file[-4:] != ".wav":
			continue
		metadata = torchaudio.info(f'{path}/{file}')
		duration = metadata.num_channels * metadata.num_frames / metadata.sample_rate
		total_duration += duration

	return int(total_duration / args.autocalculate_voice_chunk_duration_size) if total_duration > 0 else 1

def compute_latents(voice, voice_latents_chunks, progress=gr.Progress(track_tqdm=True)):
	global tts
	global args
	
	unload_whisper()
	unload_voicefixer()

	if not tts:
		if tts_loading:
			raise Exception("TTS is still initializing...")
		load_tts()

	voice_samples, conditioning_latents = load_voice(voice, load_latents=False)

	if voice_samples is None:
		return

	conditioning_latents = tts.get_conditioning_latents(voice_samples, return_mels=not args.latents_lean_and_mean, progress=progress, slices=voice_latents_chunks, force_cpu=args.force_cpu_for_conditioning_latents)

	if len(conditioning_latents) == 4:
		conditioning_latents = (conditioning_latents[0], conditioning_latents[1], conditioning_latents[2], None)
			
	if hasattr(tts, 'autoregressive_model_hash'):
		torch.save(conditioning_latents, f'{get_voice_dir()}/{voice}/cond_latents_{tts.autoregressive_model_hash[:8]}.pth')
	else:
		torch.save(conditioning_latents, f'{get_voice_dir()}/{voice}/cond_latents.pth')

	return voice

# superfluous, but it cleans up some things
class TrainingState():
	def __init__(self, config_path, keep_x_past_datasets=0, start=True, gpus=1):
		# parse config to get its iteration
		with open(config_path, 'r') as file:
			self.config = yaml.safe_load(file)

		self.killed = False

		self.dataset_dir = f"./training/{self.config['name']}/"
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

		self.last_info_check_at = 0
		self.statistics = []
		self.losses = []
		self.metrics = {
			'step': "",
			'rate': "",
			'loss': "",
		}

		self.loss_milestones = [ 1.0, 0.15, 0.05 ]

		self.load_losses()
		if keep_x_past_datasets > 0:
			self.cleanup_old(keep=keep_x_past_datasets)
		if start:
			self.spawn_process(config_path=config_path, gpus=gpus)

	def spawn_process(self, config_path, gpus=1):
		self.cmd = ['train.bat', config_path] if os.name == "nt" else ['./train.sh', str(int(gpus)), config_path]

		print("Spawning process: ", " ".join(self.cmd))
		self.process = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

	def load_losses(self, update=False):
		if not os.path.isdir(f'{self.dataset_dir}/tb_logger/'):
			return
		try:
			from tensorboard.backend.event_processing import event_accumulator
			use_tensorboard = True
		except Exception as e:
			use_tensorboard = False

		keys = ['loss_text_ce', 'loss_mel_ce', 'loss_gpt_total']
		infos = {}
		highest_step = self.last_info_check_at

		if not update:
			self.statistics = []

		if use_tensorboard:
			logs = sorted([f'{self.dataset_dir}/tb_logger/{d}' for d in os.listdir(f'{self.dataset_dir}/tb_logger/') if d[:6] == "events" ])
			if update:
				logs = [logs[-1]]

			for log in logs:
				try:
					ea = event_accumulator.EventAccumulator(log, size_guidance={event_accumulator.SCALARS: 0})
					ea.Reload()

					for key in keys:
						scalar = ea.Scalars(key)
						for s in scalar:
							if update and s.step <= self.last_info_check_at:
								continue
							highest_step = max( highest_step, s.step )
							self.statistics.append( { "step": s.step, "value": s.value, "type": key } )

							if key == 'loss_gpt_total':
								self.losses.append( { "step": s.step, "value": s.value, "type": key } )

				except Exception as e:
					pass

		else:
			logs = sorted([f'{self.dataset_dir}/{d}' for d in os.listdir(self.dataset_dir) if d[-4:] == ".log" ])
			if update:
				logs = [logs[-1]]

			for log in logs:
				with open(log, 'r', encoding="utf-8") as f:
					lines = f.readlines()
					for line in lines:
						if line.find('INFO: [epoch:') >= 0:
							# easily rip out our stats...
							match = re.findall(r'\b([a-z_0-9]+?)\b: +?([0-9]\.[0-9]+?e[+-]\d+|[\d,]+)\b', line)
							if not match or len(match) == 0:
								continue

							info = {}
							for k, v in match:
								info[k] = float(v.replace(",", ""))

							if 'iter' in info:
								it = info['iter']
								infos[it] = info

			for k in infos:
				if 'loss_gpt_total' in infos[k]:
					for key in keys:
						if update and int(k) <= self.last_info_check_at:
							continue
						highest_step = max( highest_step, s.step )
						self.statistics.append({ "step": int(k), "value": infos[k][key], "type": key })

						if key == "loss_gpt_total":
							self.losses.append({ "step": int(k), "value": infos[k][key], "type": key })

		self.last_info_check_at = highest_step

	def cleanup_old(self, keep=2):
		if keep <= 0:
			return

		if not os.path.isdir(self.dataset_dir):
			return
			
		models = sorted([ int(d[:-8]) for d in os.listdir(f'{self.dataset_dir}/models/') if d[-8:] == "_gpt.pth" ])
		states = sorted([ int(d[:-6]) for d in os.listdir(f'{self.dataset_dir}/training_state/') if d[-6:] == ".state" ])
		remove_models = models[:-2]
		remove_states = states[:-2]

		for d in remove_models:
			path = f'{self.dataset_dir}/models/{d}_gpt.pth'
			print("Removing", path)
			os.remove(path)
		for d in remove_states:
			path = f'{self.dataset_dir}/training_state/{d}.state'
			print("Removing", path)
			os.remove(path)

	def parse(self, line, verbose=False, keep_x_past_datasets=0, buffer_size=8, progress=None ):
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
				# to-do, actually validate this works, and probably kill training when it's found, the model's dead by this point
				if ': nan' in line:
					should_return = True

					print("! NAN DETECTED !")
					self.buffer.append("! NAN DETECTED !")

				# easily rip out our stats...
				match = re.findall(r'\b([a-z_0-9]+?)\b: +?([0-9]\.[0-9]+?e[+-]\d+|[\d,]+)\b', line)
				if match and len(match) > 0:
					for k, v in match:
						self.info[k] = float(v.replace(",", ""))

				self.load_losses(update=True)
				should_return = True

			elif line.find('Saving models and training states') >= 0:
				self.checkpoint = self.checkpoint + 1

				percent = self.checkpoint / float(self.checkpoints)
				message = f'[{self.checkpoint}/{self.checkpoints}] Saving checkpoint...'
				if progress is not None:
					progress(percent, message)

				print(f'{"{:.3f}".format(percent*100)}% {message}')
				self.buffer.append(f'{"{:.3f}".format(percent*100)}% {message}')

				self.cleanup_old(keep=keep_x_past_datasets)

			elif line.find('%|') > 0:
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
						self.metric.append(f"{self.it}/{self.its}")
					if steps > 1:
						self.metric.append(f"{step}/{steps}")
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
				self.metrics['loss'].append(f'LR: {"{:.9f}".format(self.info["learning_rate_gpt_0"])}')

			if len(self.losses) > 0:
				self.metrics['loss'].append(f'Loss: {"{:.3f}".format(self.losses[-1]["value"])}')

			if len(self.losses) >= 2:
				# """riemann sum""" but not really as this is for derivatives and not integrals
				deriv = 0
				accum_length = len(self.losses)//2 # i *guess* this is fine when you think about it
				for i in range(accum_length):
					d1_loss = self.losses[-i-1]["value"]
					d2_loss = self.losses[-i-2]["value"]
					dloss = (d2_loss - d1_loss)

					d1_step = self.losses[-i-1]["step"]
					d2_step = self.losses[-i-2]["step"]
					dstep = (d2_step - d1_step)

					if dstep == 0:
						continue
			
					inst_deriv = dloss / dstep
					deriv += inst_deriv

				deriv = deriv / accum_length

				if deriv != 0: # dloss < 0:
					next_milestone = None
					for milestone in self.loss_milestones:
						if d1_loss > milestone:
							next_milestone = milestone
							break

					if next_milestone:
						# tfw can do simple calculus but not basic algebra in my head
						est_its = (next_milestone - d1_loss) / deriv
						if est_its >= 0:
							self.metrics['loss'].append(f'Est. milestone {next_milestone} in: {int(est_its)}its')
					else:
						est_loss = inst_deriv * (self.its - self.it) + d1_loss
						if est_loss >= 0:
							self.metrics['loss'].append(f'Est. final loss: {"{:.3f}".format(est_loss)}')

			self.metrics['loss'] = ", ".join(self.metrics['loss'])

			message = f"[{self.metrics['step']}] [{self.metrics['rate']}] [ETA: {eta_hhmmss}]\n[{self.metrics['loss']}]"

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

def run_training(config_path, verbose=False, gpus=1, keep_x_past_datasets=0, progress=gr.Progress(track_tqdm=True)):
	global training_state
	if training_state and training_state.process:
		return "Training already in progress"
	
	# I don't know if this is still necessary, as it was bitching at me for not doing this, despite it being in a separate process
	torch.multiprocessing.freeze_support()

	unload_tts()
	unload_whisper()
	unload_voicefixer()

	training_state = TrainingState(config_path=config_path, keep_x_past_datasets=keep_x_past_datasets, gpus=gpus)

	for line in iter(training_state.process.stdout.readline, ""):
		if training_state.killed:
			return

		result, percent, message = training_state.parse( line=line, verbose=verbose, keep_x_past_datasets=keep_x_past_datasets, progress=progress )
		print(f"[Training] [{datetime.now().isoformat()}] {line[:-1]}")
		if result:
			yield result

			if progress is not None and message:
				progress(percent, message)

	if training_state:
		training_state.process.stdout.close()
		return_code = training_state.process.wait()
		training_state = None

def get_training_losses():
	global training_state
	if not training_state or not training_state.statistics:
		return
	return pd.DataFrame(training_state.statistics)

def update_training_dataplot(config_path=None):
	global training_state
	update = None

	if not training_state:
		if config_path:
			training_state = TrainingState(config_path=config_path, start=False)
			if training_state.statistics:
				update = gr.LinePlot.update(value=pd.DataFrame(training_state.statistics))
			del training_state
			training_state = None
	elif training_state.statistics:
		training_state.load_losses()
		update = gr.LinePlot.update(value=pd.DataFrame(training_state.statistics))

	return update

def reconnect_training(verbose=False, progress=gr.Progress(track_tqdm=True)):
	global training_state
	if not training_state or not training_state.process:
		return "Training not in progress"

	for line in iter(training_state.process.stdout.readline, ""):
		result, percent, message = training_state.parse( line=line, verbose=verbose, keep_x_past_datasets=keep_x_past_datasets, progress=progress )
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
		print(e)
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

	if not args.whisper_cpp:
		if not language:
			language = None

		return whisper_model.transcribe(file, language=language)

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


def prepare_dataset( files, outdir, language=None, progress=None ):
	unload_tts()

	global whisper_model
	if whisper_model is None:
		load_whisper_model(language=language)

	os.makedirs(outdir, exist_ok=True)

	results = {}
	transcription = []

	for file in enumerate_progress(files, desc="Iterating through voice files", progress=progress):
		basename = os.path.basename(file)
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

			torchaudio.save(f"{outdir}/{sliced_name}", sliced_waveform, sampling_rate)

			idx = idx + 1
			line = f"{sliced_name}|{segment['text'].strip()}"
			transcription.append(line)
			with open(f'{outdir}/train.txt', 'a', encoding="utf-8") as f:
				f.write(f'{line}\n')
	
	with open(f'{outdir}/whisper.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps(results, indent='\t'))
	
	joined = '\n'.join(transcription)
	with open(f'{outdir}/train.txt', 'w', encoding="utf-8") as f:
		f.write(joined)

	unload_whisper()

	return f"Processed dataset to: {outdir}\n{joined}"

def calc_iterations( epochs, lines, batch_size ):
	iterations = int(epochs * lines / float(batch_size))
	return iterations

def schedule_learning_rate( iterations, schedule=EPOCH_SCHEDULE ):
	return [int(iterations * d) for d in schedule]

def optimize_training_settings( epochs, learning_rate, text_ce_lr_weight, learning_rate_schedule, batch_size, gradient_accumulation_size, print_rate, save_rate, resume_path, half_p, bnb, workers, source_model, voice ):
	name = f"{voice}-finetune"
	dataset_name = f"{voice}-train"
	dataset_path = f"./training/{voice}/train.txt"
	validation_name = f"{voice}-val"
	validation_path = f"./training/{voice}/train.txt"

	with open(dataset_path, 'r', encoding="utf-8") as f:
		lines = len(f.readlines())

	messages = []

	if batch_size > lines:
		batch_size = lines
		messages.append(f"Batch size is larger than your dataset, clamping batch size to: {batch_size}")	

	if batch_size % lines != 0:
		nearest_slice = int(lines / batch_size) + 1
		batch_size = int(lines / nearest_slice)
		messages.append(f"Batch size not neatly divisible by dataset size, adjusting batch size to: {batch_size} ({nearest_slice} steps per epoch)")
	
	if gradient_accumulation_size == 0:
		gradient_accumulation_size = 1
	
	if batch_size / gradient_accumulation_size < 2:
		gradient_accumulation_size = int(batch_size / 2)
		if gradient_accumulation_size == 0:
			gradient_accumulation_size = 1

		messages.append(f"Gradient accumulation size is too large for a given batch size, clamping gradient accumulation size to: {gradient_accumulation_size}")
	elif batch_size % gradient_accumulation_size != 0:
		gradient_accumulation_size = int(batch_size / gradient_accumulation_size)
		if gradient_accumulation_size == 0:
			gradient_accumulation_size = 1

		messages.append(f"Batch size is not evenly divisible by the gradient accumulation size, adjusting gradient accumulation size to: {gradient_accumulation_size}")

	iterations = calc_iterations(epochs=epochs, lines=lines, batch_size=batch_size)

	if epochs < print_rate:
		print_rate = epochs
		messages.append(f"Print rate is too small for the given iteration step, clamping print rate to: {print_rate}")
	
	if epochs < save_rate:
		save_rate = epochs
		messages.append(f"Save rate is too small for the given iteration step, clamping save rate to: {save_rate}")

	if resume_path and not os.path.exists(resume_path):
		resume_path = None
		messages.append("Resume path specified, but does not exist. Disabling...")

	if bnb:
		messages.append("BitsAndBytes requested. Please note this is ! EXPERIMENTAL !")

	if half_p:
		if bnb:
			half_p = False
			messages.append("Half Precision requested, but BitsAndBytes is also requested. Due to redundancies, disabling half precision...")
		else:
			messages.append("Half Precision requested. Please note this is ! EXPERIMENTAL !")
			if not os.path.exists(get_halfp_model_path()):
				convert_to_halfp()	

	messages.append(f"For {epochs} epochs with {lines} lines in batches of {batch_size}, iterating for {iterations} steps ({int(iterations / epochs)} steps per epoch)")

	return (
		learning_rate,
		text_ce_lr_weight,
		learning_rate_schedule,
		batch_size,
		gradient_accumulation_size,
		print_rate,
		save_rate,
		resume_path,
		messages
	)

def save_training_settings( iterations=None, learning_rate=None, text_ce_lr_weight=None, learning_rate_schedule=None, batch_size=None, gradient_accumulation_size=None, print_rate=None, save_rate=None, name=None, dataset_name=None, dataset_path=None, validation_name=None, validation_path=None, output_name=None, resume_path=None, half_p=None, bnb=None, workers=None, source_model=None ):
	if not source_model:
		source_model = f"./models/tortoise/autoregressive{'_half' if half_p else ''}.pth"

	settings = {
		"iterations": iterations if iterations else 500,
		"batch_size": batch_size if batch_size else 64,
		"learning_rate": learning_rate if learning_rate else 1e-5,
		"gen_lr_steps": learning_rate_schedule if learning_rate_schedule else EPOCH_SCHEDULE,
		"gradient_accumulation_size": gradient_accumulation_size if gradient_accumulation_size else 4,
		"print_rate": print_rate if print_rate else 1,
		"save_rate": save_rate if save_rate else 50,
		"name": name if name else "finetune",
		"dataset_name": dataset_name if dataset_name else "finetune",
		"dataset_path": dataset_path if dataset_path else "./training/finetune/train.txt",
		"validation_name": validation_name if validation_name else "finetune",
		"validation_path": validation_path if validation_path else "./training/finetune/train.txt",

		"text_ce_lr_weight": text_ce_lr_weight if text_ce_lr_weight else 0.01,

		'resume_state': f"resume_state: '{resume_path}'",
		'pretrain_model_gpt': f"pretrain_model_gpt: '{source_model}'",

		'float16': 'true' if half_p else 'false',
		'bitsandbytes': 'true' if bnb else 'false',

		'workers': workers if workers else 2,
	}

	if resume_path:
		settings['pretrain_model_gpt'] = f"# {settings['pretrain_model_gpt']}"
	else:
		settings['resume_state'] = f"# resume_state: './training/{name if name else 'finetune'}/training_state/#.state'"

	if half_p:
		if not os.path.exists(get_halfp_model_path()):
			convert_to_halfp()

	if not output_name:
		output_name = f'{settings["name"]}.yaml'


	with open(f'./models/.template.yaml', 'r', encoding="utf-8") as f:
		yaml = f.read()

	# i could just load and edit the YAML directly, but this is easier, as I don't need to bother with path traversals
	for k in settings:
		if settings[k] is None:
			continue
		yaml = yaml.replace(f"${{{k}}}", str(settings[k]))

	outfile = f'./training/{output_name}'
	with open(outfile, 'w', encoding="utf-8") as f:
		f.write(yaml)

	return f"Training settings saved to: {outfile}"

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
	os.makedirs(dir, exist_ok=True)
	res = sorted([d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and len(os.listdir(os.path.join(dir, d))) > 0 ])
	if append_defaults:
		res = res + ["random", "microphone"]
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
		if not os.path.isdir(f'./training/{training}/') or not os.path.isdir(f'./training/{training}/models/'):
			continue
		models = sorted([ int(d[:-8]) for d in os.listdir(f'./training/{training}/models/') if d[-8:] == "_gpt.pth" ])
		found = found + [ f'./training/{training}/models/{d}_gpt.pth' for d in models ]

	res = base + additionals + found
	
	if prefixed:
		for i in range(len(res)):
			path = res[i]
			hash = hash_file(path)
			shorthash = hash[:8]

			res[i] = f'[{shorthash}] {path}'

	return res

def get_dataset_list(dir="./training/"):
	return sorted([d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and len(os.listdir(os.path.join(dir, d))) > 0 and "train.txt" in os.listdir(os.path.join(dir, d)) ])

def get_training_list(dir="./training/"):
	return sorted([f'./training/{d}/train.yaml' for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and len(os.listdir(os.path.join(dir, d))) > 0 and "train.yaml" in os.listdir(os.path.join(dir, d)) ])

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

def check_for_updates():
	if not os.path.isfile('./.git/FETCH_HEAD'):
		print("Cannot check for updates: not from a git repo")
		return False

	with open(f'./.git/FETCH_HEAD', 'r', encoding="utf-8") as f:
		head = f.read()
	
	match = re.findall(r"^([a-f0-9]+).+?https:\/\/(.+?)\/(.+?)\/(.+?)\n", head)
	if match is None or len(match) == 0:
		print("Cannot check for updates: cannot parse FETCH_HEAD")
		return False

	match = match[0]

	local = match[0]
	host = match[1]
	owner = match[2]
	repo = match[3]

	res = curl(f"https://{host}/api/v1/repos/{owner}/{repo}/branches/") #this only works for gitea instances

	if res is None or len(res) == 0:
		print("Cannot check for updates: cannot fetch from remote")
		return False

	remote = res[0]["commit"]["id"]

	if remote != local:
		print(f"New version found: {local[:8]} => {remote[:8]}")
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
		'use-bigvgan-vocoder': True,
		'concurrency-count': 2,
		'autocalculate-voice-chunk-duration-size': 10,
		'output-sample-rate': 44100,
		'output-volume': 1,
		
		'autoregressive-model': None,
		'whisper-model': "base",
		'whisper-cpp': False,

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
	parser.add_argument("--use-bigvgan-vocoder", default=default_arguments['use-bigvgan-vocoder'], action='store_true', help="Uses BigVGAN in place of the default vocoder")
	parser.add_argument("--device-override", default=default_arguments['device-override'], help="A device string to override pass through Torch")
	parser.add_argument("--sample-batch-size", default=default_arguments['sample-batch-size'], type=int, help="Sets how many batches to use during the autoregressive samples pass")
	parser.add_argument("--concurrency-count", type=int, default=default_arguments['concurrency-count'], help="How many Gradio events to process at once")
	parser.add_argument("--autocalculate-voice-chunk-duration-size", type=float, default=default_arguments['autocalculate-voice-chunk-duration-size'], help="Number of seconds to suggest voice chunk size for (for example, 100 seconds of audio at 10 seconds per chunk will suggest 10 chunks)")
	parser.add_argument("--output-sample-rate", type=int, default=default_arguments['output-sample-rate'], help="Sample rate to resample the output to (from 24KHz)")
	parser.add_argument("--output-volume", type=float, default=default_arguments['output-volume'], help="Adjusts volume of output")
	
	parser.add_argument("--autoregressive-model", default=default_arguments['autoregressive-model'], help="Specifies which autoregressive model to use for sampling.")
	parser.add_argument("--whisper-model", default=default_arguments['whisper-model'], help="Specifies which whisper model to use for transcription.")
	parser.add_argument("--whisper-cpp", default=default_arguments['whisper-cpp'], action='store_true', help="Leverages lightmare/whispercpp for transcription")
	
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

def update_args( listen, share, check_for_updates, models_from_local_only, low_vram, embed_output_metadata, latents_lean_and_mean, voice_fixer, voice_fixer_use_cuda, force_cpu_for_conditioning_latents, defer_tts_load, prune_nonfinal_outputs, use_bigvgan_vocoder, device_override, sample_batch_size, concurrency_count, autocalculate_voice_chunk_duration_size, output_volume, autoregressive_model, whisper_model, whisper_cpp, training_default_halfp, training_default_bnb ):
	global args

	args.listen = listen
	args.share = share
	args.check_for_updates = check_for_updates
	args.models_from_local_only = models_from_local_only
	args.low_vram = low_vram
	args.force_cpu_for_conditioning_latents = force_cpu_for_conditioning_latents
	args.defer_tts_load = defer_tts_load
	args.prune_nonfinal_outputs = prune_nonfinal_outputs
	args.use_bigvgan_vocoder = use_bigvgan_vocoder
	args.device_override = device_override
	args.sample_batch_size = sample_batch_size
	args.embed_output_metadata = embed_output_metadata
	args.latents_lean_and_mean = latents_lean_and_mean
	args.voice_fixer = voice_fixer
	args.voice_fixer_use_cuda = voice_fixer_use_cuda
	args.concurrency_count = concurrency_count
	args.output_sample_rate = 44000
	args.autocalculate_voice_chunk_duration_size = autocalculate_voice_chunk_duration_size
	args.output_volume = output_volume
	
	args.autoregressive_model = autoregressive_model
	args.whisper_model = whisper_model
	args.whisper_cpp = whisper_cpp

	args.training_default_halfp = training_default_halfp
	args.training_default_bnb = training_default_bnb

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
		'use-bigvgan-vocoder': args.use_bigvgan_vocoder,
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
		'whisper-model': args.whisper_model,
		'whisper-cpp': args.whisper_cpp,

		'training-default-halfp': args.training_default_halfp,
		'training-default-bnb': args.training_default_bnb,
	}

	os.makedirs('./config/', exist_ok=True)
	with open(f'./config/exec.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps(settings, indent='\t') )



def import_generate_settings(file="./config/generate.json"):
	settings, _ = read_generate_settings(file, read_latents=False)
	
	if settings is None:
		return None

	return (
		None if 'text' not in settings else settings['text'],
		None if 'delimiter' not in settings else settings['delimiter'],
		None if 'emotion' not in settings else settings['emotion'],
		None if 'prompt' not in settings else settings['prompt'],
		None if 'voice' not in settings else settings['voice'],
		None,
		None,
		None if 'seed' not in settings else settings['seed'],
		None if 'candidates' not in settings else settings['candidates'],
		None if 'num_autoregressive_samples' not in settings else settings['num_autoregressive_samples'],
		None if 'diffusion_iterations' not in settings else settings['diffusion_iterations'],
		0.8 if 'temperature' not in settings else settings['temperature'],
		"DDIM" if 'diffusion_sampler' not in settings else settings['diffusion_sampler'],
		8   if 'breathing_room' not in settings else settings['breathing_room'],
		0.0 if 'cvvp_weight' not in settings else settings['cvvp_weight'],
		0.8 if 'top_p' not in settings else settings['top_p'],
		1.0 if 'diffusion_temperature' not in settings else settings['diffusion_temperature'],
		1.0 if 'length_penalty' not in settings else settings['length_penalty'],
		2.0 if 'repetition_penalty' not in settings else settings['repetition_penalty'],
		2.0 if 'cond_free_k' not in settings else settings['cond_free_k'],
		None if 'experimentals' not in settings else settings['experimentals'],
	)


def reset_generation_settings():
	with open(f'./config/generate.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps({}, indent='\t') )
	return import_generate_settings()

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

def load_tts( restart=False, model=None ):
	global args
	global tts

	if restart:
		unload_tts()


	if model:
		args.autoregressive_model = model

	print(f"Loading TorToiSe... (using model: {args.autoregressive_model})")

	tts_loading = True
	try:
		tts = TextToSpeech(minor_optimizations=not args.low_vram, autoregressive_model_path=args.autoregressive_model)
	except Exception as e:
		tts = TextToSpeech(minor_optimizations=not args.low_vram)
		load_autoregressive_model(args.autoregressive_model)

	if not hasattr(tts, 'autoregressive_model_hash'):
		tts.autoregressive_model_hash = hash_file(tts.autoregressive_model_path)

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

	print(f"Loading model: {autoregressive_model_path}")

	if hasattr(tts, 'load_autoregressive_model') and tts.load_autoregressive_model(autoregressive_model_path):
		tts.load_autoregressive_model(autoregressive_model_path)
	# polyfill in case a user did NOT update the packages
	# this shouldn't happen anymore, as I just clone mrq/tortoise-tts, and inject it into sys.path
	else:
		from tortoise.models.autoregressive import UnifiedVoice

		tts.autoregressive_model_path = autoregressive_model_path if autoregressive_model_path and os.path.exists(autoregressive_model_path) else get_model_path('autoregressive.pth', tts.models_dir)

		del tts.autoregressive
		tts.autoregressive = UnifiedVoice(max_mel_tokens=604, max_text_tokens=402, max_conditioning_inputs=2, layers=30,
										  model_dim=1024,
										  heads=16, number_text_tokens=255, start_text_token=255, checkpointing=False,
										  train_solo_embeddings=False).cpu().eval()
		tts.autoregressive.load_state_dict(torch.load(tts.autoregressive_model_path))
		tts.autoregressive.post_init_gpt2_config(kv_cache=tts.use_kv_cache)
		if tts.preloaded_tensors:
			tts.autoregressive = tts.autoregressive.to(tts.device)

	if not hasattr(tts, 'autoregressive_model_hash'):
		tts.autoregressive_model_hash = hash_file(autoregressive_model_path)

	print(f"Loaded model: {tts.autoregressive_model_path}")

	do_gc()
	
	return autoregressive_model_path

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

def unload_voicefixer():
	global voicefixer

	if voicefixer:
		del voicefixer
		voicefixer = None
		print("Unloaded Voicefixer")

	do_gc()

def load_whisper_model(language=None, model_name=None, progress=None):
	global whisper_model
	
	if not model_name:
		model_name = args.whisper_model
	else:
		args.whisper_model = model_name
		save_args_settings()

	if language and f'{model_name}.{language}' in WHISPER_SPECIALIZED_MODELS:
		model_name = f'{model_name}.{language}'
		print(f"Loading specialized model for language: {language}")

	notify_progress(f"Loading Whisper model: {model_name}", progress)

	if args.whisper_cpp:
		from whispercpp import Whisper
		if not language:
			language = 'auto'

		b_lang = language.encode('ascii')
		whisper_model = Whisper(model_name, models_dir='./models/', language=b_lang)
	else:
		import whisper
		whisper_model = whisper.load_model(model_name)

	print("Loaded Whisper model")

def unload_whisper():
	global whisper_model

	if whisper_model:
		del whisper_model
		whisper_model = None
		print("Unloaded Whisper")

	do_gc()