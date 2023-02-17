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

import torch
import torchaudio
import music_tag
import gradio as gr
import gradio.utils

from datetime import datetime

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices, get_voice_dir
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.device import get_device_name, set_device_name


args = None
tts = None
webui = None
voicefixer = None
whisper = None
dlas = None

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
		'voice-fixer': True,
		'voice-fixer-use-cuda': True,
		'force-cpu-for-conditioning-latents': False,
		'device-override': None,
		'concurrency-count': 2,
		'output-sample-rate': 44100,
		'output-volume': 1,
	}

	if os.path.isfile('./config/exec.json'):
		with open(f'./config/exec.json', 'r', encoding="utf-8") as f:
			overrides = json.load(f)
			for k in overrides:
				default_arguments[k] = overrides[k]

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
	parser.add_argument("--device-override", default=default_arguments['device-override'], help="A device string to override pass through Torch")
	parser.add_argument("--sample-batch-size", default=default_arguments['sample-batch-size'], type=int, help="Sets how many batches to use during the autoregressive samples pass")
	parser.add_argument("--concurrency-count", type=int, default=default_arguments['concurrency-count'], help="How many Gradio events to process at once")
	parser.add_argument("--output-sample-rate", type=int, default=default_arguments['output-sample-rate'], help="Sample rate to resample the output to (from 24KHz)")
	parser.add_argument("--output-volume", type=float, default=default_arguments['output-volume'], help="Adjusts volume of output")
	args = parser.parse_args()

	args.embed_output_metadata = not args.no_embed_output_metadata

	set_device_name(args.device_override)

	args.listen_host = None
	args.listen_port = None
	args.listen_path = None
	if args.listen:
		try:
			match = re.findall(r"^(?:(.+?):(\d+))?(\/.+?)?$", args.listen)[0]

			args.listen_host = match[0] if match[0] != "" else "127.0.0.1"
			args.listen_port = match[1] if match[1] != "" else None
			args.listen_path = match[2] if match[2] != "" else "/"
		except Exception as e:
			pass

	if args.listen_port is not None:
		args.listen_port = int(args.listen_port)
	
	return args

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

	try:
		tts
	except NameError:
		raise gr.Error("TTS is still initializing...")

	if voice != "microphone":
		voices = [voice]
	else:
		voices = []

	if voice == "microphone":
		if mic_audio is None:
			raise gr.Error("Please provide audio from mic when choosing `microphone` as a voice input")
		mic = load_audio(mic_audio, tts.input_sample_rate)
		voice_samples, conditioning_latents = [mic], None
	elif voice == "random":
		voice_samples, conditioning_latents = None, tts.get_random_conditioning_latents()
	else:
		progress(0, desc="Loading voice...")
		voice_samples, conditioning_latents = load_voice(voice)

	if voice_samples is not None:
		sample_voice = torch.cat(voice_samples, dim=-1).squeeze().cpu()

		conditioning_latents = tts.get_conditioning_latents(voice_samples, return_mels=not args.latents_lean_and_mean, progress=progress, slices=voice_latents_chunks, force_cpu=args.force_cpu_for_conditioning_latents)
		if len(conditioning_latents) == 4:
			conditioning_latents = (conditioning_latents[0], conditioning_latents[1], conditioning_latents[2], None)
			
		if voice != "microphone":
			torch.save(conditioning_latents, f'{get_voice_dir()}/{voice}/cond_latents.pth')
		voice_samples = None
	else:
		if conditioning_latents is not None:
			sample_voice, _ = load_voice(voice, load_latents=False)
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

	if delimiter == "\\n":
		delimiter = "\n"

	if delimiter != "" and delimiter in text:
		texts = text.split(delimiter)
	else:
		texts = split_and_recombine_text(text)
 
	full_start_time = time.time()
 
	outdir = f"./results/{voice}/"
	os.makedirs(outdir, exist_ok=True)

	audio_cache = {}

	resample = None
	# not a ternary in the event for some reason I want to rely on librosa's upsampling interpolator rather than torchaudio's, for some reason
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

	# I know there's something to pad I don't care
	pad = ""
	for i in range(4,0,-1):
		if idx < 10 ** i:
			pad = f"{pad}0"
	idx = f"{pad}{idx}"

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
			if prompt.strip() != "":
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
		'delimiter': '\\n' if delimiter == "\n" else delimiter,
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
	}

	# kludgy yucky codesmells
	for name in audio_cache:
		if 'output' not in audio_cache[name]:
			continue

		output_voices.append(f'{outdir}/{voice}_{name}.wav')
		with open(f'{outdir}/{voice}_{name}.json', 'w', encoding="utf-8") as f:
			f.write(json.dumps(info, indent='\t') )

	if args.voice_fixer and voicefixer:
		fixed_output_voices = []
		for path in progress.tqdm(output_voices, desc="Running voicefix..."):
			fixed = path.replace(".wav", "_fixed.wav")
			voicefixer.restore(
				input=path,
				output=fixed,
				cuda=get_device_name() == "cuda" and args.voice_fixer_use_cuda,
				#mode=mode,
			)
			fixed_output_voices.append(fixed)
		output_voices = fixed_output_voices

	if voice is not None and conditioning_latents is not None:
		with open(f'{get_voice_dir()}/{voice}/cond_latents.pth', 'rb') as f:
			info['latents'] = base64.b64encode(f.read()).decode("ascii")

	if args.embed_output_metadata:
		for name in progress.tqdm(audio_cache, desc="Embedding metadata..."):
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

def setup_tortoise(restart=False):
	global args
	global tts
	global voicefixer

	if args.voice_fixer and not restart:
		try:
			from voicefixer import VoiceFixer
			print("Initializating voice-fixer")
			voicefixer = VoiceFixer()
			print("initialized voice-fixer")
		except Exception as e:
			print(f"Error occurred while tring to initialize voicefixer: {e}")

	print("Initializating TorToiSe...")
	tts = TextToSpeech(minor_optimizations=not args.low_vram)
	print("TorToiSe initialized, ready for generation.")
	return tts