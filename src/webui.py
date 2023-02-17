import os
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

import tortoise.api
from tortoise.utils.audio import get_voice_dir, get_voices

from utils import *

args = setup_args()

def run_generation(
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
	progress=gr.Progress(track_tqdm=True)
):
	try:
		sample, outputs, stats = generate(
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
			progress
		)
	except Exception as e:
		message = str(e)
		if message == "Kill signal detected":
			reload_tts()

		raise gr.Error(message)
	

	return (
		outputs[0],
		gr.update(value=sample, visible=sample is not None),
		gr.update(choices=outputs, value=outputs[0], visible=len(outputs) > 1, interactive=True),
		gr.update(visible=len(outputs) > 1),
		gr.update(value=stats, visible=True),
	)

def compute_latents(voice, voice_latents_chunks, progress=gr.Progress(track_tqdm=True)):
	global tts
	global args

	try:
		tts
	except NameError:
		raise gr.Error("TTS is still initializing...")

	voice_samples, conditioning_latents = load_voice(voice, load_latents=False)

	if voice_samples is None:
		return

	conditioning_latents = tts.get_conditioning_latents(voice_samples, return_mels=not args.latents_lean_and_mean, progress=progress, slices=voice_latents_chunks, force_cpu=args.force_cpu_for_conditioning_latents)

	if len(conditioning_latents) == 4:
		conditioning_latents = (conditioning_latents[0], conditioning_latents[1], conditioning_latents[2], None)
			
	torch.save(conditioning_latents, f'{get_voice_dir()}/{voice}/cond_latents.pth')

	return voice

def update_presets(value):
	PRESETS = {
		'Ultra Fast': {'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
		'Fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
		'Standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
		'High Quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
	}
	
	if value in PRESETS:
		preset = PRESETS[value]
		return (gr.update(value=preset['num_autoregressive_samples']), gr.update(value=preset['diffusion_iterations']))
	else:
		return (gr.update(), gr.update())

def setup_gradio():
	global args
	global ui
	
	if not args.share:
		def noop(function, return_value=None):
			def wrapped(*args, **kwargs):
				return return_value
			return wrapped
		gradio.utils.version_check = noop(gradio.utils.version_check)
		gradio.utils.initiated_analytics = noop(gradio.utils.initiated_analytics)
		gradio.utils.launch_analytics = noop(gradio.utils.launch_analytics)
		gradio.utils.integration_analytics = noop(gradio.utils.integration_analytics)
		gradio.utils.error_analytics = noop(gradio.utils.error_analytics)
		gradio.utils.log_feature_analytics = noop(gradio.utils.log_feature_analytics)
		#gradio.utils.get_local_ip_address = noop(gradio.utils.get_local_ip_address, 'localhost')

	if args.models_from_local_only:
		os.environ['TRANSFORMERS_OFFLINE']='1'

	with gr.Blocks() as ui:
		with gr.Tab("Generate"):
			with gr.Row():
				with gr.Column():
					text = gr.Textbox(lines=4, label="Prompt")
			with gr.Row():
				with gr.Column():
					delimiter = gr.Textbox(lines=1, label="Line Delimiter", placeholder="\\n")

					emotion = gr.Radio(
						["Happy", "Sad", "Angry", "Disgusted", "Arrogant", "Custom"],
						value="Custom",
						label="Emotion",
						type="value",
						interactive=True
					)
					prompt = gr.Textbox(lines=1, label="Custom Emotion + Prompt (if selected)")
					voice = gr.Dropdown(
						get_voice_list(),
						label="Voice",
						type="value",
					)
					mic_audio = gr.Audio(
						label="Microphone Source",
						source="microphone",
						type="filepath",
					)
					refresh_voices = gr.Button(value="Refresh Voice List")
					voice_latents_chunks = gr.Slider(label="Voice Chunks", minimum=1, maximum=64, value=1, step=1)
					recompute_voice_latents = gr.Button(value="(Re)Compute Voice Latents")
					recompute_voice_latents.click(compute_latents,
						inputs=[
							voice,
							voice_latents_chunks,
						],
						outputs=voice,
					)
					
					prompt.change(fn=lambda value: gr.update(value="Custom"),
						inputs=prompt,
						outputs=emotion
					)
					mic_audio.change(fn=lambda value: gr.update(value="microphone"),
						inputs=mic_audio,
						outputs=voice
					)
				with gr.Column():
					candidates = gr.Slider(value=1, minimum=1, maximum=6, step=1, label="Candidates")
					seed = gr.Number(value=0, precision=0, label="Seed")

					preset = gr.Radio(
						["Ultra Fast", "Fast", "Standard", "High Quality"],
						label="Preset",
						type="value",
					)
					num_autoregressive_samples = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Samples")
					diffusion_iterations = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Iterations")

					temperature = gr.Slider(value=0.2, minimum=0, maximum=1, step=0.1, label="Temperature")
					breathing_room = gr.Slider(value=8, minimum=1, maximum=32, step=1, label="Pause Size")
					diffusion_sampler = gr.Radio(
						["P", "DDIM"], # + ["K_Euler_A", "DPM++2M"],
						value="P",
						label="Diffusion Samplers",
						type="value",
					)

					preset.change(fn=update_presets,
						inputs=preset,
						outputs=[
							num_autoregressive_samples,
							diffusion_iterations,
						],
					)

					show_experimental_settings = gr.Checkbox(label="Show Experimental Settings")
					reset_generation_settings_button = gr.Button(value="Reset to Default")
				with gr.Column(visible=False) as col:
					experimental_column = col

					experimental_checkboxes = gr.CheckboxGroup(["Half Precision", "Conditioning-Free"], value=["Conditioning-Free"], label="Experimental Flags")
					cvvp_weight = gr.Slider(value=0, minimum=0, maximum=1, label="CVVP Weight")
					top_p = gr.Slider(value=0.8, minimum=0, maximum=1, label="Top P")
					diffusion_temperature = gr.Slider(value=1.0, minimum=0, maximum=1, label="Diffusion Temperature")
					length_penalty = gr.Slider(value=1.0, minimum=0, maximum=8, label="Length Penalty")
					repetition_penalty = gr.Slider(value=2.0, minimum=0, maximum=8, label="Repetition Penalty")
					cond_free_k = gr.Slider(value=2.0, minimum=0, maximum=4, label="Conditioning-Free K")

					show_experimental_settings.change(
						fn=lambda x: gr.update(visible=x),
						inputs=show_experimental_settings,
						outputs=experimental_column
					)
				with gr.Column():
					submit = gr.Button(value="Generate")
					stop = gr.Button(value="Stop")

					generation_results = gr.Dataframe(label="Results", headers=["Seed", "Time"], visible=False)
					source_sample = gr.Audio(label="Source Sample", visible=False)
					output_audio = gr.Audio(label="Output")
					candidates_list = gr.Dropdown(label="Candidates", type="value", visible=False)
					output_pick = gr.Button(value="Select Candidate", visible=False)
					
		with gr.Tab("History"):
			with gr.Row():
				with gr.Column():
					headers = {
						"Name": "",
						"Samples": "num_autoregressive_samples",
						"Iterations": "diffusion_iterations",
						"Temp.": "temperature",
						"Sampler": "diffusion_sampler",
						"CVVP": "cvvp_weight",
						"Top P": "top_p",
						"Diff. Temp.": "diffusion_temperature",
						"Len Pen": "length_penalty",
						"Rep Pen": "repetition_penalty",
						"Cond-Free K": "cond_free_k",
						"Time": "time",
					}
					history_info = gr.Dataframe(label="Results", headers=list(headers.keys()))
			with gr.Row():
				with gr.Column():
					history_voices = gr.Dropdown(
						get_voice_list("./results/"),
						label="Voice",
						type="value",
					)

					history_view_results_button = gr.Button(value="View Files")
				with gr.Column():
					history_results_list = gr.Dropdown(label="Results",type="value", interactive=True)
					history_view_result_button = gr.Button(value="View File")
				with gr.Column():
					history_audio = gr.Audio()
					history_copy_settings_button = gr.Button(value="Copy Settings")
				
				def history_view_results( voice ):
					results = []
					files = []
					outdir = f"./results/{voice}/"
					for i, file in enumerate(sorted(os.listdir(outdir))):
						if file[-4:] != ".wav":
							continue

						metadata, _ = read_generate_settings(f"{outdir}/{file}", read_latents=False)
						if metadata is None:
							continue
							
						values = []
						for k in headers:
							v = file
							if k != "Name":
								v = metadata[headers[k]]
							values.append(v)


						files.append(file)
						results.append(values)

					return (
						results,
						gr.Dropdown.update(choices=sorted(files))
					)

				history_view_results_button.click(
					fn=history_view_results,
					inputs=history_voices,
					outputs=[
						history_info,
						history_results_list,
					]
				)
				history_view_result_button.click(
					fn=lambda voice, file: f"./results/{voice}/{file}",
					inputs=[
						history_voices,
						history_results_list,
					],
					outputs=history_audio
				)
		with gr.Tab("Utilities"):
			with gr.Row():
				with gr.Column():
					audio_in = gr.File(type="file", label="Audio Input", file_types=["audio"])
					copy_button = gr.Button(value="Copy Settings")
					import_voice_name = gr.Textbox(label="Voice Name")
					import_voice_button = gr.Button(value="Import Voice")
				with gr.Column():
					metadata_out = gr.JSON(label="Audio Metadata")
					latents_out = gr.File(type="binary", label="Voice Latents")

					def read_generate_settings_proxy(file, saveAs='.temp'):
						j, latents = read_generate_settings(file)

						if latents:
							outdir = f'{get_voice_dir()}/{saveAs}/'
							os.makedirs(outdir, exist_ok=True)
							with open(f'{outdir}/cond_latents.pth', 'wb') as f:
								f.write(latents)
							
							latents = f'{outdir}/cond_latents.pth'

						return (
							j,
							gr.update(value=latents, visible=latents is not None),
							None if j is None else j['voice']
						)

					audio_in.upload(
						fn=read_generate_settings_proxy,
						inputs=audio_in,
						outputs=[
							metadata_out,
							latents_out,
							import_voice_name
						]
					)

				import_voice_button.click(
					fn=import_voice,
					inputs=[
						audio_in,
						import_voice_name,
					]
				)
		with gr.Tab("Training"):
			with gr.Tab("Prepare Dataset"):
				with gr.Row():
					with gr.Column():
						dataset_settings = [
							gr.Dropdown( get_voice_list(), label="Dataset Source", type="value" ),
							gr.Textbox(label="Language", placeholder="English")
						]
						dataset_voices = dataset_settings[0]

					with gr.Column():
						prepare_dataset_button = gr.Button(value="Prepare")

						def prepare_dataset_proxy( voice, language ):
							return prepare_dataset( get_voices(load_latents=False)[voice], outdir=f"./training/{voice}/", language=language )

						prepare_dataset_button.click(
							prepare_dataset_proxy,
							inputs=dataset_settings,
							outputs=None
						)
			with gr.Tab("Generate Configuration"):
				with gr.Row():
					with gr.Column():
						training_settings = [
							gr.Slider(label="Batch Size", value=128),
							gr.Slider(label="Learning Rate", value=1e-5, minimum=0, maximum=1e-4, step=1e-6),
							gr.Number(label="Print Frequency", value=50),
							gr.Number(label="Save Frequency", value=50),
						]
						save_yaml_button = gr.Button(value="Save Training Configuration")
					with gr.Column():
						training_settings = training_settings + [
							gr.Textbox(label="Training Name", placeholder="finetune"),
							gr.Textbox(label="Dataset Name", placeholder="finetune"),
							gr.Textbox(label="Dataset Path", placeholder="./training/finetune/train.txt"),
							gr.Textbox(label="Validation Name", placeholder="finetune"),
							gr.Textbox(label="Validation Path", placeholder="./training/finetune/train.txt"),
						]

						save_yaml_button.click(save_training_settings,
							inputs=training_settings,
							outputs=None
						)
		with gr.Tab("Settings"):
			with gr.Row():
				exec_inputs = []
				with gr.Column():
					exec_inputs = exec_inputs + [
						gr.Textbox(label="Listen", value=args.listen, placeholder="127.0.0.1:7860/"),
						gr.Checkbox(label="Public Share Gradio", value=args.share),
						gr.Checkbox(label="Check For Updates", value=args.check_for_updates),
						gr.Checkbox(label="Only Load Models Locally", value=args.models_from_local_only),
						gr.Checkbox(label="Low VRAM", value=args.low_vram),
						gr.Checkbox(label="Embed Output Metadata", value=args.embed_output_metadata),
						gr.Checkbox(label="Slimmer Computed Latents", value=args.latents_lean_and_mean),
						gr.Checkbox(label="Voice Fixer", value=args.voice_fixer),
						gr.Checkbox(label="Use CUDA for Voice Fixer", value=args.voice_fixer_use_cuda),
						gr.Checkbox(label="Force CPU for Conditioning Latents", value=args.force_cpu_for_conditioning_latents),
						gr.Textbox(label="Device Override", value=args.device_override),
						gr.Dropdown(label="Whisper Model", value=args.whisper_model, choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large"]),
					]
					gr.Button(value="Check for Updates").click(check_for_updates)
					gr.Button(value="Reload TTS").click(reload_tts)
				with gr.Column():
					exec_inputs = exec_inputs + [
						gr.Number(label="Sample Batch Size", precision=0, value=args.sample_batch_size),
						gr.Number(label="Concurrency Count", precision=0, value=args.concurrency_count),
						gr.Number(label="Ouptut Sample Rate", precision=0, value=args.output_sample_rate),
						gr.Slider(label="Ouptut Volume", minimum=0, maximum=2, value=args.output_volume),
					]

				for i in exec_inputs:
					i.change(
						fn=export_exec_settings,
						inputs=exec_inputs
					)

		input_settings = [
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
		]

		# YUCK
		def update_voices():
			return (
				gr.Dropdown.update(choices=get_voice_list()),
				gr.Dropdown.update(choices=get_voice_list()),
				gr.Dropdown.update(choices=get_voice_list("./results/")),
			)

		def history_copy_settings( voice, file ):
			return import_generate_settings( f"./results/{voice}/{file}" )

		refresh_voices.click(update_voices,
			inputs=None,
			outputs=[
				voice,
				dataset_voices,
				history_voices
			]
		)

		output_pick.click(
			lambda x: x,
			inputs=candidates_list,
			outputs=output_audio,
		)

		submit.click(
			lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
			outputs=[source_sample, candidates_list, output_pick, generation_results],
		)

		submit_event = submit.click(run_generation,
			inputs=input_settings,
			outputs=[output_audio, source_sample, candidates_list, output_pick, generation_results],
		)


		copy_button.click(import_generate_settings,
			inputs=audio_in, # JSON elements cannot be used as inputs
			outputs=input_settings
		)

		reset_generation_settings_button.click(
			fn=reset_generation_settings,
			inputs=None,
			outputs=input_settings
		)

		history_copy_settings_button.click(history_copy_settings,
			inputs=[
				history_voices,
				history_results_list,
			],
			outputs=input_settings
		)

		if os.path.isfile('./config/generate.json'):
			ui.load(import_generate_settings, inputs=None, outputs=input_settings)
		
		if args.check_for_updates:
			ui.load(check_for_updates)

		stop.click(fn=cancel_generate, inputs=None, outputs=None, cancels=[submit_event])


	ui.queue(concurrency_count=args.concurrency_count)
	webui = ui
	return webui