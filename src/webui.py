import os
import argparse
import time
import json
import base64
import re
import inspect
import urllib.request

import torch
import torchaudio
import music_tag
import gradio as gr
import gradio.utils

from datetime import datetime

import tortoise.api
from tortoise.utils.audio import get_voice_dir, get_voices
from tortoise.utils.device import get_device_count

from utils import *

args = setup_args()

GENERATE_SETTINGS = {}
TRANSCRIBE_SETTINGS = {}
EXEC_SETTINGS = {}
TRAINING_SETTINGS = {}
GENERATE_SETTINGS_ARGS = []

PRESETS = {
	'Ultra Fast': {'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
	'Fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
	'Standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
	'High Quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
}

HISTORY_HEADERS = {
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
	"Datetime": "datetime",
	"Model": "model",
	"Model Hash": "model_hash",
}

# can't use *args OR **kwargs if I want to retain the ability to use progress
def generate_proxy(
	text,
	delimiter,
	emotion,
	prompt,
	voice,
	mic_audio,
	voice_latents_chunks,
	candidates,
	seed,
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
	experimentals,
	progress=gr.Progress(track_tqdm=True)
):
	kwargs = locals()

	try:
		sample, outputs, stats = generate(**kwargs)
	except Exception as e:
		message = str(e)
		if message == "Kill signal detected":
			unload_tts()

		raise e
	
	return (
		outputs[0],
		gr.update(value=sample, visible=sample is not None),
		gr.update(choices=outputs, value=outputs[0], visible=len(outputs) > 1, interactive=True),
		gr.update(value=stats, visible=True),
	)


def update_presets(value):
	if value in PRESETS:
		preset = PRESETS[value]
		return (gr.update(value=preset['num_autoregressive_samples']), gr.update(value=preset['diffusion_iterations']))
	else:
		return (gr.update(), gr.update())

def get_training_configs():
	configs = []
	for i, file in enumerate(sorted(os.listdir(f"./training/"))):
		if file[-5:] != ".yaml" or file[0] == ".":
			continue
		configs.append(f"./training/{file}")

	return configs

def update_training_configs():
	return gr.update(choices=get_training_list())

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
		for k in HISTORY_HEADERS:
			v = file
			if k != "Name":
				v = metadata[HISTORY_HEADERS[k]] if HISTORY_HEADERS[k] in metadata else '?'
			values.append(v)


		files.append(file)
		results.append(values)

	return (
		results,
		gr.Dropdown.update(choices=sorted(files))
	)

def import_generate_settings_proxy( file=None ):
	global GENERATE_SETTINGS_ARGS
	settings = import_generate_settings( file )

	res = []
	for k in GENERATE_SETTINGS_ARGS:
		res.append(settings[k] if k in settings else None)
	print(GENERATE_SETTINGS_ARGS)
	print(settings)
	print(res)
	return tuple(res)

def compute_latents_proxy(voice, voice_latents_chunks, progress=gr.Progress(track_tqdm=True)):
	compute_latents( voice=voice, voice_latents_chunks=voice_latents_chunks, progress=progress )
	return voice


def import_voices_proxy(files, name, progress=gr.Progress(track_tqdm=True)):
	import_voices(files, name, progress)
	return gr.update()

def read_generate_settings_proxy(file, saveAs='.temp'):
	j, latents = read_generate_settings(file)

	if latents:
		outdir = f'{get_voice_dir()}/{saveAs}/'
		os.makedirs(outdir, exist_ok=True)
		with open(f'{outdir}/cond_latents.pth', 'wb') as f:
			f.write(latents)
		
		latents = f'{outdir}/cond_latents.pth'

	return (
		gr.update(value=j, visible=j is not None),
		gr.update(value=latents, visible=latents is not None),
		None if j is None else j['voice'],
		gr.update(visible=j is not None),
	)

def prepare_dataset_proxy( voice, language, validation_size, skip_existings, progress=gr.Progress(track_tqdm=True) ):
	messages = []
	message = prepare_dataset( get_voices(load_latents=False)[voice], outdir=f"./training/{voice}/", language=language, skip_existings=skip_existings, progress=progress )
	messages.append(message)
	if validation_size > 0:
		message = prepare_validation_dataset( voice, text_length=validation_size )
		messages.append(message)
	return "\n".join(messages)

def update_args_proxy( *args ):
	kwargs = {}
	keys = list(EXEC_SETTINGS.keys())
	for i in range(len(args)):
		k = keys[i]
		v = args[i]
		kwargs[k] = v

	update_args(**kwargs)
def optimize_training_settings_proxy( *args ):
	kwargs = {}
	keys = list(TRAINING_SETTINGS.keys())
	for i in range(len(args)):
		k = keys[i]
		v = args[i]
		kwargs[k] = v

	settings, messages = optimize_training_settings(**kwargs)
	output = list(settings.values())
	return output[:-1] + ["\n".join(messages)]

def import_training_settings_proxy( voice ):
	messages = []
	injson = f'./training/{voice}/train.json'
	statedir = f'./training/{voice}/finetune/training_state/'
	output = {}

	try:
		with open(injson, 'r', encoding="utf-8") as f:
			settings = json.loads(f.read())
	except:
		messages.append(f"Error import /{voice}/train.json")

		for k in TRAINING_SETTINGS:
			output[k] = TRAINING_SETTINGS[k].value

		output = list(output.values())
		return output[:-1] + ["\n".join(messages)]

	if os.path.isdir(statedir):
		resumes = sorted([int(d[:-6]) for d in os.listdir(statedir) if d[-6:] == ".state" ])

		if len(resumes) > 0:
			settings['resume_state'] = f'{statedir}/{resumes[-1]}.state'
			messages.append(f"Found most recent training state: {settings['resume_state']}")

	output = {}
	for k in TRAINING_SETTINGS:
		if k not in settings:
			continue
		output[k] = settings[k]

	output = list(output.values())
	print(list(TRAINING_SETTINGS.keys()))
	print(output)
	messages.append(f"Imported training settings: {injson}")

	return output[:-1] + ["\n".join(messages)]

def save_training_settings_proxy( *args ):
	kwargs = {}
	keys = list(TRAINING_SETTINGS.keys())
	for i in range(len(args)):
		k = keys[i]
		v = args[i]
		kwargs[k] = v

	settings, messages = save_training_settings(**kwargs)
	return "\n".join(messages)

def update_voices():
	return (
		gr.Dropdown.update(choices=get_voice_list(append_defaults=True)),
		gr.Dropdown.update(choices=get_voice_list()),
		gr.Dropdown.update(choices=get_voice_list("./results/")),
	)

def history_copy_settings( voice, file ):
	return import_generate_settings( f"./results/{voice}/{file}" )

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

	voice_list_with_defaults = get_voice_list(append_defaults=True)
	voice_list = get_voice_list()
	result_voices = get_voice_list("./results/")
	autoregressive_models = get_autoregressive_models()
	dataset_list = get_dataset_list()

	global GENERATE_SETTINGS_ARGS
	GENERATE_SETTINGS_ARGS = list(inspect.signature(generate_proxy).parameters.keys())[:-1]
	for i in range(len(GENERATE_SETTINGS_ARGS)):
		arg = GENERATE_SETTINGS_ARGS[i]
		GENERATE_SETTINGS[arg] = None

	with gr.Blocks() as ui:
		with gr.Tab("Generate"):
			with gr.Row():
				with gr.Column():
					GENERATE_SETTINGS["text"] = gr.Textbox(lines=4, label="Input Prompt")
			with gr.Row():
				with gr.Column():
					GENERATE_SETTINGS["delimiter"] = gr.Textbox(lines=1, label="Line Delimiter", placeholder="\\n")

					GENERATE_SETTINGS["emotion"] = gr.Radio( ["Happy", "Sad", "Angry", "Disgusted", "Arrogant", "Custom", "None"], value="None", label="Emotion", type="value", interactive=True )
					GENERATE_SETTINGS["prompt"] = gr.Textbox(lines=1, label="Custom Emotion", visible=False)
					GENERATE_SETTINGS["voice"] = gr.Dropdown(choices=voice_list_with_defaults, label="Voice", type="value", value=voice_list_with_defaults[0]) # it'd be very cash money if gradio was able to default to the first value in the list without this shit
					GENERATE_SETTINGS["mic_audio"] = gr.Audio( label="Microphone Source", source="microphone", type="filepath", visible=False )
					GENERATE_SETTINGS["voice_latents_chunks"] = gr.Number(label="Voice Chunks", precision=0, value=0)
					with gr.Row():
						refresh_voices = gr.Button(value="Refresh Voice List")
						recompute_voice_latents = gr.Button(value="(Re)Compute Voice Latents")

					GENERATE_SETTINGS["voice"].change(
						fn=update_baseline_for_latents_chunks,
						inputs=GENERATE_SETTINGS["voice"],
						outputs=GENERATE_SETTINGS["voice_latents_chunks"]
					)
					GENERATE_SETTINGS["voice"].change(
						fn=lambda value: gr.update(visible=value == "microphone"),
						inputs=GENERATE_SETTINGS["voice"],
						outputs=GENERATE_SETTINGS["mic_audio"],
					)
				with gr.Column():
					GENERATE_SETTINGS["candidates"] = gr.Slider(value=1, minimum=1, maximum=6, step=1, label="Candidates")
					GENERATE_SETTINGS["seed"] = gr.Number(value=0, precision=0, label="Seed")

					preset = gr.Radio( ["Ultra Fast", "Fast", "Standard", "High Quality"], label="Preset", type="value" )

					GENERATE_SETTINGS["num_autoregressive_samples"] = gr.Slider(value=128, minimum=2, maximum=512, step=1, label="Samples")
					GENERATE_SETTINGS["diffusion_iterations"] = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Iterations")

					GENERATE_SETTINGS["temperature"] = gr.Slider(value=0.2, minimum=0, maximum=1, step=0.1, label="Temperature")
					
					show_experimental_settings = gr.Checkbox(label="Show Experimental Settings")
					reset_generation_settings_button = gr.Button(value="Reset to Default")
				with gr.Column(visible=False) as col:
					experimental_column = col

					GENERATE_SETTINGS["experimentals"] = gr.CheckboxGroup(["Half Precision", "Conditioning-Free"], value=["Conditioning-Free"], label="Experimental Flags")
					GENERATE_SETTINGS["breathing_room"] = gr.Slider(value=8, minimum=1, maximum=32, step=1, label="Pause Size")
					GENERATE_SETTINGS["diffusion_sampler"] = gr.Radio(
						["P", "DDIM"], # + ["K_Euler_A", "DPM++2M"],
						value="DDIM", label="Diffusion Samplers", type="value"
					)
					GENERATE_SETTINGS["cvvp_weight"] = gr.Slider(value=0, minimum=0, maximum=1, label="CVVP Weight")
					GENERATE_SETTINGS["top_p"] = gr.Slider(value=0.8, minimum=0, maximum=1, label="Top P")
					GENERATE_SETTINGS["diffusion_temperature"] = gr.Slider(value=1.0, minimum=0, maximum=1, label="Diffusion Temperature")
					GENERATE_SETTINGS["length_penalty"] = gr.Slider(value=1.0, minimum=0, maximum=8, label="Length Penalty")
					GENERATE_SETTINGS["repetition_penalty"] = gr.Slider(value=2.0, minimum=0, maximum=8, label="Repetition Penalty")
					GENERATE_SETTINGS["cond_free_k"] = gr.Slider(value=2.0, minimum=0, maximum=4, label="Conditioning-Free K")
				with gr.Column():
					with gr.Row():
						submit = gr.Button(value="Generate")
						stop = gr.Button(value="Stop")

					generation_results = gr.Dataframe(label="Results", headers=["Seed", "Time"], visible=False)
					source_sample = gr.Audio(label="Source Sample", visible=False)
					output_audio = gr.Audio(label="Output")
					candidates_list = gr.Dropdown(label="Candidates", type="value", visible=False, choices=[""], value="")

					def change_candidate( val ):
						if not val:
							return
						return val

					candidates_list.change(
						fn=change_candidate,
						inputs=candidates_list,
						outputs=output_audio,
					)
		with gr.Tab("History"):
			with gr.Row():
				with gr.Column():
					history_info = gr.Dataframe(label="Results", headers=list(HISTORY_HEADERS.keys()))
			with gr.Row():
				with gr.Column():
					history_voices = gr.Dropdown(choices=result_voices, label="Voice", type="value", value=result_voices[0] if len(result_voices) > 0 else "")
				with gr.Column():
					history_results_list = gr.Dropdown(label="Results",type="value", interactive=True, value="")
				with gr.Column():
					history_audio = gr.Audio()
					history_copy_settings_button = gr.Button(value="Copy Settings")
		with gr.Tab("Utilities"):
			with gr.Row():
				with gr.Column():
					audio_in = gr.Files(type="file", label="Audio Input", file_types=["audio"])
					import_voice_name = gr.Textbox(label="Voice Name")
					import_voice_button = gr.Button(value="Import Voice")
				with gr.Column(visible=False) as col:
					utilities_metadata_column = col

					metadata_out = gr.JSON(label="Audio Metadata")
					copy_button = gr.Button(value="Copy Settings")
					latents_out = gr.File(type="binary", label="Voice Latents")
		with gr.Tab("Training"):
			with gr.Tab("Prepare Dataset"):
				with gr.Row():
					with gr.Column():
						DATASET_SETTINGS = {}
						DATASET_SETTINGS['voice'] = gr.Dropdown( choices=voice_list, label="Dataset Source", type="value", value=voice_list[0] if len(voice_list) > 0 else "" )
						with gr.Row():
							DATASET_SETTINGS['language'] = gr.Textbox(label="Language", value="en")
							DATASET_SETTINGS['validation_size'] = gr.Number(label="Validation Text Length Cull Size", value=12, precision=0)
						DATASET_SETTINGS['skip'] = gr.Checkbox(label="Skip Already Transcribed", value=False)

						with gr.Row():
							transcribe_button = gr.Button(value="Transcribe")
							prepare_validation_button = gr.Button(value="Prepare Validation")

						dataset_settings = list(DATASET_SETTINGS.values())
					with gr.Column():
						prepare_dataset_output = gr.TextArea(label="Console Output", interactive=False, max_lines=8)
			with gr.Tab("Generate Configuration"):
				with gr.Row():
					with gr.Column():
						TRAINING_SETTINGS["epochs"] = gr.Number(label="Epochs", value=500, precision=0)
						with gr.Row():
							TRAINING_SETTINGS["learning_rate"] = gr.Slider(label="Learning Rate", value=1e-5, minimum=0, maximum=1e-4, step=1e-6)
							TRAINING_SETTINGS["text_ce_lr_weight"] = gr.Slider(label="Text_CE LR Ratio", value=0.01, minimum=0, maximum=1)
							
						with gr.Row():
							lr_schemes = list(LEARNING_RATE_SCHEMES.keys())
							TRAINING_SETTINGS["learning_rate_scheme"] = gr.Radio(lr_schemes, label="Learning Rate Scheme", value=lr_schemes[0], type="value")
							TRAINING_SETTINGS["learning_rate_schedule"] = gr.Textbox(label="Learning Rate Schedule", placeholder=str(LEARNING_RATE_SCHEDULE), visible=True)
							TRAINING_SETTINGS["learning_rate_restarts"] = gr.Number(label="Learning Rate Restarts", value=4, precision=0, visible=False)

							TRAINING_SETTINGS["learning_rate_scheme"].change(
								fn=lambda x: ( gr.update(visible=x == lr_schemes[0]), gr.update(visible=x == lr_schemes[1]) ),
								inputs=TRAINING_SETTINGS["learning_rate_scheme"],
								outputs=[
									TRAINING_SETTINGS["learning_rate_schedule"],
									TRAINING_SETTINGS["learning_rate_restarts"],
								]
							)
						with gr.Row():
							TRAINING_SETTINGS["batch_size"] = gr.Number(label="Batch Size", value=128, precision=0)
							TRAINING_SETTINGS["gradient_accumulation_size"] = gr.Number(label="Gradient Accumulation Size", value=4, precision=0)
						with gr.Row():
							TRAINING_SETTINGS["save_rate"] = gr.Number(label="Save Frequency (in epochs)", value=5, precision=0)
							TRAINING_SETTINGS["validation_rate"] = gr.Number(label="Validation Frequency (in epochs)", value=5, precision=0)

						with gr.Row():
							TRAINING_SETTINGS["half_p"] = gr.Checkbox(label="Half Precision", value=args.training_default_halfp)
							TRAINING_SETTINGS["bitsandbytes"] = gr.Checkbox(label="BitsAndBytes", value=args.training_default_bnb)

						with gr.Row():
							TRAINING_SETTINGS["workers"] = gr.Number(label="Worker Processes", value=2, precision=0)
							TRAINING_SETTINGS["gpus"] = gr.Number(label="GPUs", value=get_device_count(), precision=0)

						TRAINING_SETTINGS["source_model"] = gr.Dropdown( choices=autoregressive_models, label="Source Model", type="value", value=autoregressive_models[0] )
						TRAINING_SETTINGS["resume_state"] = gr.Textbox(label="Resume State Path", placeholder="./training/${voice}/finetune/training_state/${last_state}.state")
						
						TRAINING_SETTINGS["voice"] = gr.Dropdown( choices=dataset_list, label="Dataset", type="value", value=dataset_list[0] if len(dataset_list) else ""  )

						with gr.Row():
							training_refresh_dataset = gr.Button(value="Refresh Dataset List")
							training_import_settings = gr.Button(value="Reuse/Import Dataset")
					with gr.Column():
						training_configuration_output = gr.TextArea(label="Console Output", interactive=False, max_lines=8)
						with gr.Row():
							training_optimize_configuration = gr.Button(value="Validate Training Configuration")
							training_save_configuration = gr.Button(value="Save Training Configuration")
			with gr.Tab("Run Training"):
				with gr.Row():
					with gr.Column():
						training_configs = gr.Dropdown(label="Training Configuration", choices=get_training_list())
						refresh_configs = gr.Button(value="Refresh Configurations")
						training_output = gr.TextArea(label="Console Output", interactive=False, max_lines=8)
						verbose_training = gr.Checkbox(label="Verbose Console Output", value=True)
						
						training_keep_x_past_datasets = gr.Slider(label="Keep X Previous States", minimum=0, maximum=8, value=0, step=1)
						with gr.Row():
							start_training_button = gr.Button(value="Train")
							stop_training_button = gr.Button(value="Stop")
							reconnect_training_button = gr.Button(value="Reconnect")
						
					with gr.Column():
						training_loss_graph = gr.LinePlot(label="Training Metrics",
							x="step",
							y="value",
							title="Training Metrics",
							color="type",
							tooltip=['step', 'value', 'type'],
							width=500,
							height=350,
						)
						training_lr_graph = gr.LinePlot(label="Training Metrics",
							x="step",
							y="value",
							title="Training Metrics",
							color="type",
							tooltip=['step', 'value', 'type'],
							width=500,
							height=350,
						)
						view_losses = gr.Button(value="View Losses")
		with gr.Tab("Settings"):
			with gr.Row():
				exec_inputs = []
				with gr.Column():
					EXEC_SETTINGS['listen'] = gr.Textbox(label="Listen", value=args.listen, placeholder="127.0.0.1:7860/")
					EXEC_SETTINGS['share'] = gr.Checkbox(label="Public Share Gradio", value=args.share)
					EXEC_SETTINGS['check_for_updates'] = gr.Checkbox(label="Check For Updates", value=args.check_for_updates)
					EXEC_SETTINGS['models_from_local_only'] = gr.Checkbox(label="Only Load Models Locally", value=args.models_from_local_only)
					EXEC_SETTINGS['low_vram'] = gr.Checkbox(label="Low VRAM", value=args.low_vram)
					EXEC_SETTINGS['embed_output_metadata'] = gr.Checkbox(label="Embed Output Metadata", value=args.embed_output_metadata)
					EXEC_SETTINGS['latents_lean_and_mean'] = gr.Checkbox(label="Slimmer Computed Latents", value=args.latents_lean_and_mean)
					EXEC_SETTINGS['voice_fixer'] = gr.Checkbox(label="Use Voice Fixer on Generated Output", value=args.voice_fixer)
					EXEC_SETTINGS['voice_fixer_use_cuda'] = gr.Checkbox(label="Use CUDA for Voice Fixer", value=args.voice_fixer_use_cuda)
					EXEC_SETTINGS['force_cpu_for_conditioning_latents'] = gr.Checkbox(label="Force CPU for Conditioning Latents", value=args.force_cpu_for_conditioning_latents)
					EXEC_SETTINGS['defer_tts_load'] = gr.Checkbox(label="Do Not Load TTS On Startup", value=args.defer_tts_load)
					EXEC_SETTINGS['prune_nonfinal_outputs'] = gr.Checkbox(label="Delete Non-Final Output", value=args.prune_nonfinal_outputs)
					EXEC_SETTINGS['device_override'] = gr.Textbox(label="Device Override", value=args.device_override)
				with gr.Column():
					EXEC_SETTINGS['sample_batch_size'] = gr.Number(label="Sample Batch Size", precision=0, value=args.sample_batch_size)
					EXEC_SETTINGS['concurrency_count'] = gr.Number(label="Gradio Concurrency Count", precision=0, value=args.concurrency_count)
					EXEC_SETTINGS['autocalculate_voice_chunk_duration_size'] = gr.Number(label="Auto-Calculate Voice Chunk Duration (in seconds)", precision=0, value=args.autocalculate_voice_chunk_duration_size)
					EXEC_SETTINGS['output_volume'] = gr.Slider(label="Output Volume", minimum=0, maximum=2, value=args.output_volume)
					
					EXEC_SETTINGS['autoregressive_model'] = gr.Dropdown(choices=autoregressive_models, label="Autoregressive Model", value=args.autoregressive_model if args.autoregressive_model else autoregressive_models[0])
					
					EXEC_SETTINGS['vocoder_model'] = gr.Dropdown(VOCODERS, label="Vocoder", value=args.vocoder_model if args.vocoder_model else VOCODERS[-1])
					EXEC_SETTINGS['whisper_backend'] = gr.Dropdown(WHISPER_BACKENDS, label="Whisper Backends", value=args.whisper_backend)
					EXEC_SETTINGS['whisper_model'] = gr.Dropdown(WHISPER_MODELS, label="Whisper Model", value=args.whisper_model)
					
					EXEC_SETTINGS['training_default_halfp'] = TRAINING_SETTINGS['half_p']
					EXEC_SETTINGS['training_default_bnb'] = TRAINING_SETTINGS['bitsandbytes']

					with gr.Row():
						autoregressive_models_update_button = gr.Button(value="Refresh Model List")
						gr.Button(value="Check for Updates").click(check_for_updates)
						gr.Button(value="(Re)Load TTS").click(
							reload_tts,
							inputs=EXEC_SETTINGS['autoregressive_model'],
							outputs=None
						)
						# kill_button = gr.Button(value="Close UI")

					def update_model_list_proxy( val ):
						autoregressive_models = get_autoregressive_models()
						if val not in autoregressive_models:
							val = autoregressive_models[0]
						return gr.update( choices=autoregressive_models, value=val )

					autoregressive_models_update_button.click(
						update_model_list_proxy,
						inputs=EXEC_SETTINGS['autoregressive_model'],
						outputs=EXEC_SETTINGS['autoregressive_model'],
					)

				exec_inputs = list(EXEC_SETTINGS.values())
				for k in EXEC_SETTINGS:
					EXEC_SETTINGS[k].change( fn=update_args_proxy, inputs=exec_inputs )
				
				EXEC_SETTINGS['autoregressive_model'].change(
					fn=update_autoregressive_model,
					inputs=EXEC_SETTINGS['autoregressive_model'],
					outputs=None
				)

				EXEC_SETTINGS['vocoder_model'].change(
					fn=update_vocoder_model,
					inputs=EXEC_SETTINGS['vocoder_model'],
					outputs=None
				)

		history_voices.change(
			fn=history_view_results,
			inputs=history_voices,
			outputs=[
				history_info,
				history_results_list,
			]
		)
		history_results_list.change(
			fn=lambda voice, file: f"./results/{voice}/{file}",
			inputs=[
				history_voices,
				history_results_list,
			],
			outputs=history_audio
		)
		audio_in.upload(
			fn=read_generate_settings_proxy,
			inputs=audio_in,
			outputs=[
				metadata_out,
				latents_out,
				import_voice_name,
				utilities_metadata_column,
			]
		)

		import_voice_button.click(
			fn=import_voices_proxy,
			inputs=[
				audio_in,
				import_voice_name,
			],
			outputs=import_voice_name #console_output
		)
		show_experimental_settings.change(
			fn=lambda x: gr.update(visible=x),
			inputs=show_experimental_settings,
			outputs=experimental_column
		)
		preset.change(fn=update_presets,
			inputs=preset,
			outputs=[
				GENERATE_SETTINGS['num_autoregressive_samples'],
				GENERATE_SETTINGS['diffusion_iterations'],
			],
		)

		recompute_voice_latents.click(compute_latents_proxy,
			inputs=[
				GENERATE_SETTINGS['voice'],
				GENERATE_SETTINGS['voice_latents_chunks'],
			],
			outputs=GENERATE_SETTINGS['voice'],
		)
		
		GENERATE_SETTINGS['emotion'].change(
			fn=lambda value: gr.update(visible=value == "Custom"),
			inputs=GENERATE_SETTINGS['emotion'],
			outputs=GENERATE_SETTINGS['prompt']
		)
		GENERATE_SETTINGS['mic_audio'].change(fn=lambda value: gr.update(value="microphone"),
			inputs=GENERATE_SETTINGS['mic_audio'],
			outputs=GENERATE_SETTINGS['voice']
		)

		refresh_voices.click(update_voices,
			inputs=None,
			outputs=[
				GENERATE_SETTINGS['voice'],
				dataset_settings[0],
				history_voices
			]
		)

		generate_settings = list(GENERATE_SETTINGS.values())
		submit.click(
			lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
			outputs=[source_sample, candidates_list, generation_results],
		)

		submit_event = submit.click(generate_proxy,
			inputs=generate_settings,
			outputs=[output_audio, source_sample, candidates_list, generation_results],
			api_name="generate",
		)


		copy_button.click(import_generate_settings_proxy,
			inputs=audio_in, # JSON elements cannot be used as inputs
			outputs=generate_settings
		)

		reset_generation_settings_button.click(
			fn=reset_generation_settings,
			inputs=None,
			outputs=generate_settings
		)

		history_copy_settings_button.click(history_copy_settings,
			inputs=[
				history_voices,
				history_results_list,
			],
			outputs=generate_settings
		)

		refresh_configs.click(
			lambda: gr.update(choices=get_training_list()),
			inputs=None,
			outputs=training_configs
		)
		start_training_button.click(run_training,
			inputs=[
				training_configs,
				verbose_training,
				training_keep_x_past_datasets,
			],
			outputs=[
				training_output,
			],
		)
		training_output.change(
			fn=update_training_dataplot,
			inputs=None,
			outputs=[
				training_loss_graph,
				training_lr_graph,
			],
			show_progress=False,
		)

		view_losses.click(
			fn=update_training_dataplot,
			inputs=[
				training_configs
			],
			outputs=[
				training_loss_graph,
				training_lr_graph,
			],
		)

		stop_training_button.click(stop_training,
			inputs=None,
			outputs=training_output #console_output
		)
		reconnect_training_button.click(reconnect_training,
			inputs=[
				verbose_training,
			],
			outputs=training_output #console_output
		)
		transcribe_button.click(
			prepare_dataset_proxy,
			inputs=dataset_settings,
			outputs=prepare_dataset_output #console_output
		)
		prepare_validation_button.click(
			prepare_validation_dataset,
			inputs=[
				dataset_settings[0],
				DATASET_SETTINGS['validation_size'],
			],
			outputs=prepare_dataset_output #console_output
		)
		
		training_refresh_dataset.click(
			lambda: gr.update(choices=get_dataset_list()),
			inputs=None,
			outputs=TRAINING_SETTINGS["voice"],
		)
		training_settings = list(TRAINING_SETTINGS.values())
		training_optimize_configuration.click(optimize_training_settings_proxy,
			inputs=training_settings,
			outputs=training_settings[:-1] + [training_configuration_output] #console_output
		)
		training_import_settings.click(import_training_settings_proxy,
			inputs=TRAINING_SETTINGS['voice'],
			outputs=training_settings[:-1] + [training_configuration_output] #console_output
		)
		training_save_configuration.click(save_training_settings_proxy,
			inputs=training_settings,
			outputs=training_configuration_output #console_output
		)

		if os.path.isfile('./config/generate.json'):
			ui.load(import_generate_settings_proxy, inputs=None, outputs=generate_settings)
		
		if args.check_for_updates:
			ui.load(check_for_updates)

		stop.click(fn=cancel_generate, inputs=None, outputs=None)


	ui.queue(concurrency_count=args.concurrency_count)
	webui = ui
	return webui