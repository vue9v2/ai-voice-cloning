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
	if not text:
		raise gr.Error("Please provide text.")
	if not voice:
		raise gr.Error("Please provide a voice.")

	try:
		sample, outputs, stats = generate(
			text=text,
			delimiter=delimiter,
			emotion=emotion,
			prompt=prompt,
			voice=voice,
			mic_audio=mic_audio,
			voice_latents_chunks=voice_latents_chunks,
			seed=seed,
			candidates=candidates,
			num_autoregressive_samples=num_autoregressive_samples,
			diffusion_iterations=diffusion_iterations,
			temperature=temperature,
			diffusion_sampler=diffusion_sampler,
			breathing_room=breathing_room,
			cvvp_weight=cvvp_weight,
			top_p=top_p,
			diffusion_temperature=diffusion_temperature,
			length_penalty=length_penalty,
			repetition_penalty=repetition_penalty,
			cond_free_k=cond_free_k,
			experimental_checkboxes=experimental_checkboxes,
			progress=progress
		)
	except Exception as e:
		message = str(e)
		if message == "Kill signal detected":
			unload_tts()

		raise gr.Error(message)
	

	return (
		outputs[0],
		gr.update(value=sample, visible=sample is not None),
		gr.update(choices=outputs, value=outputs[0], visible=len(outputs) > 1, interactive=True),
		gr.update(value=stats, visible=True),
	)

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

def get_training_configs():
	configs = []
	for i, file in enumerate(sorted(os.listdir(f"./training/"))):
		if file[-5:] != ".yaml" or file[0] == ".":
			continue
		configs.append(f"./training/{file}")

	return configs

def update_training_configs():
	return gr.update(choices=get_training_list())

history_headers = {
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
		for k in history_headers:
			v = file
			if k != "Name":
				v = metadata[history_headers[k]] if history_headers[k] in metadata else '?'
			values.append(v)


		files.append(file)
		results.append(values)

	return (
		results,
		gr.Dropdown.update(choices=sorted(files))
	)

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
		gr.update(visible=j is not None),
		gr.update(value=latents, visible=latents is not None),
		None if j is None else j['voice']
	)

def prepare_dataset_proxy( voice, language, progress=gr.Progress(track_tqdm=True) ):
	return prepare_dataset( get_voices(load_latents=False)[voice], outdir=f"./training/{voice}/", language=language, progress=progress )

def optimize_training_settings_proxy( *args, **kwargs ):
	tup = optimize_training_settings(*args, **kwargs)

	return (
		gr.update(value=tup[0]),
		gr.update(value=tup[1]),
		gr.update(value=tup[2]),
		gr.update(value=tup[3]),
		gr.update(value=tup[4]),
		gr.update(value=tup[5]),
		gr.update(value=tup[6]),
		gr.update(value=tup[7]),
		"\n".join(tup[8])
	)

def import_training_settings_proxy( voice ):
	indir = f'./training/{voice}/'
	outdir = f'./training/{voice}-finetune/'

	in_config_path = f"{indir}/train.yaml"
	out_config_path = None
	out_configs = []
	if os.path.isdir(outdir):
		out_configs = sorted([d[:-5] for d in os.listdir(outdir) if d[-5:] == ".yaml" ])
	if len(out_configs) > 0:
		out_config_path = f'{outdir}/{out_configs[-1]}.yaml'

	config_path = out_config_path if out_config_path else in_config_path

	messages = []
	with open(config_path, 'r') as file:
		config = yaml.safe_load(file)
		messages.append(f"Importing from: {config_path}")

	dataset_path = f"./training/{voice}/train.txt"
	with open(dataset_path, 'r', encoding="utf-8") as f:
		lines = len(f.readlines())
		messages.append(f"Basing epoch size to {lines} lines")

	batch_size = config['datasets']['train']['batch_size']
	mega_batch_factor = config['train']['mega_batch_factor']

	
	iterations = config['train']['niter']
	steps_per_iteration = int(lines / batch_size)
	epochs = int(iterations / steps_per_iteration)


	learning_rate = config['steps']['gpt_train']['optimizer_params']['lr']
	text_ce_lr_weight = config['steps']['gpt_train']['losses']['text_ce']['weight']
	learning_rate_schedule = [ int(x / steps_per_iteration) for x in config['train']['gen_lr_steps'] ]


	print_rate = int(config['logger']['print_freq'] / steps_per_iteration)
	save_rate = int(config['logger']['save_checkpoint_freq'] / steps_per_iteration)

	statedir = f'{outdir}/training_state/' # NOOO STOP MIXING YOUR CASES
	resumes = []
	resume_path = None
	source_model = None
	
	if "pretrain_model_gpt" in config['path']:
		source_model = config['path']['pretrain_model_gpt']
	elif "resume_state" in config['path']:
		resume_path = config['path']['resume_state']


	if os.path.isdir(statedir):
		resumes = sorted([int(d[:-6]) for d in os.listdir(statedir) if d[-6:] == ".state" ])

	if len(resumes) > 0:
		resume_path = f'{statedir}/{resumes[-1]}.state'
		messages.append(f"Latest resume found: {resume_path}")


	half_p = config['fp16']
	bnb = True

	if "ext" in config and "bitsandbytes" in config["ext"]:
		bnb = config["ext"]["bitsandbytes"]

	messages = "\n".join(messages)

	return (
		epochs,
		learning_rate,
		text_ce_lr_weight,
		learning_rate_schedule,
		batch_size,
		mega_batch_factor,
		print_rate,
		save_rate,
		resume_path,
		half_p,
		bnb,
		source_model,
		messages
	)


def save_training_settings_proxy( epochs, learning_rate, text_ce_lr_weight, learning_rate_schedule, batch_size, mega_batch_factor, print_rate, save_rate, resume_path, half_p, bnb, source_model, voice ):
	name = f"{voice}-finetune"
	dataset_name = f"{voice}-train"
	dataset_path = f"./training/{voice}/train.txt"
	validation_name = f"{voice}-val"
	validation_path = f"./training/{voice}/train.txt"

	with open(dataset_path, 'r', encoding="utf-8") as f:
		lines = len(f.readlines())

	messages = []

	iterations = calc_iterations(epochs=epochs, lines=lines, batch_size=batch_size)
	messages.append(f"For {epochs} epochs with {lines} lines, iterating for {iterations} steps")

	print_rate = int(print_rate * iterations / epochs)
	save_rate = int(save_rate * iterations / epochs)

	if not learning_rate_schedule:
		learning_rate_schedule = EPOCH_SCHEDULE
	learning_rate_schedule = schedule_learning_rate( iterations / epochs )

	messages.append(save_training_settings(
		iterations=iterations,
		batch_size=batch_size,
		learning_rate=learning_rate,
		text_ce_lr_weight=text_ce_lr_weight,
		learning_rate_schedule=learning_rate_schedule,
		mega_batch_factor=mega_batch_factor,
		print_rate=print_rate,
		save_rate=save_rate,
		name=name,
		dataset_name=dataset_name,
		dataset_path=dataset_path,
		validation_name=validation_name,
		validation_path=validation_path,
		output_name=f"{voice}/train.yaml",
		resume_path=resume_path,
		half_p=half_p,
		bnb=bnb,
		source_model=source_model,
	))
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

	with gr.Blocks() as ui:
		with gr.Tab("Generate"):
			with gr.Row():
				with gr.Column():
					text = gr.Textbox(lines=4, label="Prompt")
			with gr.Row():
				with gr.Column():
					delimiter = gr.Textbox(lines=1, label="Line Delimiter", placeholder="\\n")

					emotion = gr.Radio( ["Happy", "Sad", "Angry", "Disgusted", "Arrogant", "Custom"], value="Custom", label="Emotion", type="value", interactive=True )
					prompt = gr.Textbox(lines=1, label="Custom Emotion + Prompt (if selected)")
					voice = gr.Dropdown(choices=voice_list_with_defaults, label="Voice", type="value", value=voice_list_with_defaults[0]) # it'd be very cash money if gradio was able to default to the first value in the list without this shit
					mic_audio = gr.Audio( label="Microphone Source", source="microphone", type="filepath" )
					voice_latents_chunks = gr.Slider(label="Voice Chunks", minimum=1, maximum=128, value=1, step=1)
					with gr.Row():
						refresh_voices = gr.Button(value="Refresh Voice List")
						recompute_voice_latents = gr.Button(value="(Re)Compute Voice Latents")

					def update_baseline_for_latents_chunks( voice ):
						path = f'{get_voice_dir()}/{voice}/'
						if not os.path.isdir(path):
							return 1

						files = os.listdir(path)
						count = 0
						for file in files:
							if file[-4:] == ".wav":
								count += 1

						return count if count > 0 else 1

					voice.change(
						fn=update_baseline_for_latents_chunks,
						inputs=voice,
						outputs=voice_latents_chunks
					)
				with gr.Column():
					candidates = gr.Slider(value=1, minimum=1, maximum=6, step=1, label="Candidates")
					seed = gr.Number(value=0, precision=0, label="Seed")

					preset = gr.Radio( ["Ultra Fast", "Fast", "Standard", "High Quality"], label="Preset", type="value" )
					num_autoregressive_samples = gr.Slider(value=128, minimum=2, maximum=512, step=1, label="Samples")
					diffusion_iterations = gr.Slider(value=128, minimum=0, maximum=512, step=1, label="Iterations")

					temperature = gr.Slider(value=0.2, minimum=0, maximum=1, step=0.1, label="Temperature")
					breathing_room = gr.Slider(value=8, minimum=1, maximum=32, step=1, label="Pause Size")
					diffusion_sampler = gr.Radio(
						["P", "DDIM"], # + ["K_Euler_A", "DPM++2M"],
						value="P", label="Diffusion Samplers", type="value" )
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
					history_info = gr.Dataframe(label="Results", headers=list(history_headers.keys()))
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
				with gr.Column():
					metadata_out = gr.JSON(label="Audio Metadata", visible=False)
					copy_button = gr.Button(value="Copy Settings", visible=False)
					latents_out = gr.File(type="binary", label="Voice Latents", visible=False)
		with gr.Tab("Training"):
			with gr.Tab("Prepare Dataset"):
				with gr.Row():
					with gr.Column():
						dataset_settings = [
							gr.Dropdown( choices=voice_list, label="Dataset Source", type="value", value=voice_list[0] if len(voice_list) > 0 else "" ),
							gr.Textbox(label="Language", placeholder="English")
						]
						prepare_dataset_button = gr.Button(value="Prepare")
					with gr.Column():
						prepare_dataset_output = gr.TextArea(label="Console Output", interactive=False, max_lines=8)
			with gr.Tab("Generate Configuration"):
				with gr.Row():
					with gr.Column():
						training_settings = [
							gr.Number(label="Epochs", value=500, precision=0),
						]
						with gr.Row():
							with gr.Column():
								training_settings = training_settings + [
									gr.Slider(label="Learning Rate", value=1e-5, minimum=0, maximum=1e-4, step=1e-6),
									gr.Slider(label="Text_CE LR Ratio", value=0.01, minimum=0, maximum=1),
								]
							training_settings = training_settings + [
								gr.Textbox(label="Learning Rate Schedule", placeholder=str(EPOCH_SCHEDULE)),
							]
						with gr.Row():
							training_settings = training_settings + [
								gr.Number(label="Batch Size", value=128, precision=0),
								gr.Number(label="Mega Batch Factor", value=4, precision=0),
							]
						with gr.Row():
							training_settings = training_settings + [
								gr.Number(label="Print Frequency (in epochs)", value=5, precision=0),
								gr.Number(label="Save Frequency (in epochs)", value=5, precision=0),
							]
						training_settings = training_settings + [
							gr.Textbox(label="Resume State Path", placeholder="./training/${voice}-finetune/training_state/${last_state}.state"),
						]
						training_halfp = gr.Checkbox(label="Half Precision", value=args.training_default_halfp)
						training_bnb = gr.Checkbox(label="BitsAndBytes", value=args.training_default_bnb)
						source_model = gr.Dropdown( choices=autoregressive_models, label="Source Model", type="value", value=autoregressive_models[0] )
						dataset_list_dropdown = gr.Dropdown( choices=dataset_list, label="Dataset", type="value", value=dataset_list[0] if len(dataset_list) else ""  )
						training_settings = training_settings + [ training_halfp, training_bnb, source_model, dataset_list_dropdown ]

						with gr.Row():
							refresh_dataset_list = gr.Button(value="Refresh Dataset List")
							import_dataset_button = gr.Button(value="Reuse/Import Dataset")
					with gr.Column():
						save_yaml_output = gr.TextArea(label="Console Output", interactive=False, max_lines=8)
						with gr.Row():
							optimize_yaml_button = gr.Button(value="Validate Training Configuration")
							save_yaml_button = gr.Button(value="Save Training Configuration")
			with gr.Tab("Run Training"):
				with gr.Row():
					with gr.Column():
						training_configs = gr.Dropdown(label="Training Configuration", choices=get_training_list())
						with gr.Row():
							refresh_configs = gr.Button(value="Refresh Configurations")
						
						training_loss_graph = gr.LinePlot(label="Training Metrics",
							x="step",
							y="value",
							title="Training Metrics",
							color="type",
							tooltip=['step', 'value', 'type'],
							width=600,
							height=350,
						)
						view_losses = gr.Button(value="View Losses")

					with gr.Column():
						training_output = gr.TextArea(label="Console Output", interactive=False, max_lines=8)
						verbose_training = gr.Checkbox(label="Verbose Console Output", value=True)
						training_buffer_size = gr.Slider(label="Console Buffer Size", minimum=4, maximum=32, value=8)
						training_keep_x_past_datasets = gr.Slider(label="Keep X Previous States", minimum=0, maximum=8, value=0, step=1)
						training_gpu_count = gr.Number(label="GPUs", value=1)
						with gr.Row():
							start_training_button = gr.Button(value="Train")
							stop_training_button = gr.Button(value="Stop")
							reconnect_training_button = gr.Button(value="Reconnect")
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
						gr.Checkbox(label="Use Voice Fixer on Generated Output", value=args.voice_fixer),
						gr.Checkbox(label="Use CUDA for Voice Fixer", value=args.voice_fixer_use_cuda),
						gr.Checkbox(label="Force CPU for Conditioning Latents", value=args.force_cpu_for_conditioning_latents),
						gr.Checkbox(label="Do Not Load TTS On Startup", value=args.defer_tts_load),
						gr.Checkbox(label="Delete Non-Final Output", value=args.prune_nonfinal_outputs),
						gr.Textbox(label="Device Override", value=args.device_override),
					]
				with gr.Column():
					exec_inputs = exec_inputs + [
						gr.Number(label="Sample Batch Size", precision=0, value=args.sample_batch_size),
						gr.Number(label="Gradio Concurrency Count", precision=0, value=args.concurrency_count),
						gr.Number(label="Output Sample Rate", precision=0, value=args.output_sample_rate),
						gr.Slider(label="Output Volume", minimum=0, maximum=2, value=args.output_volume),
					]
					
					autoregressive_model_dropdown = gr.Dropdown(choices=autoregressive_models, label="Autoregressive Model", value=args.autoregressive_model if args.autoregressive_model else autoregressive_models[0])
					
					whisper_model_dropdown = gr.Dropdown(["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large"], label="Whisper Model", value=args.whisper_model)
					use_whisper_cpp = gr.Checkbox(label="Use Whisper.cpp", value=args.whisper_cpp)
					
					exec_inputs = exec_inputs + [ autoregressive_model_dropdown, whisper_model_dropdown, use_whisper_cpp, training_halfp, training_bnb ]

					with gr.Row():
						autoregressive_models_update_button = gr.Button(value="Refresh Model List")
						gr.Button(value="Check for Updates").click(check_for_updates)
						gr.Button(value="(Re)Load TTS").click(
							reload_tts,
							inputs=autoregressive_model_dropdown,
							outputs=None
						)

					def update_model_list_proxy( val ):
						autoregressive_models = get_autoregressive_models()
						if val not in autoregressive_models:
							val = autoregressive_models[0]
						return gr.update( choices=autoregressive_models, value=val )

					autoregressive_models_update_button.click(
						update_model_list_proxy,
						inputs=autoregressive_model_dropdown,
						outputs=autoregressive_model_dropdown,
					)

				for i in exec_inputs:
					i.change( fn=update_args, inputs=exec_inputs )
				
				autoregressive_model_dropdown.change(
					fn=update_autoregressive_model,
					inputs=autoregressive_model_dropdown,
					outputs=None
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
				copy_button,
				latents_out,
				import_voice_name
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
				num_autoregressive_samples,
				diffusion_iterations,
			],
		)

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

		refresh_voices.click(update_voices,
			inputs=None,
			outputs=[
				voice,
				dataset_settings[0],
				history_voices
			]
		)

		submit.click(
			lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
			outputs=[source_sample, candidates_list, generation_results],
		)

		submit_event = submit.click(run_generation,
			inputs=input_settings,
			outputs=[output_audio, source_sample, candidates_list, generation_results],
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

		refresh_configs.click(
			lambda: gr.update(choices=get_training_list()),
			inputs=None,
			outputs=training_configs
		)
		start_training_button.click(run_training,
			inputs=[
				training_configs,
				verbose_training,
				training_gpu_count,
				training_buffer_size,
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
			],
		)

		stop_training_button.click(stop_training,
			inputs=None,
			outputs=training_output #console_output
		)
		reconnect_training_button.click(reconnect_training,
			inputs=[
				verbose_training,
				training_buffer_size,
			],
			outputs=training_output #console_output
		)
		prepare_dataset_button.click(
			prepare_dataset_proxy,
			inputs=dataset_settings,
			outputs=prepare_dataset_output #console_output
		)
		refresh_dataset_list.click(
			lambda: gr.update(choices=get_dataset_list()),
			inputs=None,
			outputs=dataset_list_dropdown,
		)
		optimize_yaml_button.click(optimize_training_settings_proxy,
			inputs=training_settings,
			outputs=training_settings[1:9] + [save_yaml_output] #console_output
		)
		import_dataset_button.click(import_training_settings_proxy,
			inputs=dataset_list_dropdown,
			outputs=training_settings[:11] + [save_yaml_output] #console_output
		)
		save_yaml_button.click(save_training_settings_proxy,
			inputs=training_settings,
			outputs=save_yaml_output #console_output
		)

		if os.path.isfile('./config/generate.json'):
			ui.load(import_generate_settings, inputs=None, outputs=input_settings)
		
		if args.check_for_updates:
			ui.load(check_for_updates)

		stop.click(fn=cancel_generate, inputs=None, outputs=None)


	ui.queue(concurrency_count=args.concurrency_count)
	webui = ui
	return webui