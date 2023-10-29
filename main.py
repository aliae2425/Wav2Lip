from __future__ import division
import os, argparse
import subprocess
from elevenlabs import generate, play, set_api_key, Voice, VoiceSettings
from IPython.display import HTML, clear_output
import wave

import cv2
from inference import Inference

main_directory = os.getcwd()
sample_directory = os.path.join(main_directory, r"sample_data")
print("init in Current directory:", main_directory)

parser = argparse.ArgumentParser(
        prog='PDA Pipeline',
        description=r"exposition pipeline pour l'etape 3"
)

parser.add_argument('--setup', action='store_true', help='setup the environment')

args = parser.parse_args()


def functon_setup():
    import requests, shutil

    if not os.path.exists(sample_directory):
        os.makedirs(sample_directory)
        print("Sample directory created")
    else:
        print("Sample directory already exists")


    checkpoints_directory = os.path.join(main_directory, r"checkpoints")

    requirements = [
        "wav2lip.pth",
        "wav2lip_gan.pth",
        "resnet50.pth",
        "mobilenet.pth"
    ]

    git_url = 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/'

    for requirement in requirements:
        if os.path.exists(os.path.join(checkpoints_directory, requirement)):
            print(f"{requirement} already exists")
            continue
        else:
            url = git_url + requirement
            with requests.get(url, stream=True) as r:
                with open(os.path.join(checkpoints_directory, requirement), 'wb') as f: shutil.copyfileobj(r.raw, f)
            print(f"{requirement} downloaded")
    
    
    print("All requirements downloaded")

def generate_text (image):
    
    return 'Bonjour, je suis Denis Brogniar, je suis un robot et Je suis heureux de vous rencontrer.'

def generate_voice (speach):

    set_api_key("22686e1a4c870e3a084dd4c5e4e8b3ff")
    
    audio = generate(
        text=speach,
        voice= Voice(
            voice_id="rbo5SubtGEMDNNQLr2ZX",
            settings=VoiceSettings(
                stability=0.31,
                similarity_boost=0.41,
                style=0.55,
                use_speaker_boost=True,
            )
        ),
        model="eleven_multilingual_v2"
    )
    
    wav_filename = os.path.join(main_directory, r"sample_data/input_audio.wav")
    wav_file = wave.open(wav_filename, 'wb')

    num_channels = 1
    sample_width = 2
    frame_rate = 44100
    num_frames = len(audio)
    compression_type = "NONE"
    compression_name = "not compressed"

    wav_file.setparams((num_channels, sample_width, frame_rate, num_frames, compression_type, compression_name))
    wav_file.writeframes(audio)
    wav_file.close()
    print("Audio saved")
    
    
    return audio

def generate_video ():
       
    output_file_path = os.path.join(main_directory, r"results/result_voice.mp4")
    
    if os.path.exists(output_file_path):
        clear_output()
        print("Final Video Preview")
        print("Download this video from", output_file_path)
    else:
        print("Processing failed. Output video not available.")
        
    os.chdir('..')
    
    return 1


def main():
    audio = generate_voice("Bonjour, je suis Denis Brogniar, je suis un robot et Je suis heureux de vous rencontrer.")
    Inference.start()


if __name__ == "__main__":
    if args.setup:
        functon_setup()
    else:
        main()
