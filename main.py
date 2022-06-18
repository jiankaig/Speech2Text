import os
import sys
from Wave2Vec2 import downloader, model

def main():
    audio_filename = 'out.wav'
    output_text_filename = 'output.txt'
    yt_url = 'https://www.youtube.com/watch?v=8rJu-eltak0'

    # download yt video if needed
    if yt_url != "" and not os.path.exists(audio_filename):
        dlr = downloader.Downloader()
        dlr.download(yt_url, audio_filename)
    
    # load SOTA model
    cls = model.Wave2Vec2("facebook/wav2vec2-large-robust-ft-libri-960h")

    # vectorise raw audio into tensor array
    encoded_audio = cls.load_wav_file(audio_filename)

    # transcribe audio to text
    transcriptions = cls.predict(encoded_audio, BATCH_SIZE=32,ignoreLast=True)

    # stich back transcriptions and save text
    full_text = f"\n".join(f"{i}: {t}" for i,t in enumerate(transcriptions)) 
    with open(output_text_filename,'w') as f:
        f.write(full_text)

if __name__ == "__main__":
    main()