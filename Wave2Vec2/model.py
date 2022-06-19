import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from tqdm import tqdm
from functools import wraps
from time import time
import torch.nn as nn

class Wave2Vec2:
    
    def timing(f):
        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()
            print(f"[INFO] func:{f.__name__} args:[{args}, {kw}] took: {te-ts:4f} sec")
            return result
        return wrap

    @timing
    def __init__(self, 
        pretrained_model_name_or_path="facebook/wav2vec2-base-960h", 
        multi_gpu_enabled=True):

        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name_or_path)
        # self.device = torch.device('cuda')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if multi_gpu_enabled and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model,[0,1]).to(self.device)
        else:
            self.model.to(self.device)
        print("[INFO] model initialized..")
    
    @timing
    def load_wav_file(self, filename):
        speech, rate = librosa.load(filename,sr=16000)
        encoded_audio = self.tokenizer(speech, return_tensors = 'pt').input_values
        encoded_audio = encoded_audio.to(self.device)
        print("[INFO] wav file vectorized..")
        return encoded_audio
    
    @timing
    def predict(self, encoded_audio, BATCH_SIZE=64,ignoreLast=False):
        try:
            SPILT_SIZE = encoded_audio.shape[1] // (BATCH_SIZE-1)
            batches = torch.split(encoded_audio,SPILT_SIZE, dim=1)
            print(f"[INFO] split data into batches with SPILT_SIZE:{SPILT_SIZE}")
            print("[INFO] prediction started")
            if ignoreLast:
                transcriptions = [self.__predict(batch, self.model, self.tokenizer) \
                    for batch in tqdm(batches[:-1])]
            else:
                transcriptions = [self.__predict(batch, self.model, self.tokenizer) \
                    for batch in tqdm(batches[:])]
            print("[INFO] prediction done")
            return transcriptions  
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory...Try to increase batch size param or restart kernel")
            elif "Kernel size can't greater than actual input size" in str(e):
                print(f"Kernel size issue...Try to set ignoreLast to True")
            
            print(e)
            raise Exception('[ERROR] prediction failed!')

    @staticmethod
    def __predict(batch, model, tokenizer):
        with torch.no_grad():
            logits = model(batch).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.decode(predicted_ids[0])
        return transcription