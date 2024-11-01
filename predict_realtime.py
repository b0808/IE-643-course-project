
import torch
import config
import model
import extract_features
from gtts import gTTS
from googletrans import Translator
map_location=torch.device('cpu') 
dif = {"Hindi":'hi', "Marathi":'mr', "kannada":'kn'}
def text_to_speech(text, output_file,model_lang):
    tts = gTTS(text=text, lang=dif[model_lang])
    tts.save(output_file)
    print(f"Audio saved as {output_file}")

def translate_to_hindi(text,model_lang):
    translator = Translator()
    translated_text = translator.translate(text, src='en', dest=dif[model_lang]).text
    return translated_text

class VideoDescriptionRealTime(object):

    def __init__(self, config,model_option):
        self.latent_dim = config.latent_dim
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.max_probability = config.max_probability
        self.model_option = model_option

        # models
        self.tokenizer_gru, self.encoder_gru, self.decoder_gru,self.encoder_lstm,self.decoder_lstm = model.inference_model()
        self.save_model_path = config.save_model_path
        self.test_path = config.test_path
        self.search_type = config.search_type
        self.num = 0

    def greedy_search(self, input_seq):   
        start_token = torch.zeros(1, 1, config.num_decoder_tokens)
        start_token[0, 0, self.tokenizer_gru.word_index['bos']] = 1.0
        
        decoded_sequence = []
       
        if(self.model_option=="LSTM + GRU"):
            with torch.no_grad():
                states_gru = self.encoder_gru(input_seq)
                states_lstm = self.encoder_lstm(input_seq)
            for _ in range(15):  
                with torch.no_grad():
                    output_lstm, state_h_lstm = self.decoder_lstm(start_token, states_lstm)
                    output_gru, state_h_gru = self.decoder_gru(start_token, states_gru)
                states_lstm = state_h_lstm
                states_gru = state_h_gru
                predicted_token = torch.argmax((output_lstm[0, -1, :]+output_gru[0, -1, :])/2).item()
                if predicted_token == 0:
                    continue
                if predicted_token == self.tokenizer_gru.word_index['eos']:
                    break
                decoded_sequence.append(predicted_token)
                start_token = torch.zeros(1, 1, config.num_decoder_tokens)
                start_token[0, 0, predicted_token] = 1.0
                
        if(self.model_option=="LSTM"):
            with torch.no_grad():
                states_lstm = self.encoder_lstm(input_seq)
            for _ in range(15):  
                with torch.no_grad():
                    output_lstm, state_h_lstm = self.decoder_lstm(start_token, states_lstm)
                states_lstm = state_h_lstm
                predicted_token = torch.argmax((output_lstm[0, -1, :])).item()
                if predicted_token == 0:
                    continue
                if predicted_token == self.tokenizer_gru.word_index['eos']:
                    break
                decoded_sequence.append(predicted_token)
                start_token = torch.zeros(1, 1, config.num_decoder_tokens)
                start_token[0, 0, predicted_token] = 1.0
                
        if(self.model_option=="GRU"):
            with torch.no_grad():
                states_gru = self.encoder_gru(input_seq)
            for _ in range(15): 
                with torch.no_grad():
                    output_gru, state_h_gru = self.decoder_gru(start_token, states_gru)
                states_gru = state_h_gru
                predicted_token = torch.argmax((output_gru[0, -1, :])).item()
                
                if predicted_token == 0:
                    continue
                if predicted_token == self.tokenizer_gru.word_index['eos']:
                    break
                decoded_sequence.append(predicted_token)
                start_token = torch.zeros(1, 1, config.num_decoder_tokens)
                start_token[0, 0, predicted_token] = 1.0
                
        inv_map = {v: k for k, v in self.tokenizer_gru.word_index.items()}
        decoded_sentence = ' '.join([inv_map[token] for token in decoded_sequence if token in inv_map])
        return decoded_sentence

    def index_to_word(self):
        # inverts word tokenizer
        index_to_word = {value: key for key, value in self.tokenizer.word_index.items()}
        return index_to_word

    def main_test(self,video_path):
        model = extract_features.model_cnn_load()
        X_test = extract_features.extract_features(video_path, model)
        if self.search_type == 'greedy':
            sentence_predicted = self.greedy_search(torch.tensor(X_test.reshape((-1, 80, 4096))))
        return sentence_predicted
        
