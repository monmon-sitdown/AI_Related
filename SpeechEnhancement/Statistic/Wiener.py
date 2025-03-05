import librosa
from librosa.core.spectrum import amplitude_to_db
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin

def add_nosie(clean,noise,snr):
    # same length of clean and noise wav
    if not len(clean) == len(noise):
        print("")
        return False
 
    # clean energy
    p_clean = np.sum(np.abs(clean)**2)
    
    # noise energy
    p_noise = np.sum(np.abs(noise)**2)

    # calculate scale rate
    scale =  np.sqrt( (p_clean/p_noise) * np.power(10,-snr/10) )

    # Obtain Additive Noise
    noisy = clean + scale * noise
    
    return noisy

def gen_color_noise(N,order_filter,fs,f_L,f_H):
    
    noise = np.random.randn(N)
    m_firwin = firwin(order_filter, [2*f_L/fs, 2*f_H/fs], pass_zero="bandpass")
    color_noise = lfilter(m_firwin, 1.0, noise)
    return color_noise


def train_wiener_filter(cleans,noises,para):
    n_fft = para["n_fft"]
    hop_length = para["hop_length"]
    win_length = para["win_length"]
    alpha = para["alpha"]
    beta = para["beta"]
    Pxxs = []
    Pnns =[]
    for clean,noise in zip(cleans,noises):
        S_clean = librosa.stft(clean,n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        S_noise = librosa.stft(noise,n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        Pxx = np.mean((np.abs(S_clean))**2,axis=1,keepdims=True) # Dx1
        Pnn = np.mean((np.abs(S_noise))**2,axis=1,keepdims=True)
        Pxxs.append(Pxx)
        Pnns.append(Pnn)
    
    train_Pxx = np.mean(np.concatenate(Pxxs,axis=1),axis=1,keepdims=True)
    train_Pnn = np.mean(np.concatenate(Pnns,axis=1),axis=1,keepdims=True)
    
    H = (train_Pxx/(train_Pxx+alpha*train_Pnn))**beta
    
    return H    
if __name__ == "__main__":
    
    files = ["sf2_cln.wav","sf3_cln.wav","sm1_cln.wav","sm2_cln.wav","sm3_cln.wav"]
    cleans =[]
    noises= []
    
    for file in files:
        print(file)
      
        clean,fs = librosa.load(file,sr=None)

        noise = gen_color_noise(len(clean),128,fs,2400,3200)
   
        noisy = add_nosie(clean,noise,5)
        cleans.append(clean)
        noises.append(noisy-clean)
   

    para_wiener = {}
    para_wiener["n_fft"] = 256
    para_wiener["hop_length"] = 128
    para_wiener["win_length"] = 256
    para_wiener["alpha"] = 1
    para_wiener["beta"] =3
    

    H= train_wiener_filter(cleans,noises,para_wiener)
    
    

    clean_wav_file = "sf1_cln.wav"
    test_clean,fs = librosa.load(clean_wav_file,sr=None) 
    test_noise = gen_color_noise(len(test_clean),128,fs,2400,3200)
    test_noisy = add_nosie(test_clean,test_noise,5)
    sf.write("test_noisy.wav",test_noisy,fs)
    

    S_test_noisy = librosa.stft(test_noisy,
                                n_fft=para_wiener["n_fft"], 
                                hop_length=para_wiener["hop_length"], 
                                win_length=para_wiener["win_length"])
    S_test_enhec = S_test_noisy*H
    test_enhenc = librosa.istft(S_test_enhec, 
                                hop_length=para_wiener["hop_length"], 
                                win_length=para_wiener["win_length"])
     
    sf.write("enhce_3.wav",test_enhenc,fs)

    
    plt.subplot(3,1,1)
    plt.specgram(test_clean,NFFT=256,Fs=fs)
    plt.xlabel("clean specgram")
    plt.subplot(3,1,2)
    plt.specgram(test_noisy,NFFT=256,Fs=fs)
    plt.xlabel("noisy specgram")   
    plt.subplot(3,1,3)
    plt.specgram(test_enhenc,NFFT=256,Fs=fs)
    plt.xlabel("enhece specgram")  
    plt.show()
    
    
    