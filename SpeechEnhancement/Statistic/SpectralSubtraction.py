#Spectral Subtraction
import librosa
from librosa.core.spectrum import amplitude_to_db
import numpy as np
from scipy.signal import lfilter, firwin, freqz
import soundfile as sf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    clean_wav_file = "clean.wav"
    clean,fs = librosa.load(clean_wav_file,sr=None) 
    print(fs)

    noisy_wav_file = "noise.wav"
    noisy,fs = librosa.load(noisy_wav_file,sr=None)
    
    # Spectrum of noise 
    S_noisy = librosa.stft(noisy,n_fft=256, hop_length=128, win_length=256)  # D x T
    D,T = np.shape(S_noisy)
    Mag_noisy= np.abs(S_noisy)
    Phase_nosiy= np.angle(S_noisy)
    Power_nosiy = Mag_noisy**2
    print(fs)

    # Estimate the energy of noise
    # Since it was known, assuming the first 30 frame 
    Mag_nosie = np.mean(np.abs(S_noisy[:,:31]),axis=1,keepdims=True)
    Power_nosie = Mag_nosie**2
    Power_nosie = np.tile(Power_nosie,[1,T])

    ## 1 Directly subtract
    # # Power subtraction
    # Power_enhenc = Power_nosiy-Power_nosie
    # # ensure the energy > 0
    # Power_enhenc[Power_enhenc<0]=0
    # Mag_enhenc = np.sqrt(Power_enhenc)

    # 2 Magnitude subtract
    # Mag_enhenc = np.sqrt(Power_nosiy) - np.sqrt(Power_nosie)
    # Mag_enhenc[Mag_enhenc<0]=0
    
    
    ## 2 Subtract more
    # # parameter
    # alpha = 6
    # gamma = 1

    # Power_enhenc = np.power(Power_nosiy,gamma) - alpha*np.power(Power_nosie,gamma)
    # Power_enhenc = np.power(Power_enhenc,1/gamma)
    
    # # The value which are too small replaced by  beta* Power_nosie 
    # beta = 0.01
    # mask = (Power_enhenc>=beta*Power_nosie)-0
    # print(mask.shape)
    # Power_enhenc = mask*Power_enhenc + beta*(1-mask)*beta*Power_nosie
    
    # Mag_enhenc = np.sqrt(Power_enhenc)
    
    
    ## 3 Smooth
    Mag_noisy_new = np.copy(Mag_noisy)
    k=1
    for t in range(k,T-k):
        Mag_noisy_new[:,t] = np.mean(Mag_noisy[:,t-k:t+k+1],axis=1)
    
    Power_nosiy = Mag_noisy_new**2
    
    # Over-Subtraction Denoising
    alpha = 4
    gamma = 1

    Power_enhenc = np.power(Power_nosiy,gamma) - alpha*np.power(Power_nosie,gamma)
    Power_enhenc = np.power(Power_enhenc,1/gamma)
    
    # The value which are too small replaced by  beta* Power_nosie 
    beta = 0.0001
    mask = (Power_enhenc>=beta*Power_nosie)-0
    Power_enhenc = mask*Power_enhenc + beta*(1-mask)*Power_nosie

    Mag_enhenc = np.sqrt(Power_enhenc)
    
    
    Mag_enhenc_new  = np.copy(Mag_enhenc)
    # Calculate the Maximum Noise Residual
    maxnr = np.max(np.abs(S_noisy[:,:31])-Mag_nosie,axis =1)

    k = 1
    for t in range(k,T-k):
        index = np.where(Mag_enhenc[:,t]<maxnr)[0]
        temp = np.min(Mag_enhenc[:,t-k:t+k+1],axis=1)
        Mag_enhenc_new[index,t] = temp[index]
            
    
    # Recover the signal
    S_enhec = Mag_enhenc_new*np.exp(1j*Phase_nosiy)
    enhenc = librosa.istft(S_enhec, hop_length=128, win_length=256)
    sf.write("enhce_3.wav",enhenc,fs)
    print(fs)

    # plot
   
    plt.subplot(3,1,1)
    plt.specgram(clean,NFFT=256,Fs=fs)
    plt.xlabel("clean specgram")
    plt.subplot(3,1,2)
    plt.specgram(noisy,NFFT=256,Fs=fs)
    plt.xlabel("noisy specgram")   
    plt.subplot(3,1,3)
    plt.specgram(enhenc,NFFT=256,Fs=fs)
    plt.xlabel("enhece specgram")  
    plt.show()
    
    
   
    
   
   
