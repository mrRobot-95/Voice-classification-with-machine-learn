model = tf.keras.models.load_model('model_breath_nr_q.h5')

file="../input"
#preprocess the audio file
for i in file:
    for (dirpath, dirnames, filenames) in walk(i):
        f.extend(filenames)
        break

for filename in f:
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    #Reshape MFCC feature to 2-D array
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    
    #predicted_label=model.predict_classes(mfccs_scaled_features)
    x_predict=model.predict(mfccs_scaled_features)
    predicted_label=np.argmax(x_predict,axis=1)
    print(predicted_label)
    
    prediction_class = labelencoder.inverse_transform(predicted_label)
    print(prediction_class)
