# Voice-classification-with-machine-learn
This is an ANN (artificial neural network) application created in Python to detect COVID-19 patients based on their voices such as breathing, coughing, and counting.
This link will direct access to training data.. (https://drive.google.com/drive/folders/1rXiyTjjdZW0udE_2YP3qzwkpr2IOM40n?usp=share_link)

There are four varieties of voices, each with a positive and healthy tone (SPEECH COUNTING,COUGH,SPEECH VOWELS,BREATHING). Positive voices begin with the letter P, whereas healthy voices begin with the letter H. There are two types of cough voice files available: cough high (CH) and cough slow (CS) (CS). Normal counting and (SCN) and speed counting (SCS) voices are available for speech counting. The training data contains three vowel files: A - SVA, E - SVE, and O - SVO. There are two types of breath files available: deep breath (BD) and slow breath (SB) (BS).

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
LABELING - Integer numbers were utilized for labeling. There are several methods for labeling the training data files. You can add 1 for healthy files and 2 for negative files to boost accuracy but restrict the number of classifiers. You may provide a varied amount of labels to each speech type, which allows you to identify distinct voice kinds but reduces the model's accuracy. The effective method is to categorize only the primary types of voices as healthy and positive, rather than the subcategories. A labeling example is shown below.

------------------------------------------
HEALTHY_SPEECH_COUNTING_Normal - HSCN - 1
------------------------------------------
HEALTHY_SPEECH_COUNTING_Fast - HSCF - 1
------------------------------------------
POSITIVE_SPEECH_COUNTING_Normal - PSCN - 2
------------------------------------------
POSITIVE_SPEECH_COUNTING_Fast - PSCF - 2
------------------------------------------
HEALTHY_COUGH_High - HCH - 3
------------------------------------------
HEALTHY_COUGH_Slow - HCS - 3
------------------------------------------
POSITIVE_COUGH_High - PCH - 4
------------------------------------------
POSITIVE_COUGH_Slow - PCS - 4
------------------------------------------
HEALTHY_SPEECH_VOWELS_A - HSVA - 5
------------------------------------------
HEALTHY_SPEECH_VOWELS_E - HSVE - 5
------------------------------------------
HEALTHY_SPEECH_VOWELS_O - HSVO - 5
------------------------------------------
POSITIVE_SPEECH_VOWELS_A - PSVA - 6
------------------------------------------
POSITIVE_SPEECH_VOWELS_E - PSVE - 6
------------------------------------------
POSITIVE_SPEECH_VOWELS_O - PSVO - 6
------------------------------------------
BREATHING_HEALTHY_Deep - HBD -7
------------------------------------------
BREATHING_HEALTHY_Slow - HBS -7 
------------------------------------------
BREATHING_POSITIVE_Deep - PBD - 8
------------------------------------------
BREATHING_POSITIVE_Slow - PBS - 8
------------------------------------------

If you utilize various types of labels for all categories and sub-categories, you may add only one or two file paths to list X (list with mypaths) and not use voices in the labeling process. This will improve accuracy, but you will need to run many models to categorize voices.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Data preprocessing:
Noise reduction will be performed on all files using the noisereduce package.
The Mel-Frequency Cepstral Coefficients will be used to extract the features (MFCC)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

After preprocessing the data, the independent and dependent datasets were dumped as pickle files using the pickle program.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

The Neural model was created utilizing the Tensorflow package. Scikit learn was used to split the data into testing and training parts. The model's running time will differ depending on the PC type and labeling technique used. Feel free to experiment with various methods. When the model run is finished, it is stored as an H5 file. Running the model only once is sufficient, and the stored h5 file may be used to predict the files.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Use predict.py to predict the new voices. Add testing files to input folder. 
