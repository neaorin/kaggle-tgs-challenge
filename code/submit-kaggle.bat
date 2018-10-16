set filepath=%1
for %%a in (%filepath%) do echo %%a

kaggle competitions list
REM kaggle competitions submit -f %1 -c tgs-salt-identification-challenge -m "model-tgs-salt-validation-dropout-batchnorm-34-0.10.h5" 