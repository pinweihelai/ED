import pandas as pd
from csim_zh import similarity
error_text = pd.read_table("d:/house_error.txt",header=None)
correct_text = pd.read_table('d:/house_correct.txt', header=None)
print(len(error_text[0]))
count = 0
err_count = 0
for i in range(len(error_text[0])):
    distance, changeText = similarity(error_text[0][i])
    if changeText == correct_text[0][i]:
        count= count+1
    if(distance < 0.5):
        err_count = err_count + 1

print(count)
print("相似度低於0.5的個數", err_count)
print("正確率為：",count/(len(error_text[0])*1.0))