import numpy as np
import pandas as pd
import time

def scale(data_input,mode="Normalize"):
    cols = len(data_input[0,:])
    data_input = np.float64(data_input)
    if mode == "Normalize":
        for i in range(cols):
            m = max(data_input[:,i])
            if m != 0: data_input[:,i] = np.divide(data_input[:,i],m)
    elif mode == "Standardize":
        for i in range(cols):
            if data_input[:, i].std() != 0:
                data_input[:, i] = np.divide(np.subtract(data_input[:, i], data_input[:, i].mean()),data_input[:, i].std())
            else:
                data_input[:, i] = np.subtract(data_input[:, i], data_input[:, i].mean())
    elif mode == "Centering":
        for i in range(cols):
            data_input[:, i] = np.subtract(data_input[:, i], data_input[:, i].mean())
    else:
        print("The mode you have entered is invalid. Try using 'Normalize' or 'Standardize' or 'Centering'")
        exit(1)
    return data_input

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


testDF = pd.read_csv("NoAccelNoCurrentTestAll.csv")
test_input = testDF.iloc[:,1:].values
test_input = scale(test_input, 'Standardize')
temp = test_input[:,0]
press = test_input[:,1]
hum = test_input[:,2]
sound = test_input[:,3]
timestamp = test_input[:,4]
true_result = testDF.iloc[:,0].values

right = 0
start = time.time()
for i in range(len(true_result)):
    Neural1_layer2 = 0.5219297 * temp[i] + 1.1985515 * press[i] - 0.32658446 * hum[i] + 0.07809316 * sound[i] - 0.01790042 * timestamp[i] - 0.5064135
    Neural2_layer2 = -0.9297212 * temp[i] - 0.1042193 * press[i] + 0.5804942 * hum[i] - 0.8308036 * sound[i] + 0.23824215 * timestamp[i] + 0.2101235
    Neural3_layer2 = -0.3037513 * temp[i] - 0.32876235 * press[i] - 0.8970543 * hum[i] + 0.32326183 * sound[i] + 0.73865795 * timestamp[i] + 0.05686649
    Neural4_layer2 = 0.29055202 * temp[i] - 0.19304761 * press[i] - 0.63902634 * hum[i] + 0.9791804 * sound[i] - 0.01591054 * timestamp[i] + 0.22175243
    Neural5_layer2 = 0.30917552 * temp[i] + 0.65579593 * press[i] + 0.07333318 * hum[i] + 0.01643576 * sound[i] - 0.5754305 * timestamp[i] - 0.42291284
    Neural6_layer2 = -0.74081516 * temp[i] - 1.0749621 * press[i] + 0.65805316 * hum[i] + 0.19255795 * sound[i] + 0.39901167 * timestamp[i] + 0.526442
    Neural7_layer2 = -0.9228999 * temp[i] - 1.1690848 * press[i] + 0.62744766 * hum[i] - 0.02623451 * sound[i] + 0.72200435 * timestamp[i] + 0.5198917
    Neural8_layer2 = -0.4856309 * temp[i] - 0.27591345 * press[i] - 0.80711395 * hum[i] + 0.5263337 * sound[i] + 0.20445614 * timestamp[i] - 0.10508852
    Neural9_layer2 = -0.5586461 * temp[i] - 0.24561171 * press[i] + 0.9844569 * hum[i] - 0.2957947 * sound[i] - 0.665909 * timestamp[i] - 0.09850075
    Neural1_layer3 = 1.1360074 * Neural1_layer2 - 0.7688577 * Neural2_layer2 - 0.32050905 * Neural3_layer2 - 0.32635334 * Neural4_layer2 + 1.1923513 * Neural5_layer2 - 1.4149119 * Neural6_layer2 - 1.1214967 * Neural7_layer2 - 0.16015287 * Neural8_layer2 + 0.3824427 * Neural9_layer2 - 0.52454656
    Neural2_layer3 = 0.7709873 * Neural1_layer2 - 0.75005496 * Neural2_layer2 + 0.62775373 * Neural3_layer2 + 0.05030813 * Neural4_layer2 - 0.07055124 * Neural5_layer2 - 0.31771618 * Neural6_layer2 - 1.0686375 * Neural7_layer2 + 0.6201567 * Neural8_layer2 - 0.6942603 * Neural9_layer2 - 0.15837345
    Neural3_layer3 = 0.6660685 * Neural1_layer2 + 0.586423 * Neural2_layer2 - 0.5620778 * Neural3_layer2 - 0.7453912 * Neural4_layer2 + 0.3927375 * Neural5_layer2 - 0.22027971 * Neural6_layer2 - 0.66568375 * Neural7_layer2 + 0.19405743 * Neural8_layer2 - 0.03692146 * Neural9_layer2 - 0.15855362
    Neural4_layer3 = 0.4841159 * Neural1_layer2 - 0.8886254 * Neural2_layer2 + 0.0847089 * Neural3_layer2 + 0.9160161 * Neural4_layer2 - 0.07918422 * Neural5_layer2 - 0.6344456 * Neural6_layer2 - 0.19246802 * Neural7_layer2 + 0.24520394 * Neural8_layer2 - 0.9537495 * Neural9_layer2 - 0.0345268
    Neural1_layer4 = 0.0413318 * Neural1_layer3 - 1.4737732 * Neural2_layer3 + 0.7426425 * Neural3_layer3 - 0.99867165 * Neural4_layer3 + 0.08084776
    Neural2_layer4 = 2.145323 * Neural1_layer3 + 0.76278967 * Neural2_layer3 + 0.85921395 * Neural3_layer3 + 0.40388995 * Neural4_layer3 - 0.36209032
    Neural3_layer4 = -0.6302475 * Neural1_layer3 - 0.07705893 * Neural2_layer3 - 0.33410388 * Neural3_layer3 + 0.16375706 * Neural4_layer3 + 0.22758816
    Motor = [Neural1_layer4,Neural2_layer4,Neural3_layer4]
    Result = softmax(Motor)
    if Result[0] >= Result[1]:
        if Result[0] >= Result[2]:
            predict = 0
        else:
            predict = 2
    elif Result[1] >= Result[2]:
        predict = 1
    else:
        predict = 2
    if predict == true_result[i]:
        right += 1

end = time.time()
Accuracy_Test = right / len(true_result)
print("Accuracy of test set is %.2f"%(Accuracy_Test*100) , "%" )
print("Total time estimated: {0}".format(end-start))

