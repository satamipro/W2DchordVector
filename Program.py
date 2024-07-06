import os;
import json;
import op_json;
import keras;
import numpy as np;


######################################################################################################################
#/makedatasetで得られたコード進行のデータセット
#/*/chordList/
#/*/teachData/
#/*/trainData/
#/*/vocabulary.json
#を用いて和音としてのベクトルを作成する
######################################################################################################################
#ここに読込先フォルダの設定（dataset内のフォルダ名を変数に代入すること！）
foldername = "test1"
######################################################################################################################

#datasetの読み込み
traindata = op_json.readAllJsonFile("dataset/" + foldername + "/trainData")
teachdata = op_json.readAllJsonFile("dataset/" + foldername + "/teachdata")
rawdata = op_json.readAllJsonFile("dataset/" + foldername + "/chordList")
#datasetのボキャブラリーの読み込み, 作成
chordlist = op_json.readAllJsonFile("dataset/" + foldername)
vocabulary = set(chordlist[0])
#print(dataset[10])

#datasetから入力と正解データ作成
train_data = []
teach_data1 = []
teach_data2 = []
teach_data3 = []
teach_data4 = []
for chords in traindata:
    for i in range(len(chords) - 4):
        train_data.append(chords[i + 2])

for teaches in teachdata:
    for i in range(len(teaches) - 4):
        teach_data1.append(teaches[i])
        teach_data2.append(teaches[i + 1])
        teach_data3.append(teaches[i + 3])
        teach_data4.append(teaches[i + 4])
train_data = np.array(train_data)
teach_data1 = np.array(teach_data1)
teach_data2 = np.array(teach_data2)
teach_data3 = np.array(teach_data3)
teach_data4 = np.array(teach_data4)


#各レイヤーの設定
inputs = keras.Input(shape=(12,))
x = keras.layers.Dense(512, activation='linear')(inputs)
output1 = keras.layers.Dense(len(vocabulary), activation='softmax')(x)
output2 = keras.layers.Dense(len(vocabulary), activation='softmax')(x)
output3 = keras.layers.Dense(len(vocabulary), activation='softmax')(x)
output4 = keras.layers.Dense(len(vocabulary), activation='softmax')(x)

#モデルを定義
model = keras.Model(inputs = inputs, outputs = [output1, output2, output3, output4])

#損失関数の定義
model.compile(optimizer="rmsprop", loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[0.2, 0.3, 0.3, 0.2])

#学習
model.fit(train_data, [teach_data1, teach_data2, teach_data3, teach_data4], epochs = 10, batch_size = 128)

model.save("Mrs. GREEN APPLE.keras")