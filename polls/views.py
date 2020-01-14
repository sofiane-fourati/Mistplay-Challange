from django.shortcuts import render
from .forms import FilesForm
import pandas as pd
import pickle
import json
import os
# Create your views here.
from django.http import HttpResponse


def index(request):
    form= FilesForm(request.POST or None, request.FILES)
    context= { 'form': form }
    if form.is_valid():
        result = []
        files = request.FILES
        df = pd.read_csv(files["file"], sep=',')
        df = preprocess(df)
        if 'y2' in df.columns:
            df.drop('y2', axis=1, inplace=True)
        if 'y' in df.columns:
            df.drop('y', axis=1, inplace=True)
        model = pickle.load(open("model.sav", 'rb'))
        r = model.predict(df)
        j = 0
        for i in df.index:
            result.append({'id':i, 'Prediction': int(r[j])})
            j += 1
        if not 'predict' in os.listdir():
            os.mkdir("predict")
        output_file = open('predict/result.json', 'w', encoding='utf-8')
        for dic in result:
            json.dump(dic, output_file) 
            output_file.write("\n")
        return render(request, 'index2.html', {'result':result})
    return render(request, 'form.html', context)
	
	
	
	
def preprocess(df):

	columns = ['x9','x10','x11','x12','x13','x14', 'x18','x19','x20','x21','x22','y2','x2_4.4', 'x2_4.4.2','x2_4.4.3','x2_4.4.4','x2_5','x2_5.0.1','x2_5.0.2','x2_5.1','x2_5.1.1','x2_6','x2_6.0.1','x2_7','x2_7.1','x2_7.1.1','x2_7.1.2','x2_8.0.0','x2_8.1.0','x2_9','x4_AT','x4_AT_DE','x4_AU','x4_AU_GB','x4_AU_GB_US','x4_AU_SG','x4_BA','x4_CA','x4_CA_GB_US','x4_CA_NZ_US','x4_CA_US','x4_CH','x4_DE','x4_DE_GB_US','x4_DK','x4_DK_LT','x4_DK_SE','x4_DK_US','x4_DZ','x4_ES','x4_EU_GB','x4_FI','x4_FR','x4_FR_RO','x4_GB','x4_GB_IL','x4_GB_NL_US','x4_GB_SG','x4_GB_US','x4_GH','x4_GR','x4_HK_SG','x4_IE','x4_IN','x4_JP','x4_KR','x4_MX','x4_MX_US','x4_MY','x4_NL','x4_NO','x4_NO_SE','x4_NZ','x4_NZ_US','x4_PE','x4_PL','x4_PT','x4_SA','x4_SE','x4_SG','x4_SG_US','x4_TH','x4_TR','x4_TR_US_VE','x4_US','x4_US_other','x4_VN','x5_0','x5_1','x5_2','x8_0','x8_1','x8_2','x8_3','x16_0','x16_1','x17_4','x17_5','x17_6','x17_7','x17_8','x17_9','x23_FALSE','x23_TRUE','x30_[0,15[','x30_[15,20[','x30_[20,25[','x30_[25,30[','x30_[30,35[','x30_[35,40[','x30_[40,45[','x30_[45,50[','x30_[50,55[','x30_[55,60[','x30_[60,65[','x30_[65,70[','x30_[70,75[','x30_[75,80[','x30_[80,85[','x30_[85,90[','x30_[90,95[','x30_[95,100[']

	df.set_index("x1", inplace=True)
	df.drop("x6", axis=1, inplace=True)
	df.drop("Unnamed: 27", axis=1, inplace=True)
	df.drop("Unnamed: 28", axis=1, inplace=True)
	df["y2"] = 0
	df.loc[df["y"]!=0, "y2"] = 1
	df.drop("x24", axis=1, inplace=True)
	df.drop("x25", axis=1, inplace=True)
	df.drop("x26", axis=1, inplace=True)
	df.drop("y", axis=1, inplace=True)
	df.drop("x3", axis=1, inplace=True)
	df.drop("x15", axis=1, inplace=True)
	df["x30"] = None
	for i in df.index:
		age = df.at[i,"x7"]
		if  age < 15:
			df.at[i,"x30"] = '[0,15['
		else:
			for j in range(15,121,5):
				if age-j <= 5:
					df.at[i,"x30"] = '['+str(j)+','+str(j+5)+'['
					break
	df.drop("x7", axis=1, inplace=True)
	types = df.dtypes
	for i in types.index:
		if types[i] == object:
			dummy = pd.get_dummies(df[i], prefix=i)
			df.drop(i, axis=1, inplace=True)
			df = pd.concat([df, dummy], axis=1)
	for col in df.columns:
		if not col in columns:
			df.drop(col, axis=1, inplace=True)
	return df
