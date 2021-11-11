import pandas as pd


path = 'EmotionsDataSet.csv'

#emo_dict = {'worry':0, 'sadness':1, 'surprise':2, 'love':3, 'neutral':4, 'anger':5}
emo_dict1 = {'neutral':0,
    'worry':1,
    'sad':2,
    'anger':3,
    'surprise':4,
    'love':5,
    'happy':6
    }
    #frustated is not here
def make_label(text):
    return emo_dict1[text]

def truncate_long_sent(DF):
    for i in range(DF.shape[0]):
        d = DF[i]
        d_l = d.split()
        if len(d_l) >= 400:
            d_l = d_l[:40]
            DF[i] = ' '.join(d_l)

def Readcsv(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1','Unnamed: 0.1.1.1'])


    df = df.sample(frac=1).reset_index(drop=True)
    print(df.Emotions.value_counts())

    df.Emotions = df.Emotions.apply(make_label)
    
    df.sw_Text = df.sw_Text.apply(str)
    df.Emotions = df.Emotions.apply(int)

    
    truncate_long_sent(df.sw_Text)

    df = df[['sw_Text','Emotions']].reset_index(drop=True)


    df.Emotions = df.Emotions.apply(make_label)

    return df

df  = Readcsv(path)
