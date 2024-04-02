import pandas as pd
import numpy as np 
from konlpy.tag import Komoran
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import koreanize_matplotlib
from PIL import Image

def NLP_PreProcess(df, col_name, n=2, wc=False, filename="./test.csv"):
    '''
    NLP_PreProcess 함수는 텍스트 데이터에 대한 전처리를 수행

    Parameters:
    - df (DataFrame): 전처리할 데이터프레임
    - col_name (str): 텍스트 데이터가 있는 열의 이름
    - n (int): 추출할 형태소의 최소 길이. 2
    - wc (bool, optional): WordCloud를 생성할지 여부. False
    - filename (str, optional): 결과를 저장할 파일경로, "./test.csv"

    Returns:
    - vocab (dict): 형태소와 빈도수 딕셔너리 

    Example:
    ```python
    import pandas as pd

    # 데이터 불러오기
    df = pd.read_csv("data.csv")

    # 전처리 수행
    vocab = NLP_PreProcess(df, "text", 2, wc=True, filename="./processed_data.csv")
    '''
    df=df.dropna().reset_index(drop=True) # 0.결측치 제거
    
    hangule_patten="[^가-힣]"  # 1.한글만 추출
    
    df[col_name]=df[col_name].str.replace(pat=hangule_patten, repl=" ", regex=True) # 2.특정 컬럼의 한글만 추출
    
    df[col_name]=df[col_name].str.replace(pat="^ +", repl="") # 3.긴 공백 제거
    
    df=df[df[col_name].str.strip() != ''].reset_index(drop=True) # 4. 공백만 있는 행 제거
    
    komoran=Komoran() # 5.형태소 분석

    vocab=dict()
    for idx in range(df.shape[0]): # 6. 형태소 분석 시작 
        result =  komoran.morphs(df[col_name][idx])
        for word in result:
            if len(word)>=n: # N 글자 이상만 받음
                if vocab.get(word) is None:
                    vocab[word]=1
                else :
                    vocab[word]+=1

    df2=pd.DataFrame(list(vocab.items()), columns=["word", "freq"])
    
    global stop_words
    df2=df2[~ df2["word"].isin(stop_words)].reset_index(drop=True) # 7. 불용어 제거
    
    df2.sort_values(by="freq", ascending=False, inplace=True) # 8. 빈도순으로 정렬
    
    df2.to_csv(filename, index=False) # 9. 파일 csv로 저장 

    vocab = dict(zip(df2['word'], df2['freq']))
    
    
    if wc: # 10. WordCloud 생성 
        wordcloud = WordCloud(font_path=r'c:\Windows\Fonts\HMFMPYUN.TTF', background_color='white', colormap='Set2').generate_from_frequencies(vocab)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    
    return vocab


def NLP_makeDict(vocab):
    '''
    vocab 딕셔너리를 기반으로 단어 사전을 생성

    Parameters:
    - vocab (dict): 형태소와 빈도수로 이루어진 딕셔너리

    Returns:
    - VOCAB_DICT (dict): 단어와 인덱스로 이루어진 사전

    Example:
    ```python
    vocab = {'영화': 53383, '재밌': 8706, '연기': 7158, ...}
    VOCAB_DICT = NLP_makeDict(vocab)
    ```
    '''
    data={
    'word': vocab.keys(),
    'freq': vocab.values()
    }
    df=pd.DataFrame(data) # 데이터 프레임 생성 
    VOCAB_DICT={0:'<UNK>', 1:'<PAD>'}
    for idx in range(df.shape[0]):
        VOCAB_DICT[idx+2]=df.iloc[idx][0] # 데이터 사전 생성
    return VOCAB_DICT



def NLP_encoding(tokens, VOCAB_DICT):
    '''
    주어진 토큰들을 단어 사전을 이용하여 인덱스로 변환

    Parameters:
    - tokens (list): 인덱스로 변환할 토큰들의 리스트
    - VOCAB_DICT (dict): 단어와 인덱스로 이루어진 사전

    Returns:
    - indexes (list): 인덱스로 변환된 토큰들의 리스트

    Example:
    ```python
    tokens = ['영화', '재밌', '연기', ...]
    indexes = NLP_encoding(tokens, VOCAB_DICT)
    ```
    '''
    indexes = []
    for token in tokens:
        if token in VOCAB_DICT.values():
            indexes.append(list(VOCAB_DICT.keys())[list(VOCAB_DICT.values()).index(token)])
        else:
            indexes.append(0)
    return indexes
    
    
    
def NLP_padding(indexes, max_len):
    '''
    주어진 인덱스들을 패딩하여 max_len 길이로 변환

    Parameters:
    - indexes (list): 패딩할 인덱스들의 리스트
    - max_len (int): 패딩할 길이

    Returns:
    - padded_indexes (list): 패딩된 인덱스들의 리스트

    Example:
    ```python
    indexes = [1, 2, 3, 4, ...]
    padded_indexes = NLP_padding(indexes, 100)
    ```
    '''
    padded_indexes = indexes[:max_len]
    if len(padded_indexes) < max_len:
        padded_indexes += [1] * (max_len - len(padded_indexes))
    return padded_indexes


def NLP_decoding(data, VOCAB_DICT):
    '''
    주어진 데이터를 단어로 변환

    Parameters:
    - data (list): 디코딩할 데이터

    Returns:
    - decoded_data (list): 디코딩된 데이터

    Example:
    ```python
    data = [1, 2, 3, 4, ...]
    decoded_data = NLP_decoding(data)
    ```
    '''
    decoded_data = []
    for idx in data:
        decoded_data.append(VOCAB_DICT[idx])
    return decoded_data
    
    
stop_words = [
    "너무", "는데", "정말", "으로", "ㄴ다", "어요", "진짜", "에서", "네요", "지만", "아니",
    "만들", "아서", "나오", "ㅂ니다", "이런", "습니다", "보다", "까지", "어서", "그냥",
    "이렇", "아도", "ㄴ데", "이것", "ㄴ가", "라고", "다시", "면서", "모르", "보이",
    "이건", "다고", "으면", "완전", "스럽", "다는", "하나", "라는", "정도", "아야",
    "그렇", "에게", "다가", "아요", "부터", "그리고", "는지", "이영화", "이나", "때문",
    "대하", "ㄴ지", "ㄹ까", "어야", "무슨", "없이", "은데", "다니", "보고", "가장",
    "어리", "필요", "ㄴ듯", "끝나", "을까", "위하", "아라", "어도", "우리", "가지",
    "어떻", "모든", "자체", "아직", "하지만", "처럼", "빠지", "는다", "아주", "전혀",
    "시키", "이제", "어디", "내내", "다면", "이거", "이랑", "모습", "그래도", "근데",
    "미치", "누구", "건지", "ㄴ다는", "대단", "그런", "같이", "그것", "조금", "요즘",
    "이다", "라도", "만큼", "특히", "죽이", "제대로", "았었", "한테", "구나", "니까",
    "라면", "이유", "제일", "나름", "무엇", "려고", "차라리", "절대", "이란", "으나",
    "너무나", "훌륭", "그러", "그리", "마다", "도대체", "밖에", "던데", "그저", "다르",
    "어라", "지나", "라니", "생각나", "엄청", "멋있", "더라", "함께", "오늘", "결국",
    "이야", "보니", "제발", "아무리", "여기", "완벽", "진정", "접하", "ㄴ걸", "남기",
    "만나", "어떤", "ㄴ다면", "ㄴ다고", "더니", "나요", "매우", "어쩌", "이리", "하네",
    "보지", "라서", "터지", "그래서", "짜리", "자기", "터지", "약간", "대체", "드리",
    "그나마", "어느", "동안", "그대로", "훨씬", "길래", "돌리", "마라", "려는", "는가",
    "아야지", "대로", "인가", "지도", "으니", "엄청나", "ㄹ수록", "ㄹ지", "에요", "그때",
    "한편", "갑자기", "그러나", "잖아", "따위", "이러", "너무너무", "나서", "거나",
    "그런지", "어이", "구만", "으면서", "충분히", "던가", "으시", "도록", "ㄹ려고",
    "ㄹ게", "또한", "니다", "달리", "으며", "괜히", "더욱", "다루", "오히려", "가요",
    "한번", "로서", "가요", "말이", "보아주", "는구나", "수도", "간다", "ㅂ시다", "어찌",
    "랄까", "많이"
]