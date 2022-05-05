
#streamlit 라이브러리를 불러오기
import streamlit as st
#AI모델을 불러오기 위한 joblib 불러오기
import joblib
import pandas as pd

# st를 이용하여 타이틀과 입력 방법을 명시한다.
# Index(['signalstrenth', 'antenacoverage', 'antenalength', 'bandwidth', 'VOC'], dtype='object')
#antenacoverage: 안테나 커버리지 antenalength: 안테나 길이 bandwidth: 밴드범위
def user_input_features() :
  signalstrenth = st.slider("신호강도: ",0,10,1)
  antenacoverage =st.slider("안테나 커버리지: ",0,10,1)
  antenalength = st.slider("안테나 길이: ",0,10,1)
  bandwidth = st.slider("밴드범위: ",0,10,1)
  
  #st.slider('How old are you?', 0, 130, 25)

  data = {'signalstrenth' : [signalstrenth],
          'antenacoverage' : [antenacoverage],
          'antenalength' : [antenalength],
          'bandwidth' : [bandwidth],
          }
  data_df = pd.DataFrame(data, index=[0])
  return data_df

# new_x= {'dist':[0.03], 'office':[10], 'home':[2.22], 'station':[1], 'co2':[0.66], 'room' : [8.33],'age':[23.1], 'pop':[4.11], 'road':[12], 'mange':[323], 'kid':[12.23] }


st.title('네트워크기지국 안테나의 성능에 따른 민원 예측')
st.write('')
st.markdown('* 아래에 데이터를 입력해주세요')

booster.save_model('model_xgb.pkl')

le_voc = joblib.load("le.pkl")
scaler_call = joblib.load("scaler.pkl")
model_call= joblib.load("model_xgb.pkl")

new_x_df = user_input_features()
data_con_scale = scaler_call.transform(new_x_df)
result = model_call.predict(data_con_scale) 

#예측결과를 화면에 뿌려준다. 
st.subheader('결과는 다음과 같습니다.')
st.write('')
st.write('')
st.write('민원 내용 예측:', result[0])
st.caption('0: Sclass')
st.caption('1: Aclass')
st.caption('2: Bclass')
