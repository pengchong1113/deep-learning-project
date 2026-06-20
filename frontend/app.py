import requests
import streamlit as st

API_URL = 'http://127.0.0.1:8000/predict'

ACTION_EMOJI = {
    'squat':          '🏋️',
    'pushup':         '💪',
    'situp':          '🤸',
    'pullup':         '🙌',
    'jumping_jacks':  '⭐',
    'jump_rope':      '🪢',
    'bench_press':    '🏋️',
    'clean_and_jerk': '🥇',
}

st.set_page_config(page_title='Fitness Assistant', page_icon='💪', layout='centered')

st.title('💪 Fitness Assistant')
st.caption('Upload a workout video and the AI will identify the exercise.')

uploaded = st.file_uploader('Choose a video file', type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded:
    st.video(uploaded)

    if st.button('Analyze', type='primary', use_container_width=True):
        with st.spinner('Analyzing your workout...'):
            uploaded.seek(0)
            response = requests.post(API_URL, files={'file': (uploaded.name, uploaded, uploaded.type)})

        if response.status_code == 200:
            result = response.json()
            action     = result['action']
            confidence = result['confidence']
            probs      = result['probabilities']

            st.divider()

            emoji = ACTION_EMOJI.get(action, '🏃')
            st.markdown(f'## {emoji} {action.replace("_", " ").title()}')

            col1, col2 = st.columns(2)
            col1.metric('Confidence', f'{confidence:.1%}')
            col2.metric('Reps counted', result.get('reps', 0))

            st.subheader('All predictions')
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            for label, prob in sorted_probs:
                st.progress(prob, text=f'{label.replace("_", " ").title()}  {prob:.1%}')

        else:
            st.error(f'Error: {response.json().get("detail", "Unknown error")}')
