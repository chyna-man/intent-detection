import streamlit as st
from predict_intent import predict_intent
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

st.title("Intent Recognition App")
st.write("Speak or type an intent, and the app will classify it.")

stt_button = Button(label="🎤 Speak", width=100)

stt_button.js_on_event("button_click", CustomJS(code="""
    var recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
 
    recognition.onresult = function (e) {
        var value = "";
        for (var i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) {
                value += e.results[i][0].transcript;
            }
        }
        if ( value != "") {
            document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
        }
    }
    recognition.start();
    """))

result = streamlit_bokeh_events(
    stt_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0
)

user_input = ""
if result and "GET_TEXT" in result:
    user_input = result.get("GET_TEXT")

user_input = st.text_input("Or type your intent:", value=user_input)

if st.button("Analyze"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            prediction = predict_intent(user_input)
            st.success(f"Predicted Intent: {prediction}")
    else:
        st.warning("Please enter or speak an intent.")
