# # robot_assistant.py
# import streamlit as st
# from streamlit_lottie import st_lottie
# from gtts import gTTS
# import json
# import os

# # 1. Load robot animation
# def load_robot_animation():
#     with open("robot.json", "r", encoding="utf-8") as f:
#         return json.load(f)

# # 2. Dynamic messages for different ML steps
# def get_robot_message(context):
#     messages = {
#         "upload": "You've uploaded your dataset — great! Think of it as your source of truth.",
#         "select_features": "Now, choose which columns will help the model make predictions — these are features.",
#         "select_target": "Pick the target — this is what you're trying to predict!",
#         "train_model": "The model is learning patterns in the data — this is called training.",
#         "metrics": "These numbers tell us how well your model did. Higher is usually better!",
#         "predict": "Prediction time! Let's see what the model has learned.",
#         "test_model": "You're testing the model on unseen data — like a final exam.",
#         "distribution": "Distribution helps you understand how your data is spread — very useful!",
#     }
#     return messages.get(context, "Hi, I'm Robo! Let's learn ML together step-by-step.")

# # 3. Audio support
# def speak_text(text, filename="robot_message.mp3"):
#     tts = gTTS(text)
#     tts.save(filename)
#     with open(filename, "rb") as f:
#         st.audio(f.read(), format="audio/mp3")

# # 4. Display robot assistant
# def show_robot(context, show_audio=True):
#     lottie = load_robot_animation()
#     message = get_robot_message(context)

#     col1, col2 = st.columns([1, 2])
#     with col1:
#         st_lottie(lottie, height=100, key=f"robot_{context}")
#     with col2:
#         st.markdown(f"**🤖 Robo says:** {message}")
#         if show_audio and st.button("🔊 Hear Robo", key=f"audio_{context}"):
#             speak_text(message)
# import streamlit as st
# from streamlit_lottie import st_lottie
# import json
# import time

# def load_robot_animation():
#     with open("robot.json", "r", encoding="utf-8") as f:
#         return json.load(f)


# import streamlit as st
# from streamlit_lottie import st_lottie
# import json
# import time

# def load_robot_animation():
#     with open("robot.json", "r", encoding="utf-8") as f:
#         return json.load(f)

# def get_robot_message(context):
#     messages = {
#         "upload": "You've uploaded your dataset — great!",
#         "select_features": "Now, choose the features.",
#         "train_model": "The model is learning patterns in the data.",
#         "metrics": "These metrics tell how well your model performed.",
#         "predict": "Let’s see what the model can predict!",
#     }
#     return messages.get(context, "Hi, I'm Robo! Let's learn step by step.")

# def show_robot(context, talking_effect=True, auto_disappear=True, delay=4):
#     animation = load_robot_animation()
#     message = get_robot_message(context)

#     # Create a placeholder for animation and message
#     robot_container = st.empty()

#     with robot_container.container():
#         col1, col2 = st.columns([1, 2])
#         with col1:
#             st_lottie(animation, height=120, key=f"robot_{context}")
#         with col2:
#             st.markdown("### 🤖 Robo says:")
#             message_placeholder = st.empty()

#             if talking_effect:
#                 typed = ""
#                 for c in message:
#                     typed += c
#                     message_placeholder.markdown(f"`{typed}_`")
#                     time.sleep(0.03)
#             else:
#                 message_placeholder.markdown(f"**{message}**")

#     # ⏳ Wait and remove
#     if auto_disappear:
#         time.sleep(delay)
#         robot_container.empty()  # Robot disappears
# import streamlit as st
# from streamlit_lottie import st_lottie
# from gtts import gTTS
# import json
# import time
# import os

# def load_robot_animation():
#     with open("robot.json", "r", encoding="utf-8") as f:
#         return json.load(f)

# def get_robot_message(context):
#     messages = {
#         "upload": "upload your dataset",
#         "select_features": "Now, choose the features.",
#         "train_model": "The model is learning patterns in the data.",
#         "metrics": "These metrics tell how well your model performed.",
#         "predict": "Let’s see what the model can predict!",
#     }
#     return messages.get(context, "Hi, I'm Robo! Let's learn step by step.")

# # 🗣 Text-to-speech using gTTS
# def speak_text(text, filename="robot_audio.mp3"):
#     tts = gTTS(text)
#     tts.save(filename)
#     with open(filename, "rb") as f:
#         st.audio(f.read(), format="audio/mp3")

# def show_robot(context, talking_effect=True, auto_disappear=True, delay=5, play_audio=True):
#     animation = load_robot_animation()
#     message = get_robot_message(context)

#     # Display in placeholder
#     robot_container = st.empty()

#     with robot_container.container():
#         col1, col2 = st.columns([1, 2])
#         with col1:
#             st_lottie(animation, height=100, speed=0.3, key=f"robot_{context}")  # 👈 Slowed animation
#         with col2:
#             st.markdown("### 🤖 Robo says:")
#             message_placeholder = st.empty()

#             if talking_effect:
#                 typed = ""
#                 for c in message:
#                     typed += c
#                     message_placeholder.markdown(f"`{typed}_`")
#                     time.sleep(0.03)
#             else:
#                 message_placeholder.markdown(f"**{message}**")

#             if play_audio:
#                 speak_text(message)

#     if auto_disappear:
#         time.sleep(delay)
#         robot_container.empty()
# import streamlit as st
# from streamlit_lottie import st_lottie
# import json
# import time

# # Load the robot animation (Lottie JSON)
# def load_robot_animation():
#     with open("robot.json", "r", encoding="utf-8") as f:
#         return json.load(f)

# # Step-by-step assistant messages
# def get_robot_message(context):
#     messages = {
#         "upload": "You've uploaded your dataset — great job!",
#         "select_features": "Now, choose which columns should be used as input features.",
#         "select_target": "Pick the target column — this is what the model will try to predict.",
#         "train_model": "The model is now learning patterns from your data.",
#         "metrics": "Here’s how your model performed! Check the accuracy and loss values.",
#         "predict": "Time to predict new values — let’s see what the model learned!",
#         "test_model": "You're testing the model on unseen data — like a final exam!",
#     }
#     return messages.get(context, "🤖 Hi! I'm Robo, your ML guide. Let’s learn step-by-step!")

# # The main assistant function
# def show_robot(context, typing_effect=True, auto_disappear=False, delay=6, speed=0.3):
#     if st.button("🤖 Ask Robo", key=f"robo_btn_{context}"):
#         animation = load_robot_animation()
#         message = get_robot_message(context)

#         # Display in temporary container
#         container = st.empty()

#         with container.container():
#             col1, col2 = st.columns([1, 2])

#             with col1:
#                 st_lottie(animation, height=120, speed=speed, key=f"robot_{context}")  # Slowed animation

#             with col2:
#                 st.markdown("### 🤖 Robo says:")
#                 message_placeholder = st.empty()

#                 if typing_effect:
#                     typed = ""
#                     for c in message:
#                         typed += c
#                         # Use a chat bubble style with `unsafe_allow_html`
#                         bubble = f"""
#                         <div style='padding:10px; background-color:#e6f7ff;
#                         border-radius:10px; border:1px solid #b3d7ff;
#                         font-family:monospace; font-size:16px'>
#                         {typed}_</div>
#                         """
#                         message_placeholder.markdown(bubble, unsafe_allow_html=True)
#                         time.sleep(0.03)
#                 else:
#                     message_placeholder.markdown(f"**{message}**")

#         if auto_disappear:
#             time.sleep(delay)
#             container.empty()

# import streamlit as st
# from streamlit_lottie import st_lottie
# import json
# import time

# # Load the robot animation (Lottie JSON)
# @st.cache_data
# def load_robot_animation():
#     with open("robot.json", "r", encoding="utf-8") as f:
#         return json.load(f)

# # Step-by-step assistant messages
# def get_robot_message(context):
#     messages = {
#         "upload": "You've uploaded your dataset — great job!",
#         "select_features": "Now, choose which columns should be used as input features.",
#         "select_target": "Pick the target column — this is what the model will try to predict.",
#         "train_model": "The model is now learning patterns from your data.",
#         "metrics": "Here’s how your model performed! Check the accuracy and loss values.",
#         "predict": "Time to predict new values — let’s see what the model learned!",
#         "test_model": "You're testing the model on unseen data — like a final exam!",
#     }
#     return messages.get(context, "🤖 Hi! I'm Robo, your ML guide. Let’s learn step-by-step!")

# # Typing effect function
# def typing_effect(message, key):
#     message_placeholder = st.empty()

#     # Check if we already started typing this message
#     if f"typed_len_{key}" not in st.session_state:
#         st.session_state[f"typed_len_{key}"] = 0

#     typed_len = st.session_state[f"typed_len_{key}"]

#     if typed_len < len(message):
#         # Increase the number of characters typed
#         st.session_state[f"typed_len_{key}"] += 1
#         typed_text = message[:st.session_state[f"typed_len_{key}"]]
#         bubble = f"""
#         <div style='padding:10px; background-color:#222831;
#                     border-radius:10px; border:1px solid #393E46;
#                     color: #EEEEEE;
#                     font-family:monospace; font-size:16px'>
#             {typed_text}_
#         </div>
#         """
#         message_placeholder.markdown(bubble, unsafe_allow_html=True)
#         time.sleep(0.05)
#         st.rerun()
#     else:
#         # Final complete message
#         typed_text = message
#         bubble = f"""
#         <div style='padding:10px; background-color:#222831;
#                     border-radius:10px; border:1px solid #393E46;
#                     color: #EEEEEE;
#                     font-family:monospace; font-size:16px'>
#             {typed_text}
#         </div>
#         """
#         message_placeholder.markdown(bubble, unsafe_allow_html=True)
#         # Optional: Reset for next time
#         st.session_state[f"typed_len_{key}"] = 0


# # The main assistant function
# def show_robot(context, typing_effect_enabled=True):
#     # Use a session state flag to control when to show the message
#     if f"show_robot_{context}" not in st.session_state:
#         st.session_state[f"show_robot_{context}"] = False

#     if st.button("🤖 Ask Robo", key=f"robo_btn_{context}"):
#         st.session_state[f"show_robot_{context}"] = True

#     if st.session_state[f"show_robot_{context}"]:
#         animation = load_robot_animation()
#         message = get_robot_message(context)

#         container = st.empty()
#         with container.container():
#             col1, col2 = st.columns([1, 2])
#             with col1:
#                 st_lottie(animation, height=120, speed=0.3, key=f"robot_{context}")
#             with col2:
#                 st.markdown("### 🤖 Robo says:")
#                 if typing_effect_enabled:
#                     typing_effect(message, key=context)
#                 else:
#                      bubble = f"""
#         <div style='padding:10px; background-color:#222831;
#                     border-radius:10px; border:1px solid #393E46;
#                     color: #EEEEEE;
#                     font-family:monospace; font-size:16px'>
#             {message}
#         </div>
#         """
#                 st.markdown(bubble, unsafe_allow_html=True)
# import streamlit as st
# from streamlit_lottie import st_lottie
# import json
# import time

# # Load animation
# def load_robot_animation():
#     with open("robot.json", "r", encoding="utf-8") as f:
#         return json.load(f)

# # Get message for the robot
# def get_robot_message(context):
#     messages = {
#         "upload": "You've uploaded your dataset — great job!",
#         "select_features": "Now, choose which columns should be used as input features.",
#         "select_target": "Pick the target column — this is what the model will try to predict.",
#         "train_model": "The model is now learning patterns from your data.",
#         "metrics": "Here’s how your model performed! Check the accuracy and loss values.",
#         "predict": "Time to predict new values — let’s see what the model learned!",
#         "test_model": "You're testing the model on unseen data — like a final exam!",
#     }
#     return messages.get(context, "🤖 Hi! I'm Robo, your ML guide. Let’s learn step-by-step!")

# # Show robot assistant
# def show_robot(context, typing_effect=True, auto_disappear=False, delay=6, speed=0.3):
#     if st.button("🤖 Ask Robo", key=f"robo_btn_{context}"):
#         animation = load_robot_animation()
#         message = get_robot_message(context)

#         container = st.empty()
#         with container.container():
#             col1, col2 = st.columns([1, 2])

#             with col1:
#                 st_lottie(animation, height=120, speed=speed, key=f"robot_{context}")

#             with col2:
#                 st.markdown("### 🤖 Robo says:")
#                 message_placeholder = st.empty()

#                 if typing_effect:
#                     typed = ""
#                     for c in message:
#                         typed += c
#                         bubble = f"""
#                         <div style='padding:10px; background-color:#222831;
#                                     border-radius:10px; border:1px solid #393E46;
#                                     color: #EEEEEE;
#                                     font-family:monospace; font-size:16px'>
#                             {typed}_
#                         </div>
#                         """
#                         message_placeholder.markdown(bubble, unsafe_allow_html=True)
#                         time.sleep(0.03)
#                 else:
#                     bubble = f"""
#                     <div style='padding:10px; background-color:#222831;
#                                 border-radius:10px; border:1px solid #393E46;
#                                 color: #EEEEEE;
#                                 font-family:monospace; font-size:16px'>
#                         {message}
#                     </div>
#                     """
#                     message_placeholder.markdown(bubble, unsafe_allow_html=True)

#         if auto_disappear:
#             time.sleep(delay)
#             container.empty()
import streamlit as st
from gtts import gTTS
import os
import base64
import time
from streamlit_lottie import st_lottie
import json

# Folder to cache generated audio
AUDIO_CACHE_DIR = "audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Load the robot Lottie animation once (replace 'robot.json' with your path)
def load_robot_animation():
    with open("robot.json", "r", encoding="utf-8") as f:
        return json.load(f)

ROBOT_MESSAGES = {
    "upload": "You've uploaded your dataset — great job!",
    "select_features": "Now, choose which columns should be used as input features.",
    "select_target": "Pick the target column — this is what the model will try to predict.",
    "train_model": "The model is now learning patterns from your data.",
    "metrics": "Here’s how your model performed! Check the accuracy and loss values.",
    "predict": "Time to predict new values — let’s see what the model learned!",
    "test_model": "You're testing the model on unseen data — like a final exam!",
}
DEFAULT_MESSAGE = "Hi! I'm Robo, your ML guide. Let’s learn step-by-step!"

def get_robot_message(context):
    return ROBOT_MESSAGES.get(context, DEFAULT_MESSAGE)

def generate_audio(message, filename):
    if not os.path.exists(filename):
        tts = gTTS(message)
        tts.save(filename)

def play_audio(audio_path, muted):
    if muted:
        return
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio autoplay controls style="width:100%;" >
      <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
      Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def show_robot(context):
    if "mute_audio" not in st.session_state:
        st.session_state["mute_audio"] = False

    # Layout: Mute toggle + robot animation + message
    col_mute, col_robot, col_msg = st.columns([1, 1, 4])

    with col_mute:
        st.session_state["mute_audio"] = st.checkbox("🔊 Mute Audio", value=st.session_state["mute_audio"])

    animation = load_robot_animation()
    with col_robot:
        st_lottie(animation, height=150, key=f"robot_anim_{context}")

    with col_msg:
        st.markdown("### Robo says:")

        message = get_robot_message(context)

        # Generate audio file path
        safe_filename = context.replace(" ", "_").lower() + ".mp3"
        audio_path = os.path.join(AUDIO_CACHE_DIR, safe_filename)
        generate_audio(message, audio_path)

        # Start playing audio (only if not muted)
        # This will cause audio to start immediately.
        # Typing effect will run alongside.
        if not st.session_state["mute_audio"]:
            play_audio(audio_path, muted=False)

        # Typing effect
        typed = ""
        message_placeholder = st.empty()

        # Typing speed in seconds per char (adjust for sync)
        typing_speed = 0.04

        # Show typing character by character
        for c in message:
            typed += c
            bubble_html = f"""
            <div style='padding:10px; background-color:#222831;
                        border-radius:10px; border:1px solid #393E46;
                        color: #EEEEEE;
                        font-family: monospace; font-size:16px;
                        white-space: pre-wrap;'>
                {typed}_
            </div>
            """
            message_placeholder.markdown(bubble_html, unsafe_allow_html=True)
            time.sleep(typing_speed)

