"""Simple web page to check one SMS with the API.

Run it with:
    streamlit run ui/app.py

The UI does NOT load the model itself. It only calls the API over HTTP.
Why? The model lives in one place (the API). The UI stays small, and other
apps (mobile, other servers) can use the same API.
"""

import os

import requests
import streamlit as st

# Where the API runs. We read it from an environment variable, so the same
# code works on your computer (localhost) and in Docker (http://api:8000).
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="SMS Fraud Checker", page_icon="📱")
st.title("📱 SMS Fraud Checker")
st.caption(f"Model API: {API_URL}")

message = st.text_area(
    "Paste an SMS message:", height=150, placeholder="e.g. You won a free prize! Call now..."
)

if st.button("Check", type="primary"):
    # Check empty input here too, so the user gets a fast, friendly message.
    if not message.strip():
        st.warning("Please type a message first.")
    else:
        try:
            # timeout: never wait forever if the API does not answer.
            response = requests.post(f"{API_URL}/predict", json={"message": message}, timeout=10)
            # raise_for_status turns HTTP errors (4xx, 5xx) into an exception.
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach the API at {API_URL}. Is it running?")
        except requests.exceptions.RequestException as error:
            st.error(f"The API returned an error: {error}")
        else:
            # This block runs only if there was NO error above.
            probability = result["probability"]
            if result["is_fraud"]:
                # st.error shows a red box, st.success a green box.
                st.error(
                    f"🚨 FRAUD — this message looks like spam ({probability:.1%} spam probability)."
                )
            else:
                st.success(
                    f"✅ NORMAL — this message looks safe ({probability:.1%} spam probability)."
                )

            # A bar from 0% to 100% makes the probability easy to see.
            st.progress(probability, text=f"Spam probability: {probability:.1%}")
            st.caption(f"Model version: {result['model_version']}")
