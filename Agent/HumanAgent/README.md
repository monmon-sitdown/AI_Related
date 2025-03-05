# AI Avatar Chatbot

## Overview

This project is a web-based AI-powered avatar chatbot that utilizes Microsoft's Cognitive Services Speech SDK. The chatbot is capable of processing text input, generating responses, and synthesizing speech with an animated avatar.

## Features

- Text-to-Speech (TTS): Converts AI responses into natural-sounding speech.

- Speech Synthesis with Avatar: Uses Microsoft's Avatar Services for realistic AI-driven conversations.

- Multi-Voice Selection: Supports different English voices (UK, US, and Indian English).

- Real-time Conversation Display: Shows chat history in an interactive UI.

- WebRTC-based Streaming: Uses WebRTC for real-time avatar video and audio streaming.

## Prerequisites

To run this project, you need:

- A valid Microsoft Cognitive Services Speech SDK Subscription Key

- A web server to serve the HTML file (e.g., local server or cloud hosting)

- A backend chat service at http://0.0.0.0:8000/chat (modify if needed)

## Installation & Usage

- Set Up the Subscription Key

- Replace the subscriptionKey variable in the script with your valid Microsoft Cognitive Services Speech SDK Subscription Key.

- Run a Local Server
  If using Python, start a simple HTTP server:

```
python -m http.server 8000
```

Then open http://localhost:8000/ in your browser.

## Interacting with the Avatar

- Type your message in the input box and click Send.

- The AI will generate a response and speak it through the avatar.

- Conversation history will be displayed in the chat box.

## Code Structure

### HTML & JavaScript:

- Loads the Speech SDK and initializes WebRTC.

- Sends user input to an AI backend for response generation.

- Updates the chat UI dynamically.

### Microsoft Speech SDK Integration:

- Configures speech synthesis and avatar settings.

- Uses WebRTC to stream video and audio.

### Backend Communication:

- Uses XMLHttpRequest to fetch AI-generated responses from http://0.0.0.0:8000/chat.

### AI module

AI assistant is based on FastAPI and LangChain, capable of interacting with users in natural language and providing various emotion modes to enhance the interaction experience.

- Supports natural language interaction.

- Includes multiple emotion modes such as friendly, excited, and angry.

- Retains memory of past conversations with users.

- Uses search engines to query unknown information.

- Provides API endpoints for integration with other applications.

### Future Improvements

- Enhance emotion recognition accuracy for better user engagement.

- Optimize performance to reduce response time.

- Add more custom tools to increase usability.
