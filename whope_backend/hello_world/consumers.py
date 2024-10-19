import json
from channels.generic.websocket import AsyncWebsocketConsumer
from huggingface_hub import InferenceClient
from django.conf import settings

token = settings.HF_TOKEN

client = InferenceClient(api_key=token)

class HelloWorldConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()  

    async def disconnect(self, close_code):
        pass  

    async def receive(self, text_data):
        answer: str = ""
        for message in client.chat_completion(
	        model="mistralai/Mistral-Nemo-Instruct-2407",
	        messages=[{"role": "user", "content": text_data}],
        	max_tokens=50,
	        stream=True,
        ):
            answer += message.choices[0].delta.content
        await self.send(text_data=answer)
