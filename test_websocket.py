import websocket
import threading

condition: threading.Condition() = threading.Condition()

def on_message(ws, message):
    with condition:
        print(f"Received: {message}")
        condition.notify()

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws):
    print("Connection closed")

def run(ws):
    while True:
        user_input: str = input("Enter message to send: ")
        with condition:
            ws.send(user_input)
            condition.wait()

def on_open(ws):
    threading.Thread(target=run, args=(ws,)).start()

if __name__ == "__main__":
    ws = websocket.WebSocketApp("ws://localhost:8000/ws/hello_world",
                                  on_open=on_open,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close)
    ws.run_forever()
