import socket, json, time

sock = socket.create_connection(("127.0.0.1", 9999))

# Reset first
sock.sendall((json.dumps({"cmd": "reset"}) + "\n").encode())
time.sleep(0.3)
print("RESET:", sock.recv(4096).decode())

# Step a few times — action 2 = move right
for i in range(5):
    sock.sendall((json.dumps({"cmd": "step", "action": 2}) + "\n").encode())
    time.sleep(0.1)
    print(f"STEP {i}:", sock.recv(4096).decode())

sock.close()