import socket 
import json 

class GodotEnv:
    def __init__(self,host = "127.0.0.1", port = 9999):
        self.sock = socket.create_connection((host,port))
        self.buffer = b""
    
    def _send(self, msg:dict):
        self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
    
    def _recv(self) -> dict:
        while b"\n" not in self.buffer:
            self.buffer += self.sock.recv(4096)
        line, _, self.buffer = self.buffer.partition(b"\n")
        return json.loads(line.decode("utf-8"))
    
    def reset(self):
        self._send({"cmd": "reset"})
        data = self._recv()
        return tuple(data["state"])
    
    def step(self, action:int):
        self._send({"cmd":"step","action":action})
        data = self._recv()
        return tuple(data["state"]),data["reward"],data["done"],data["score"]

    def close(self):
        self.sock.close()