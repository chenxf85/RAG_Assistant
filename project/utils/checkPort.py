import socket
def getValidPort(port:int=11500,max_port:int=11600)->int:
    result=0
    while result == 0 and port < max_port:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        if result == 0:  # 端口被占用，尝试下一个
            port += 1
    if port >= max_port:
        raise RuntimeError("没有找到空闲端口用于启动 Ollama 服务")
    return port