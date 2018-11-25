import socket
import subprocess as proc

def listen():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "127.0.0.1"
    port = 1337
    s.bind( (host, port) )

    s.listen(5)
    print("Listening for connections on " + host + ":" + str(port))
    try:
        while(True):
            c, ip = s.accept()
            print("Connection established from " + str(ip));
            data = c.recv(1024).decode()
            print(data)
            handle(data)
            response = "ACK"
            c.send(response.encode())
    except:
        s.shutdown(socket.SHUT_RDWR)
        s.close()
        c.close()

"""
Handle the command sent from the master
@param data : String - request sent from the master
"""
def handle(data):
    cmd = data.split()
    proc.call(cmd[1:])

def main():
    listen()

if __name__ == "__main__":
    main()
