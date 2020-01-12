import socket


server_receive = socket.socket()
ip_port = ('10.168.44.40', 45678) # your server

server_receive.bind(ip_port)
server_receive.listen(5)
conn, addr = server_receive.accept()

while(True):
	
	data = conn.recv(1024)
	if not data:
		break
	elif len(data) == 0:
		continue
	else:
		print(str(data, encoding='utf-8'))
	msg = 'received'
	
	if(len(data) == 0):
		continue
	conn.sendall(bytes(msg, encoding='utf-8'))
