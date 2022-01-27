#!/usr/bin/env python3

# GXSM socket client class

import sys
import os		# use os because python IO is bugy
import time
import threading
import re
import socket
import json


############################################################
# Socket Client
############################################################

# defaults as set in GXSM socket server for connection:
#HOST = '127.0.0.1'  # The server's hostname or IP address
#PORT = 65432        # The port used by the server


class SocketClient:
    def __init__(self, host, port):
        self.sok = None
        #with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        #   self.sok = client
        try:
            self.sok=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sok.connect ((host, port))
            self.send_as_json ({'echo': [{'message': 'Hello GXSM3! Establishing Socket Link.'}]})
            data = self.receive_json ()
            print('Received: ', data)
        except:
            pass
            
    def __del__(self):
        if not self.sok:
            print ('No connection')
            raise Exception('You have to connect first before receiving data')
        self.sok.close()

    def send_as_json(self, data):
        if not self.sok:
            print ('No connection')
            raise Exception('You have to connect first before receiving data')
        try:
            serialized = json.dumps(data)
        except (TypeError, ValueError):
            raise Exception('You can only send JSON-serializable data')
        
        print('Sending JSON: N={} D={}'.format(len(serialized.encode('utf-8')),serialized))

        # send the length of the serialized data first
        sd = '{}\n{}'.format(len(serialized), serialized)
        # send the serialized data
        self.sok.sendall(sd.encode('utf-8'))
            
    def request_start_scan(self):
        self.send_as_json({'action': ['start-scan']})
        return self.receive()

    def request_stop_scan(self):
        self.send_as_json({'action': ['stop-scan']})
        return self.receive()

    def request_autosave(self):
        self.send_as_json({'action': ['autosave']})
        return self.receive()

    def request_autoupdate(self):
        self.send_as_json({'action': ['autoupdate']})
        return self.receive()

    def request_action(self, id):
        self.send_as_json({'action': [{'id':id}]})
        return self.receive()

    def request_action_v(self, id, value):
        self.send_as_json({'action': [{'id':id, 'value':value}]})
        return self.receive()

    def request_set_parameter(self, id, value):
        self.send_as_json({'command': [{'set': id, 'value': value}]})
        return self.receive()

    def request_get_parameter(self, id):
        self.send_as_json({'command': [{'get': id}]})
        return self.receive()

    def request_gets_parameter(self, id):
        self.send_as_json({'command': [{'gets': id}]})
        return self.receive()

    def request_query_info(self, x):
        self.send_as_json({'command': [{'query': x}]})
        return self.receive()

    def request_query_info_args(self, x, i=0,j=0,k=0):
        self.send_as_json({'command': [{'query': x, 'args': [i,j,k]}]})
        return self.receive()

    def receive(self):
        if not self.sok:
            print ('No connection')
            raise Exception('You have to connect first before receiving data')
        return self.receive_json()

    def receive_json(self):
        #print ('receive_json...\n')
        if not self.sok:
            print ('No connection')
            raise Exception('You have to connect first before receiving data')
        # try simple assume one message
        try:
            data = self.sok.recv (1024)
            if data:
                #print ('Got Data: {}'.format(data))
                count, jsdata = data.split(b'\n')
                #print ('N={} D={}'.format(count,jsdata))
                try:
                    deserialized = json.loads(jsdata)
                    print ('Received JSON: N={} D={}'.format(count,deserialized))
                    return deserialized
                except (TypeError, ValueError):
                    deserialized = json.loads({'JSON-Deserialize-Error'})
                    raise Exception('Data received was not in JSON format')
                    return deserialized
            else:
                pass
        except:
            pass
        
    def receive_json_long(self, socket):
        # read the length of the data, letter by letter until we reach EOL
        length_str = b''
        print ('Waiting for response.\n')
        char = self.sok.recv(1)
        while char != '\n':
            length_str += char
            char = self.sok.recv(1)
        total = int(length_str)
        #print('receiving json bytes # ', total, ' [', length_str, ']\n')
        # use a memoryview to receive the data chunk by chunk efficiently
        view = memoryview(bytearray(total))
        next_offset = 0
        while total - next_offset > 0:
            recv_size = self.sok.recv_into(view[next_offset:], total - next_offset)
            next_offset += recv_size
        try:
            deserialized = json.loads(view.tobytes())
        except (TypeError, ValueError):
            raise Exception('Data received was not in JSON format')

        print('received JSON: {}\n', deserialized)
        
        return deserialized
    
    def recv_and_close(self):
        data = self.receive()
        self.close()
        return data
    
    def close(self):
        if self.s:
            self.sok.close()
            self.sok = None

############################################################
# END SOCKET
############################################################

