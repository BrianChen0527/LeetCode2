import time
import datetime
import contextlib
import os
import threading
import sys
import re


def handle_room_message(self, msg):
    """
    Handle user messages sent to a chat room. If msg type is ‘leave_room’ remove
    specified user from the room. If msg type is ‘send_message’ send message to
    all users in the room except the message sender.
    """
    if msg["type"] == "leave_room":
        self.rooms[msg["room_name"]]["members"].remove(msg["username"])
    elif msg["type"] == "send_message":
        for user in self.rooms[msg["room_name"]]["members"]:
            if user == msg["username"]:
                continue
            host, port = self.users[user]
            tcp_client(host, port, msg["text"])


def handle_message(self, msg):
    """
    Handle user messages to the server. If message_type is “register”
    create and keep track of user and the host and port that user is
    listening on. Send register_ack message. If message_type is “join”,
    create room if room_name doesn’t exist and start tcp_server thread
    for that room. Then add a user as a member of the room. Send join_ack message.
    """
    if msg['type'] == 'register':
        host, port = msg['host'], msg['port']
        self.users[msg["username"]] = (host, port)
        tcp_client(host, port, { "type": "register_ack", })
    elif msg['type'] == 'join_room':
        if msg["room_name"] not in self.rooms:
            self.rooms[msg["room_name"]] = {
                'port': self.free_port,
                'members': []
            }
            chat_room = threading.Thread(target=tcp_server,
                                         args=(self.host,
                                               self.free_port,
                                               handle_room_message))
            chat_room.start()
            self.free_port += 1

        self.rooms[msg["room_name"]]['members'].append(msg['username'])

        ack_msg = {
            "type": "join_ack",
            "room_host": self.host,
            "room_port": self.rooms[msg["room_name"]]['port']
        }
        usr_host, usr_port = self.users[msg['username']]
        tcp_client(usr_host, usr_port, ack_msg)

def map1():
    for line in sys.stdin:
        class_name, year, credits, workload, grade = line.strip().split()
        print(f"{class_name}\t{year} {credits} {workload} {grade}")

def reduce1(key, group):
    group = list(group)
    tot_workload, tot_grade = 0, 0
    year, creds = 0, 0
    for G in group:
        year, creds, workload, grade = G.strip().split()
        tot_workload += workload
        tot_grade += grade

    print(f"{key} {year} {creds} {tot_workload/(credits*len(group))} {tot_grade/len(group)}")

def map2():
    classes = {}
    students = []
    for line in sys.stdin:
        terms = line.strip().split()
        if len(terms) == 5:
            class_name, year, creds, workload, grade = terms
            classes[class_name] = [year, creds, workload, grade]
        else:
            student_ID, year, desired_class = terms
            students.append([student_ID, year, desired_class])

    for student in students:
        student_ID, year, desired_class = student
        if desired_class in classes and classes[desired_class][0] <= year:
            print(f"{student_ID}\t{desired_class} {grade} {workload}")

def reduce2(key, group):
    courses = list(group)
    for course in courses:
        desired_class, grade, workload = course
        print(f"{key} {desired_class} {grade} {workload}")

