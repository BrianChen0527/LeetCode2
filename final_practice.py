import time
import datetime
import contextlib
import os
import threading
import sys
import re

class Manager:
    def __init__(self, host, port, client_host, client_port):
        """
        Initialize member variables, start child threads, and
        continuously call self._update every 1 second while a
        shutdown message has not been received. Note: all messages
        sent by the manager should be sent within self._update.
        """
        self.host = host
        self.port = port
        self.client_host = client_host
        self.client_port = client_port
        self.user_location = 0
        self.coffee_pots = []
        self.flags = {'shutdown': False}

        tcp_thread = threading.Thread(target=self.tcp_server,
                                      args=(self.host, self.port, self._handle_tcp_message(), ))

        udp_thread = threading.Thread(target=self.udp_server,
                                      args=(self.msg))

        tcp_thread.start()
        udp_thread.start()

        while not self.flags['shutdown']:
            if self.user_location:
                self._update()
            time.sleep(1)

        tcp_thread.join()
        udp_thread.join()


    def _update(self):
        """
        Send a message to the coffee pot if necessary. Send a message
        to update the smartwatch with the nearest coffee pot location
        with available coffee.
        """
        tcp_server(self.host, self.port, self.handle_tcp_message, msg)

        self.coffee_pots = sorted(self.coffee_pots, key=lambda x: get_distance(x['pot_location'])

        null_msg = {"message_type": "coffee_unavailable"}
        if not self.coffee_pots:
            tcp_client(self.client_host, self.client_port, null_msg)

        coffee_msg = {'message_type': 'make_coffee'}
        closest_pot = self.coffee_pots[0]
        if not closest_pot['available']:
            tcp_client(closest_pot['host'], closest_pot['port'], coffee_msg)

        coffee_msg = {
            "message_type": "coffee_available",
            "longitude": "",
            "latitude": ""}
        no_coffee = True
        for pot in self.coffee_pots:
            if pot['available']:
                coffee_msg['longitude'] = pot['longitude']
                tcp_client(self.client_host, self.client_port, coffee_msg)
                no_coffee = False

        if no_coffee:
            tcp_client(self.client_host, self.client_port, null_msg)


    def _handle_tcp_message(self, msg):
        """
        Handle TCP shutdown messages and TCP messages from coffee
        pots.
        """
        if msg['message_type'] == 'shutdown':
            self.flags['shutdown'] = True
        elif msg['message_type'] == 'register':
            long, lat = msg['longitude'], msg['latitude']
            new_pot = {
                'host': msg['host'],
                'port': msg['port'],
                'longitude': long,
                'latitude': lat,
                'pot_location': (long, lat),
                'status': 'coffee_unavailable'
            }
            self.coffee_pots.append(new_pot)
        elif msg['message_type'] == 'status_update':
            for pot in self.coffee_pots:
                if pot['host'] == msg['host'] and pot['port'] == msg['port']:
                    pot['status'] = msg['status']


    def _handle_udp_message(self, msg):
        """
        Handle UDP location updates from smartwatch.
        """
        self.user_location =  (msg['longitude'], msg['latitude'])


def map1():
    for line in sys.stdin:
        terms = line.strip().split(',')
        user = terms[0]
        media = terms[1].split('/')[1]
        action = terms[1].split('/')[2]
        category = terms[1].split('/')[3]
        response = terms[2]

        if response < 400:
            print(f"{user}\t{media} {action} {category}")

def reduce1(key, group):
    interactions = list(group)

    ad_clicks, ad_views, likes_comments = 0, 0, 0
    categories = {}
    for interaction in interactions:
        i = interaction.strip.split()
        media, action, category = i[0], i[1], i[2]
        if category not in categories:
            categories[category] = {
                'ad_clicks': 0,
                'ad_views': 0,
                'likes_comments': 0
            }
        if media == 'post':
            categories[category]['likes_comments'] += 1
        if media == 'ad':
            if action == 'view':
                categories[category]['ad_views'] += 1
            if action == 'click':
                categories[category]['ad_clicks'] += 1

    category_scores = []
    for c in categories:
        cat = categories[c]
        score = cat['ad_clicks'] + 0.5*cat['ad_views'] + 0.2*cat['likes_comments']
        category_scores.append({
            'category': c,
            'score': score
        })
    bids = get_bids(key, category_scores)
    for bid in bids:
        print(f"{bid['category']}\t{key} {bid['bid']}")


def map2():
    for line in sys.stdin:
        print(line.strip())


def reduce2(key, group)
    group = list(group)
    bids = []
    for g in group:
        user, bid = g.strip().split()
        bids.append({'user': user, 'bid': bid})

    bids = sorted(bids, key=lambda x: (x['bid'], x['user']))
    print(f"{key} won by {bids[-1]['user']} for {bids[-2]['bid']}")

