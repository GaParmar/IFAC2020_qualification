import os
import json
import time
import asyncio

import requests
from tornado.ioloop import IOLoop
from tornado.web import Application, RedirectHandler, StaticFileHandler, \
    RequestHandler
from tornado.httpserver import HTTPServer
import tornado.gen
import tornado.websocket
from socket import gethostname

class RemoteWebServer():
    '''
    A controller that repeatedly polls a remote webserver and expects
    the response to be angle, throttle and drive mode.
    '''

    def __init__(self, remote_url, connection_timeout=.25):

        self.control_url = remote_url
        self.time = 0.
        self.angle = 0.
        self.throttle = 0.
        self.mode = 'user'
        self.recording = False
        # use one session for all requests
        self.session = requests.Session()

    def update(self):
        '''
        Loop to run in separate thread the updates angle, throttle and
        drive mode.
        '''

        while True:
            # get latest value from server
            self.angle, self.throttle, self.mode, self.recording = self.run()

    def run_threaded(self):
        '''
        Return the last state given from the remote server.
        '''
        return self.angle, self.throttle, self.mode, self.recording

    def run(self):
        '''
        Posts current car sensor data to webserver and returns
        angle and throttle recommendations.
        '''

        data = {}
        response = None
        while response is None:
            try:
                response = self.session.post(self.control_url,
                                             files={'json': json.dumps(data)},
                                             timeout=0.25)

            except requests.exceptions.ReadTimeout as err:
                print("\n Request took too long. Retrying")
                # Lower throttle to prevent runaways.
                return self.angle, self.throttle * .8, None

            except requests.ConnectionError as err:
                # try to reconnect every 3 seconds
                print("\n Vehicle could not connect to server. Make sure you've " +
                    "started your server and you're referencing the right port.")
                time.sleep(3)

        data = json.loads(response.text)
        angle = float(data['angle'])
        throttle = float(data['throttle'])
        drive_mode = str(data['drive_mode'])
        recording = bool(data['recording'])

        return angle, throttle, drive_mode, recording

    def shutdown(self):
        pass


class LocalWebController(tornado.web.Application):

    def __init__(self, port=8887, mode='user'):
        '''
        Create and publish variables needed on many of
        the web handlers.
        '''

        print('Starting Donkey Server...', end='')

        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = mode
        self.recording = False
        self.port = port

        self.num_records = 0
        self.wsclients = []


        handlers = [
            (r"/", RedirectHandler, dict(url="/drive")),
            (r"/drive", DriveAPI),
            (r"/wsDrive", WebSocketDriveAPI),
            (r"/wsCalibrate", WebSocketCalibrateAPI),
            (r"/calibrate", CalibrateHandler),
            (r"/video", VideoAPI),
            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path}),
        ]

        settings = {'debug': True}
        super().__init__(handlers, **settings)
        print("... you can now go to {}.local:8887 to drive "
              "your car.".format(gethostname()))

    def update(self):
        ''' Start the tornado webserver. '''
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.listen(self.port)
        IOLoop.instance().start()

    def run_threaded(self, img_arr=None, num_records=0):
        self.img_arr = img_arr
        self.num_records = num_records

        # Send record count to websocket clients
        if (self.num_records is not None and self.recording is True):
            if self.num_records % 10 == 0:
                for wsclient in self.wsclients:
                    try:
                        data = {
                            'num_records': self.num_records
                        }
                        wsclient.write_message(json.dumps(data))
                    except:
                        pass

        return self.angle, self.throttle, self.mode, self.recording

    def run(self, img_arr=None):
        self.img_arr = img_arr
        return self.angle, self.throttle, self.mode, self.recording

    def shutdown(self):
        pass


class DriveAPI(RequestHandler):

    def get(self):
        data = {}
        self.render("templates/vehicle.html", **data)

    def post(self):
        '''
        Receive post requests as user changes the angle
        and throttle of the vehicle on a the index webpage
        '''
        data = tornado.escape.json_decode(self.request.body)
        self.application.angle = data['angle']
        self.application.throttle = data['throttle']
        self.application.mode = data['drive_mode']
        self.application.recording = data['recording']


class CalibrateHandler(RequestHandler):
    """ Serves the calibration web page"""
    async def get(self):
        await self.render("templates/calibrate.html")


class WebSocketDriveAPI(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        # print("New client connected")
        self.application.wsclients.append(self)

    def on_message(self, message):
        data = json.loads(message)

        self.application.angle = data['angle']
        self.application.throttle = data['throttle']
        self.application.mode = data['drive_mode']
        self.application.recording = data['recording']

    def on_close(self):
        # print("Client disconnected")
        self.application.wsclients.remove(self)


class WebSocketCalibrateAPI(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        print("New client connected")

    def on_message(self, message):
        print(f"wsCalibrate {message}")
        data = json.loads(message)
        if 'throttle' in data:
            print(data['throttle'])
            self.application.throttle = data['throttle']

        if 'angle' in data:
            print(data['angle'])
            self.application.angle = data['angle']

        if 'config' in data:
            config = data['config']
            if self.application.drive_train_type == "SERVO_ESC":
                if 'STEERING_LEFT_PWM' in config:
                    self.application.drive_train['steering'].left_pulse = config['STEERING_LEFT_PWM']

                if 'STEERING_RIGHT_PWM' in config:
                    self.application.drive_train['steering'].right_pulse = config['STEERING_RIGHT_PWM']

                if 'THROTTLE_FORWARD_PWM' in config:
                    self.application.drive_train['throttle'].max_pulse = config['THROTTLE_FORWARD_PWM']

                if 'THROTTLE_STOPPED_PWM' in config:
                    self.application.drive_train['throttle'].zero_pulse = config['THROTTLE_STOPPED_PWM']

                if 'THROTTLE_REVERSE_PWM' in config:
                    self.application.drive_train['throttle'].min_pulse = config['THROTTLE_REVERSE_PWM']

            elif self.application.drive_train_type == "MM1":
                if 'MM1_STEERING_MID' in config:
                    self.application.drive_train.STEERING_MID = config['MM1_STEERING_MID']
                if 'MM1_MAX_FORWARD' in config:
                    self.application.drive_train.MAX_FORWARD = config['MM1_MAX_FORWARD']
                if 'MM1_MAX_REVERSE' in config:
                    self.application.drive_train.MAX_REVERSE = config['MM1_MAX_REVERSE']


    def on_close(self):
        print("Client disconnected")


class VideoAPI(RequestHandler):
    '''
    Serves a MJPEG of the images posted from the vehicle.
    '''

    async def get(self):

        self.set_header("Content-type",
                        "multipart/x-mixed-replace;boundary=--boundarydonotcross")

        served_image_timestamp = time.time()
        my_boundary = "--boundarydonotcross\n"
        while True:

            interval = .01
            if served_image_timestamp + interval < time.time() and \
                    hasattr(self.application, 'img_arr'):

                img = utils.arr_to_binary(self.application.img_arr)
                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(img)
                served_image_timestamp = time.time()
                try:
                    await self.flush()
                except tornado.iostream.StreamClosedError:
                    pass
            else:
                await tornado.gen.sleep(interval)


class BaseHandler(RequestHandler):
    """ Serves the FPV web page"""
    async def get(self):
        data = {}
        await self.render("templates/base_fpv.html", **data)


class WebFpv(Application):
    """
    Class for running an FPV web server that only shows the camera in real-time.
    The web page contains the camera view and auto-adjusts to the web browser
    window size. Conjecture: this picture up-scaling is performed by the
    client OS using graphics acceleration. Hence a web browser on the PC is
    faster than a pure python application based on open cv or similar.
    """

    def __init__(self, port=8890):
        self.port = port
        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')

        """Construct and serve the tornado application."""
        handlers = [
            (r"/", BaseHandler),
            (r"/video", VideoAPI),
            (r"/static/(.*)", StaticFileHandler,
             {"path": self.static_file_path})
        ]

        settings = {'debug': True}
        super().__init__(handlers, **settings)
        print("Started Web FPV server. You can now go to {}.local:{} to "
              "view the car camera".format(gethostname(), self.port))

    def update(self):
        """ Start the tornado webserver. """
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.listen(self.port)
        IOLoop.instance().start()

    def run_threaded(self, img_arr=None):
        self.img_arr = img_arr

    def run(self, img_arr=None):
        self.img_arr = img_arr

    def shutdown(self):
        pass