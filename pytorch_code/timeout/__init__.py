import signal
import time

class TimeoutError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super(TimeoutError, self).__init__(message)

class Timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

if __name__ == "__main__":
    try:
        with Timeout(seconds=5):
            for i in range(10):
                print(i)
                time.sleep(1)
    except TimeoutError:
        print("time up, exit here!")