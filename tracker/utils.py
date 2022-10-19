import os
import sys

import numpy as np
import pandas as pd

from contextlib import contextmanager
from collections import Counter
from misc.utils import flatten
from scipy.spatial.distance import cdist
from misc.log_utils import log


class Track:

    def __init__(self, person_id):
        #list of tupe (frame_id, person_id, x, y)
        self.first_frame = None
        self.last_frame = None
        self.detections = dict()
        self.person_id = person_id

       
    def add_detection(self, frame_id, x, y):
        if self.first_frame is None:
            self.first_frame = frame_id

        self.last_frame = frame_id
        self.detections[frame_id] = (x,y)#, 'Confidence':1.0})

    def get_track_as_list_of_dict(self):
        track_as_list = [{'FrameId':k, 'Id':int(self.person_id), 'X':int(v[0]), 'Y':int(v[1])} for k,v in self.detections.items()]
        
        return track_as_list

    def get_frame_id(self):
        return list(self.detections.keys())

    def extend_track(self, track):
        for k, v in track.detections.items():
            if self.first_frame is None:
                self.first_frame = k
            
            assert k not in self.detections
            self.detections[k] = v

            self.last_frame = k

    def is_matching(self, potential_following_track, dist_threshold=4):
        if self.last_frame + 1 == potential_following_track.first_frame:
            previous_coord = np.array(self.detections[self.last_frame]).reshape((1,2))
            next_coord = np.array(potential_following_track.detections[potential_following_track.first_frame]).reshape((1,2))
            dist = cdist(previous_coord, next_coord, metric='chebyshev')

            return dist < dist_threshold

        return False

def make_dataframe_from_tracks(track_list):
    
    tracks_as_df = pd.DataFrame(flatten([track.get_track_as_list_of_dict() for track in track_list]))
    # print(tracks_as_df)
    if  tracks_as_df.empty:
        tracks_as_df = pd.DataFrame(columns =['FrameId','Id','X','Y'])
    
    tracks_as_df = tracks_as_df.set_index(['FrameId', 'Id'])
    
    return tracks_as_df


def get_nb_det_per_frame_from_tracks(track_list):
    frame_ids = flatten([track.get_frame_id() for track in track_list])
    id_to_count = Counter(frame_ids)

    return id_to_count


from contextlib import contextmanager
import ctypes
import io

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')


@contextmanager
def suppress_stdout():
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.flush()
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        with open(os.devnull, "w") as devnull:
            _redirect_stdout(devnull.fileno())
            # Yield to caller, then redirect stdout back to the saved fd
            yield
            _redirect_stdout(saved_stdout_fd)
            # Copy contents of temporary file to the given stream
    finally:
        os.close(saved_stdout_fd)