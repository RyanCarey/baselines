from os import path, listdir
import numpy as np
from scipy import stats as st
import math
import cv2

def preprocess(state, resize_shape=(84,84)):
    # Resize state
    state = cv2.resize(state, resize_shape)

    if len(state.shape) == 3:
        if state.shape[2] == 3:
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

    # Check type is compatible
    if state.dtype != np.float32:
        state = state.astype(np.float32)

    # normalize
    if state.max() > 1:
        state *= 1. / 255.

    return state.reshape(-1, 84, 84)


class AtariDataset():

    TRAJS_SUBDIR = 'trajectories'
    SCREENS_SUBDIR = 'screens'

    def __init__(self, data_path, game):
        
        '''
            Loads the dataset trajectories into memory. 
            data_path is the root of the dataset (the folder, which contains
            the 'screens' and 'trajectories' folders. 
        '''

        self.trajs_path = path.join(data_path, AtariDataset.TRAJS_SUBDIR, game)
        self.screens_path = path.join(data_path, AtariDataset.SCREENS_SUBDIR, game)
        self.game = game
    
        #check that the we have the trajs where expected
        assert path.exists(self.trajs_path)
        
        self.trajectories = self.load_trajectories()

    def load_trajectories(self):
        trajectories = {}
        for traj in listdir(self.trajs_path):
            curr_traj = []
            with open(path.join(self.trajs_path, traj)) as f:
                for i,line in enumerate(f):
                    #first line is the metadata, second is the header
                    if i > 1:
                        #TODO will fix the spacing and True/False/integer in the next replay session
                        #frame,reward,score,terminal, action
                        curr_data = line.rstrip('\n').replace(" ","").split(',')
                        if curr_data[3] not in ['True', 'False', '1', '0']: raise ValueError
                        curr_trans = {}
                        curr_trans['frame']    = int(curr_data[0])
                        curr_trans['reward']   = int(curr_data[1])
                        curr_trans['score']    = int(curr_data[2])
                        curr_trans['terminal'] = True if curr_data[3] in ['True', '1']  else False
                        curr_trans['action']   = int(curr_data[4])
                        curr_traj.append(curr_trans)
            trajectories[int(traj.split('.txt')[0])] = curr_traj
        return trajectories
                   

    def compile_data(self, score_lb=0, score_ub=math.inf, max_nb_transitions=None):
        data = []
        shuffled_trajs = np.array(list(self.trajectories.keys()))
        np.random.shuffle(shuffled_trajs)

        for t in shuffled_trajs:
            st_dir   = path.join(self.screens_path, str(t))
            cur_traj = self.trajectories[t]
            cur_traj_len = len(listdir(st_dir))

            # cut off trajectories with final score beyound the limit
            if not score_lb <= cur_traj[-1]['score'] <= score_ub:
                continue

            #we're here if the trajectory is within lb/ub
            for pid in range(0, cur_traj_len):

                #screens are numbered from 1, transitions from 0
                #TODO change screen numbering from zero during next data replay
                state = preprocess(cv2.imread(path.join(st_dir, str(pid) + '.png'), cv2.IMREAD_GRAYSCALE))

                data.append({'action': get_action_name(cur_traj[pid]['action']),
                             'state':  state,
                             'reward': cur_traj[pid]['reward'],
                             'terminal': cur_traj[pid]['terminal'] == 1
                            })

                # if nb_transitions is None, we want the whole dataset limited only by lb and ub
                if max_nb_transitions and len(data) == max_nb_transitions:
                    print("Total frames: %d" % len(dataset))
                    return data

        #we're here if we need all the data
        return data
     
