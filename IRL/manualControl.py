"""
Manually control the agent to provide expert trajectories.
The main aim is to get the feature expectaitons of the expert trajectories.

Left arrow key: turn Left
right arrow key: turn right
up arrow key: dont turn, move forward

If curses is used:
down arrow key: exit 
Always exit using down arrow key rather than Ctrl+C or your terminal will be tken over by curses
"""

from simulation import carmunk
import random
from random import randint
import numpy as np
from neuralNets import net1
import msvcrt
#import curses # for keypress, doesn't work for Windows


NUM_FEATURES = 8
GAMMA = 0.9 # the discount factor for RL algorithm

def demo(screen):
    while True:
        screen.print_at('Hello world!',
                        randint(0, screen.width), randint(0, screen.height),
                        colour=randint(0, screen.colours - 1),
                        bg=randint(0, screen.colours - 1))
        ev = screen.get_key()
        if ev in (ord('Q'), ord('q')):
            return
        screen.refresh()
    
def play():
    '''
    The goal is to get feature expectations of a policy. Note that feature expectations are independent from weights.
    '''
    
    game_state = carmunk.GameState()   # set up the simulation environment
    game_state.frame_step((2)) # make a forward move in the simulation
    currFeatureExp = np.zeros(NUM_FEATURES)
    prevFeatureExp = np.zeros(NUM_FEATURES)

    moveCount = 0
    while True:
        moveCount += 1

        # get the actual move from keyboard
        move = msvcrt.getch()
        if move == b'H':   # UP key -- move forward
            action = 2
        elif move == b'K': # LEFT key -- turn left
            action = 1
        elif move == b'M': # RIGHT key -- turn right
            action = 0
        else:
            action = 2

        '''
        # curses
        event = screen.getch()
        if event == curses.KEY_LEFT:
            action = 1
        elif event == curses.KEY_RIGHT:
            action = 0
        elif event == curses.KEY_DOWN:
            break
        else:
            action = 2
        '''

        # take an action 
        # start recording feature expectations only after 100 frames
        _, _, readings = game_state.frame_step(action)

        if moveCount > 100:
            currFeatureExp += (GAMMA**(moveCount-101))*np.array(readings)
            
        # report the change percentage
        changePercentage = (np.linalg.norm(currFeatureExp - prevFeatureExp)*100.0)/np.linalg.norm(currFeatureExp)

        print (moveCount)
        print ("The change percentage in feature expectation is: ", changePercentage)
        prevFeatureExp = np.array(currFeatureExp)

        if moveCount % 2000 == 0:
            break

    return currFeatureExp

if __name__ == "__main__":
    '''
    # Use curses to generate keyboard input
    screen = curses.initscr()
    curses.noecho()
    curses.curs_set(0)
    screen.keypad(1)
    screen.addstr("Play the game")
    curses.endwin()
    '''

    featureExp = play()
    print (featureExp)