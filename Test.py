"""
Test cases
"""
import Action
import RobState
from Util import *

###########################
###  Test for Sampling  ###
###########################

from Sample import generate_examples

def io_generation_test1():
    generate_examples(1, True)

def io_generation_test2():
    inputs = ['john Smith',
              'DOUG Q. Macklin',
              'Frank Lee (123)',
              'Laura Jane Jones',
              'Steve P. Green (9)']
    actions = [Action.GetToken('Word', -1),
               Action.Commit(),
               Action.ConstStr(','),
               Action.Commit(),
               Action.ConstStr(' '),
               Action.Commit(),
               Action.GetToken('Word', 0),
               Action.ToCase('Proper'),
               Action.Commit()]
    outputs = ['' for _ in inputs]
    print(execute_actions(actions, RobState.RobState.new(inputs, outputs)))

def io_generation_test3():
    pstate = RobState.RobState.new(["12A", "2A4", "A45", "4&6", "&67"],
                                   ["", "", "", "", ""])
    print (pstate)
    fs = [
            Action.ToCase("Lower"),
            Action.Replace("&", "["),
            Action.Substr(1, 2),
            Action.Commit()
            ]

    print(execute_actions(fs, pstate))

def io_generation_test4():
    pstate = RobState.RobState.new(['Mr.Pu', 'Mr.Poo'],
                                   ['', ''])

    fs = [Action.GetToken('Word', 1),
          Action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [Action.GetUpTo('.'),
          Action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [Action.GetFrom('.'),
          Action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [Action.GetFirst('Word', 2),
          Action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [Action.GetAll('Word'),
          Action.Commit()]
    print(execute_actions(fs, pstate))

if __name__ == '__main__':
    io_generation_test4()
