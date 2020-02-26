"""
Test cases
"""
import action
from train import *
from util import *

###########################
###  Test for Sampling  ###
###########################

from sample import generate_examples

def io_generation_test1():
    generate_examples(1, True)

def io_generation_test2():
    inputs = ['john Smith',
              'DOUG Q. Macklin',
              'Frank Lee (123)',
              'Laura Jane Jones',
              'Steve P. Green (9)']
    actions = [action.GetToken('Word', -1),
               action.Commit(),
               action.ConstStr(','),
               action.Commit(),
               action.ConstStr(' '),
               action.Commit(),
               action.GetToken('Word', 0),
               action.ToCase('Proper'),
               action.Commit()]
    outputs = ['' for _ in inputs]
    print(execute_actions(actions, action.RobState.new(inputs, outputs)))

def io_generation_test3():
    pstate = action.RobState.new(["12A", "2A4", "A45", "4&6", "&67"],
                                   ["", "", "", "", ""])
    print (pstate)
    fs = [
            action.ToCase("Lower"),
            action.Replace("&", "["),
            action.Substr(1, 2),
            action.Commit()
            ]

    print(execute_actions(fs, pstate))

def io_generation_test4():
    pstate = action.RobState.new(['Mr.Pu', 'Mr.Poo'],
                                   ['', ''])

    fs = [action.GetToken('Word', 1),
          action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [action.GetUpTo('.'),
          action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [action.GetFrom('.'),
          action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [action.GetFirst('Word', 2),
          action.Commit()]
    print(execute_actions(fs, pstate))

    fs = [action.GetAll('Word'),
          action.Commit()]
    print(execute_actions(fs, pstate))

def io_generation_test5():
    pstate = action.RobState.new(["(hello)1)23", "(mis)ter)123"],
                                   ["HELLO", "MIS"])
    fs = [action.GetSpan("(", 0, "End", ")", 0, "Start"),
          action.ToCase("AllCaps"),
          action.Commit()]

    print(execute_actions(fs, pstate))

def io_generation_test6():
    print(get_supervised_sample())

if __name__ == '__main__':
    io_generation_test6()
