#!/usr/bin/env python3

import argparse

# HELPER PROGRAM FOR SUDOKU STUFF


def convertBoardToString(board):
    pass

def convertStringToBoard(string):
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SudokuHelper", description="Helper for sudoku")
    parser.add_argument("-b", "--board")
    parser.add_argument("-s", "--string")

    args = parser.parse_args()
    
    if args.board:
        convertBoardToString(args.board)
    elif args.string:
        convertStringToBoard(args.string)
    else:
        parser.print_help()
