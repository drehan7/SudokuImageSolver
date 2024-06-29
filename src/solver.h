#ifndef SUDOKU_SOLVER_H
#define SUDOKU_SOLVER_H
#include <vector>
#include <string>
#include <cassert>
#include <set>
#include <unordered_map>
#include <map>

int sub(int x, int y);

constexpr int MAXROWCOL = 9;

typedef std::vector<std::vector<char>> BoardVec;

class SudokuSolver {
    private:
        void printBoard( BoardVec& boardVec );
        bool IsPossiblePlacement( BoardVec& board, int row, int col, char value );
        bool solveSudoku( BoardVec& board );
    public:
        BoardVec convertToBoardVec( std::string& board );
        std::string convertBoardToString( BoardVec& board ) const ;
        BoardVec validate( std::string& board );

        std::string solve( std::string& board );

        bool compareBoardVec( BoardVec& a, BoardVec& b );
};
#endif // SUDOKU_SOLVER_H
