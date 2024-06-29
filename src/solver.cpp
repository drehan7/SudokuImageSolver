#include "solver.h"

bool SudokuSolver::compareBoardVec( BoardVec& a, BoardVec& b )
{
    if ( a.size() != b.size() ) {
        return false;
    }

    for ( int i = 0; i < a.size(); ++i ) {
        if ( a.at(i).size() != b.at(i).size() ) {
            return false;
        }
        for ( int j = 0; j < a.at(i).size(); ++j ) {
            if ( a.at(i).at(j) != b.at(i).at(j) ) {
                return false;
            }
        }
    }

    return true;
}

void SudokuSolver::printBoard( BoardVec& boardVec )
{
    for ( int i = 0; i < boardVec.size(); ++i ) {
        for ( int j = 0; j < boardVec.at(i).size(); ++j ) {
            printf("%c,", boardVec.at(i).at(j) );
        }

        printf("\n");
    }
}

// Takes in 81 char string as a board representation
// Converts to 2dvec board rep.
BoardVec SudokuSolver::convertToBoardVec( std::string& board )
{
    assert( board.size() == 81 );

    BoardVec ret;

    std::vector<char> tmp;

    for ( int i = 0; i < board.size(); ++i ) {
        if ( i!=0 &&  i % 9 == 0 ) 
        {
            ret.push_back( tmp );
            tmp.clear();
        }
        tmp.push_back( board[i] );
    }

    if ( !tmp.empty() ) {
        ret.push_back(tmp);
    }

    return ret;
}

std::string SudokuSolver::convertBoardToString( BoardVec& board ) const
{
    std::string strBoard;

    for ( int i = 0; i < board.size(); ++i ) {
        for ( int j = 0; j < board.size(); ++j ) {
            strBoard += board[i][j];
        }
    }

    return strBoard;
}

// Validate Sudoku board
BoardVec SudokuSolver::validate( std::string& board )
{
    BoardVec boardVec = convertToBoardVec( board );

    constexpr int sz = 9;

    bool row[sz][sz] = {false};
    bool col[sz][sz] = {false};
    bool square[sz][sz] = {false};

    for ( int i = 0; i < boardVec.size(); ++i ) {
        for ( int j = 0; j < boardVec.size(); ++j ) {
            char val = boardVec[i][j];
            if ( val == '.' ) continue;

            int idx = val - '0' - 1; // char to num idx
            int area = ( i / 3 ) * 3 + ( j / 3 );

            // if number already exists
            if ( row[i][idx] || col[j][idx] || square[area][idx] ) {
                return BoardVec();
            }

            row[i][idx] = true;
            col[j][idx] = true;
            square[area][idx] = true;
        }
    }

    return boardVec;
}

bool SudokuSolver::IsPossiblePlacement( BoardVec& board, int row, int col, char value )
{
    // Check row
    auto boardRow = board[row];
    for ( auto& ch : boardRow ) {
        if ( ch == value ) {
            return false;
        }
    }

    // Check col
    for ( int i = 0; i < board.size(); ++i ) {
        if ( board[i][col] == value ) {
            return false;
        }
    }

    // Check square
    // Start idx at top left of subgrid
    int subRow = row/3 * 3;
    int subCol = col/3 * 3;

    for ( int si = subRow; si < subRow+3; ++si ) {
        for ( int sj = subCol; sj < subCol+3; ++sj ) {
            if ( board[si][sj] == value ) {
                return false;
            }
        }
    }

    return true;
}

bool SudokuSolver::solveSudoku( BoardVec& board )
{
    for ( int i = 0; i < board.size(); ++i )
    {
        for ( int j = 0; j < board.size(); ++j )
        {
            char val = board[i][j];

            if ( val == '.' )
            {
                for ( int k = 0; k < board.size(); ++k )
                {
                    char ch = std::to_string((k+1))[0];
                    if ( IsPossiblePlacement(board, i, j, ch))
                    {
                        board[i][j] = ch;
                        if ( solveSudoku( board )) {
                            return true;
                        }
                        board[i][j] = '.';
                    }
                }
                return false;
            }
        }
    }

    return true;
}

std::string SudokuSolver::solve( std::string& board )
{
    BoardVec newBoard = validate( board );
    if ( newBoard.empty() ) {
        return "";
    }

    solveSudoku( newBoard );

    return convertBoardToString( newBoard );
}
