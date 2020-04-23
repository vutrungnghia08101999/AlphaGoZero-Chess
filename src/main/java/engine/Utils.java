package engine;

import chessobjects.*;
import rules.*;
import utils.Move;
import utils.Spot;

import java.util.ArrayList;

public class Utils {

    public static Spot getKingPosition(Board board, int team){
        for(int row = 1; row <= 8; ++row){
            for(int col = 1; col <= 8; ++col){
                if(board.board[row][col] != null){
                    if(board.board[row][col].getTeam() == team && board.board[row][col] instanceof King)
                        return new Spot(row, col);
                }
            }
        }
        return new Spot(-1, -1);
    }

    public static boolean isContainedSpot(ArrayList<Spot> arr, Spot spot){
        for(Spot s: arr){
            if(spot.equals(s))
                return true;
        }
        return false;
    }

    public static boolean isContainedMove(ArrayList<Move> arr, Move mv){
        for(Move move: arr){
            if(move.equals(mv))
                return true;
        }
        return false;
    }

    public static AbstractRules chooseRules(Piece piece){
        assert piece != null;
        if (piece instanceof King)
            return new KingRules();
        else if(piece instanceof Queen)
            return new QueenRules();
        else if(piece instanceof Rook)
            return new RookRules();
        else if(piece instanceof Knight)
            return new KnightRules();
        else if(piece instanceof Bishop)
            return new BishopRules();
        else
            return new PawnRules();
    }

    public static boolean isInCenter(Spot spot){
        int row = spot.getRow();
        int col = spot.getCol();
        assert row >= 1 && row <= 8;
        assert col >= 1 && col <= 8;

        if(row < 4 || row > 5)
            return false;

        if(col < 4 || col > 5)
            return false;
        return true;
    }

    public static boolean isAdjacentOrTheSame(Spot s1, Spot s2){
        int x1 = s1.getRow();
        int y1 = s1.getCol();
        int x2 = s2.getRow();
        int y2 = s2.getCol();
        return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) <= 2;
    }

    public static boolean isWeakPawn(Spot spot, Board board){
        int row = spot.getRow();
        int col = spot.getCol();
        assert board.board[row][col] instanceof Pawn;

        Piece piece = board.board[row][col];
        int team = piece.getTeam();
        if(team == 0){
            for(int r = 2; r <= row - 1; ++r)
                if(board.board[r][col] instanceof Pawn)
                    return true;
            int startRow = Math.max(2, row-1);
            int endRow = Math.min(8, row + 1);
            for(int r = startRow; r <= endRow; ++r){
                if(col-1 >= 1){
                    if(board.board[r][col-1] instanceof Pawn)
                        if(board.board[r][col-1].getTeam() == team)
                            return false;
                }
                if(col+1 <= 8){
                    if(board.board[r][col+1] instanceof Pawn)
                        if(board.board[r][col+1].getTeam() == team)
                            return false;
                }
            }
        }
        else{
            for(int r = 7; r >= row + 1; --r)
                if(board.board[r][col] instanceof Pawn)
                    return true;
            int startRow = Math.min(7, row+1);
            int endRow = Math.max(1, row-1);
            for(int r = startRow; r >= endRow; --r){
                if(col-1 >= 1){
                    if(board.board[r][col-1] instanceof Pawn)
                        if(board.board[r][col-1].getTeam() == team)
                            return false;
                }
                if(col+1 <= 8){
                    if(board.board[r][col+1] instanceof Pawn)
                        if(board.board[r][col+1].getTeam() == team)
                            return false;
                }
            }
        }

        return true;
    }
}
