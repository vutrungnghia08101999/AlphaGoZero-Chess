package rules;

import chessobjects.Board;
import chessobjects.Pawn;
import utils.Spot;

import java.util.ArrayList;

public class PawnRules extends AbstractRules {

    @Override
    public ArrayList<Spot> getInfluenceSpots(Spot start, Board board) {
        int row = start.getRow();
        int col = start.getCol();
        assert board.board[row][col] instanceof Pawn;
        ArrayList<Spot> influenceSpots = new ArrayList<Spot>();

        Pawn pawn = (Pawn)board.board[row][col];
        if(pawn.getTeam() == 0){
            if(row + 1 <= 8 && col - 1 >= 1)
                influenceSpots.add(new Spot(row + 1, col - 1));
            if(row + 1 <= 8 && col + 1 <= 8)
                influenceSpots.add(new Spot(row + 1, col + 1));
        }
        else if(pawn.getTeam() == 1){
            if(row - 1 >= 1 && col - 1 >= 1)
                influenceSpots.add(new Spot(row - 1, col - 1));
            if(row - 1 >= 1 && col + 1 <= 8)
                influenceSpots.add(new Spot(row - 1, col + 1));
        }

        return influenceSpots;
    }

    @Override
    public ArrayList<Spot> getReachedSpots(Spot start, Board board) {
        int row = start.getRow();
        int col = start.getCol();
        assert board.board[row][col] instanceof Pawn;
        ArrayList<Spot> reachedSpots = new ArrayList<Spot>();

        Pawn pawn = (Pawn)board.board[row][col];
        //white team, upper team, move down
        if(pawn.getTeam() == 0){
            if(row + 1 <= 8){
                if(board.board[row + 1][col] == null)
                    reachedSpots.add(new Spot(row + 1, col));
            }

            if(row == 2){
                if(board.board[row + 1][col] == null && board.board[row + 2][col] == null)
                    reachedSpots.add(new Spot(row + 2, col));
            }
        }
        //Black team, lower team, move up
        else if(pawn.getTeam() == 1){
            if(row - 1 >= 1){
                if(board.board[row - 1][col] == null)
                    reachedSpots.add(new Spot(row - 1, col));
            }

            if(row == 7){
                if(board.board[row - 1][col] == null && board.board[row - 2][col] == null)
                    reachedSpots.add(new Spot(row - 2, col));
            }
        }

        ArrayList<Spot> influenceSpots = getInfluenceSpots(start, board);
        for(Spot spot: influenceSpots){
            int r = spot.getRow();
            int c = spot.getCol();
            if(board.board[r][c] == null)
                continue;
            if(board.board[r][c].getTeam() == pawn.getTeam())
                continue;
            reachedSpots.add(spot);
        }
        return reachedSpots;
    }
}
