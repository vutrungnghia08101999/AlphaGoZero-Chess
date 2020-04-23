package rules;

import chessobjects.Board;
import utils.Spot;
import java.util.ArrayList;

public abstract class AbstractRules {
    public abstract ArrayList<Spot> getInfluenceSpots(Spot start, Board board);
    public abstract ArrayList<Spot> getReachedSpots(Spot start, Board board);

    // This function is used 100% for Queen, Knight, Bishop, Rook, King and 0% for pawn.
    protected ArrayList<Spot> influenceToReachedSpots(Spot start, ArrayList<Spot> influenceSpots, Board board){
        int row = start.getRow();
        int col = start.getCol();
        assert board.board[row][col] != null;
        int team = board.board[row][col].getTeam();

        ArrayList<Spot> reachedSpots = new ArrayList<Spot>();
        for(Spot spot: influenceSpots){
            int r = spot.getRow();
            int c = spot.getCol();
            if(board.board[r][c] != null)
                if(board.board[r][c].getTeam() == team)
                    continue;
            reachedSpots.add(spot);
        }
        return reachedSpots;
    }
}