package rules;

import chessobjects.Board;
import chessobjects.King;
import chessobjects.Rook;
import utils.Spot;

import java.util.ArrayList;

public class KingRules extends AbstractRules {
    @Override
    public ArrayList<Spot> getInfluenceSpots(Spot start, Board board) {
        int row = start.getRow();
        int col = start.getCol();
        assert board.board[row][col] instanceof King;
        ArrayList<Spot> influenceSpots = new ArrayList<Spot>();
        for(int i = row - 1; i <= row + 1; ++i){
            for(int j = col - 1; j <= col + 1; ++j){
                if(i >= 1 && i <= 8 && j >= 1 && j <= 8 && (i != row || j != col))
                    influenceSpots.add(new Spot(i, j));
            }
        }

        return influenceSpots;
    }

    @Override
    public ArrayList<Spot> getReachedSpots(Spot start, Board board) {
        ArrayList<Spot> protectedSpots = getInfluenceSpots(start, board);
        ArrayList<Spot> influenceSpots = super.influenceToReachedSpots(start, protectedSpots, board);
        // castling spots will be add by CPU, because It involve other objects so I don't want to implement here.
        return influenceSpots;
    }
}
